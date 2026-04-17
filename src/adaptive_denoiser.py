"""
Adaptive denoiser for automatic noise-level detection and strategy selection.

Estimates the noise level of a price surface and dispatches to the optimal
SINDy variant (finite differences, Savitzky-Golay, neural derivatives, or
weak-form SINDy) based on the estimated noise regime.
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from src.utils import set_all_seeds, setup_logging

logger = setup_logging(__name__)


def estimate_noise_level(V, S_grid, t_grid):
    """
    Estimate the noise level of a price surface without knowing the clean surface.

    Uses two complementary methods:
    1. Polynomial residual: Fits a degree-4 2D polynomial and measures residual RMS.
    2. Second-difference MAD: Uses median absolute deviation of d²V/dS² to estimate noise.

    Parameters
    ----------
    V : ndarray, shape (n_S, n_t)
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)

    Returns
    -------
    float
        Estimated noise level as a fraction of std(V).
    """
    n_S, n_t = V.shape
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')

    V_std = float(np.std(V))
    if V_std < 1e-15:
        return 0.0

    # Method 1: Polynomial residual
    try:
        S_flat = S_mesh.ravel()
        t_flat = t_mesh.ravel()
        V_flat = V.ravel()

        # Normalize inputs for numerical stability
        S_norm = (S_flat - S_flat.mean()) / (S_flat.std() + 1e-10)
        t_norm = (t_flat - t_flat.mean()) / (t_flat.std() + 1e-10)

        X = np.column_stack([S_norm, t_norm])
        poly = PolynomialFeatures(degree=4, include_bias=True)
        X_poly = poly.fit_transform(X)

        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_poly, V_flat)
        V_pred = reg.predict(X_poly)

        residual_rms = float(np.sqrt(np.mean((V_flat - V_pred) ** 2)))
        est_poly = residual_rms / V_std
    except Exception as e:
        logger.warning(f"Polynomial noise estimation failed: {e}")
        est_poly = None

    # Method 2: Second-difference MAD
    try:
        dS = S_grid[1] - S_grid[0]
        # Second finite difference along S axis (interior only)
        d2V = (V[2:, :] - 2.0 * V[1:-1, :] + V[:-2, :]) / (dS ** 2)

        # MAD of second differences
        mad = float(np.median(np.abs(d2V - np.median(d2V))))

        # noise_std ≈ MAD * h² / 1.4826
        # But d2V = true_d2V + noise_d2V where noise_d2V ~ noise_std * sqrt(6) / h²
        # So MAD(d2V) ≈ 0.6745 * std(noise_d2V) = 0.6745 * noise_std * sqrt(6) / h²
        # => noise_std ≈ MAD * h² / (0.6745 * sqrt(6))
        # Actually, more simply: the MAD of pure noise second differences
        # for noise_std σ is: MAD ≈ 0.6745 * σ * sqrt(6) / h²
        # So σ ≈ MAD * h² / (0.6745 * sqrt(6))
        noise_std_est = mad * dS ** 2 / (0.6745 * np.sqrt(6.0))
        est_mad = noise_std_est / V_std
    except Exception as e:
        logger.warning(f"MAD noise estimation failed: {e}")
        est_mad = None

    # Combine estimates
    estimates = [e for e in [est_poly, est_mad] if e is not None and e >= 0]
    if len(estimates) == 0:
        logger.warning("All noise estimation methods failed, returning 0.0")
        return 0.0

    combined = float(np.mean(estimates))

    logger.info(
        f"Noise estimation: poly={est_poly:.4f}, MAD={est_mad:.4f}, "
        f"combined={combined:.4f}"
    )

    return combined


def select_derivative_strategy(estimated_noise_pct):
    """
    Select the optimal derivative estimation strategy based on estimated noise.

    Empirically calibrated thresholds based on R²(clean) crossover analysis
    on 50×50 grids. R²(clean) measures discovered PDE accuracy against
    analytical derivatives — the honest metric (unlike R²(noisy) which
    misleadingly increases with noise).

    Regimes:
    - noise < 0.005 (0.5%): Finite differences — exact on clean data
    - 0.005 ≤ noise < 0.03 (3%): Savitzky-Golay — classical smoothing
      dominates at moderate noise (R²(clean) ≈ 0.87-0.99)
    - 0.03 ≤ noise < 0.50: Weak SINDy — integral methods avoid
      differentiation entirely, giving R²(clean) ≈ 0.83-0.89
    - noise ≥ 0.50: Unreliable (weak SINDy, but warn)

    Why NOT neural derivatives:
    Neural derivative estimation was hypothesized to outperform finite-
    difference methods by learning a smooth surface approximation. However,
    R²(clean) evaluation reveals that autograd second derivatives of fitted
    neural networks introduce approximation bias that exceeds the noise
    reduction benefit. Even on clean data, the best neural config achieves
    R²(clean) ≈ 0.95 vs 1.00 for FD. Savitzky-Golay smoothing (classical
    signal processing) provides superior noise reduction at moderate levels,
    while weak-form SINDy (integral methods) dominates at high noise by
    avoiding differentiation entirely. This is a publishable negative result:
    the obvious ML solution is beaten by classical methods for second-
    derivative-sensitive PDE discovery.

    Crossover analysis (50×50 grid):
    - FD → SavGol crossover at ~0.5% noise
    - SavGol → Weak crossover at ~2.5-3% noise (SavGol=0.871, Weak=0.891)

    Parameters
    ----------
    estimated_noise_pct : float

    Returns
    -------
    strategy : str
        One of 'fd', 'savgol', 'weak', 'unreliable'.
    params : dict
        Recommended hyperparameters for the selected strategy.
    """
    if estimated_noise_pct < 0.005:
        strategy = 'fd'
        params = {'smooth': False}
    elif estimated_noise_pct < 0.03:
        strategy = 'savgol'
        params = {'smooth': True, 'savgol_window': 21, 'savgol_poly': 5}
    elif estimated_noise_pct < 0.50:
        strategy = 'weak'
        params = {'n_test_functions': 100}
    else:
        strategy = 'unreliable'
        params = {'n_test_functions': 100}
        logger.warning(
            f"Estimated noise {estimated_noise_pct:.1%} is very high. "
            "PDE discovery results may be unreliable."
        )

    logger.info(
        f"Selected strategy: {strategy} "
        f"(estimated noise = {estimated_noise_pct:.4f})"
    )

    return strategy, params


def adaptive_sindy_discover(V, S_grid, t_grid, true_sigma=None, true_r=None,
                             force_strategy=None, seed=42, K=100, T=1.0,
                             option_type='call'):
    """
    Adaptive SINDy: auto-detect noise level and dispatch to optimal method.

    Parameters
    ----------
    V : ndarray, shape (n_S, n_t)
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    true_sigma : float or None
    true_r : float or None
    force_strategy : str or None
        Override auto-detection with a specific strategy.
    seed : int
    K : float
    T : float
    option_type : str

    Returns
    -------
    dict
        Same keys as discover_pde() plus:
        - 'estimated_noise': float
        - 'selected_strategy': str
    """
    from src.sindy_discovery import discover_pde
    from src.neural_derivatives import sindy_with_neural_derivatives
    from src.weak_sindy import weak_sindy_discover

    set_all_seeds(seed)

    # Step 1: Estimate noise
    estimated_noise = estimate_noise_level(V, S_grid, t_grid)

    # Step 2: Select strategy
    if force_strategy is not None:
        strategy = force_strategy
        _, params = select_derivative_strategy(estimated_noise)
        logger.info(f"Forced strategy: {strategy} (estimated noise: {estimated_noise:.4f})")
    else:
        strategy, params = select_derivative_strategy(estimated_noise)

    # Step 3: Dispatch
    if strategy == 'fd':
        result = discover_pde(
            V, S_grid, t_grid, true_sigma=true_sigma, true_r=true_r,
            smooth=False, K=K, T=T, option_type=option_type,
        )
    elif strategy == 'savgol':
        w = params.get('savgol_window', 7)
        p = params.get('savgol_poly', 3)
        result = discover_pde(
            V, S_grid, t_grid, true_sigma=true_sigma, true_r=true_r,
            smooth=True, savgol_window=w, savgol_poly=p,
            K=K, T=T, option_type=option_type,
        )
    elif strategy in ('weak', 'unreliable'):
        n_tf = params.get('n_test_functions', 100)
        result = weak_sindy_discover(
            V, S_grid, t_grid, n_test_functions=n_tf,
            true_sigma=true_sigma, true_r=true_r, seed=seed,
        )
    elif strategy == 'neural':
        ep = params.get('fit_epochs', 1500)
        result = sindy_with_neural_derivatives(
            V, S_grid, t_grid, true_sigma=true_sigma, true_r=true_r,
            fit_epochs=ep, seed=seed, K=K, T=T, option_type=option_type,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Attach adaptive metadata
    result['estimated_noise'] = estimated_noise
    result['selected_strategy'] = strategy

    logger.info(
        f"Adaptive SINDy: noise={estimated_noise:.4f}, strategy={strategy}, "
        f"R²={result['r2_score']:.6f}, active={result['n_active']}"
    )

    return result
