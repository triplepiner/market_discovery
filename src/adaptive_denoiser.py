"""
Adaptive denoiser for automatic noise-level detection and strategy selection.

Estimates the noise level of a price surface and dispatches to the optimal
SINDy variant (finite differences, Savitzky-Golay, GP-derived derivatives,
or weak-form SINDy) based on the estimated noise regime.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from src.utils import set_all_seeds, setup_logging

logger = setup_logging(__name__)


# Empirically calibrated thresholds — see recalibrate_adaptive_with_gp().
# GP first beats SavGol at ~0.5% noise and dominates through ~10% noise,
# then collapses sharply at 15%+ where Weak SINDy takes over.
GP_CROSSOVER = 0.01      # SavGol -> GP at 1% noise (safe margin above 0.5%)
WEAK_CROSSOVER = 0.12    # GP -> Weak at 12% noise (GP collapses by 15%)


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


def select_derivative_strategy(estimated_noise_pct, n_data_points=None):
    """
    Select the optimal derivative estimation strategy based on estimated noise.

    Empirically calibrated thresholds based on R²(clean) crossover analysis
    on 50×50 grids comparing FD, Savitzky-Golay, GP-derived derivatives, and
    Weak SINDy. See ``recalibrate_adaptive_with_gp()``.

    Regimes:
    - noise < 0.005 (0.5%): Finite differences — exact on clean data
    - 0.005 ≤ noise < 0.01 (1%): Savitzky-Golay — narrow safety band
      where GP advantage over SavGol is < 0.01 R²
    - 0.01 ≤ noise < 0.12 (12%): GP-derived derivatives — RBF Gaussian
      Process posterior gives analytical derivatives that dominate at
      moderate-to-high noise (R²(clean) ≈ 0.92-0.99 in this regime)
    - 0.12 ≤ noise < 0.50: Weak SINDy — integral methods avoid
      differentiation entirely; GP collapses at ≥15% noise where Weak
      keeps R²(clean) ≈ 0.83-0.88
    - noise ≥ 0.50: Unreliable (weak SINDy, but warn)

    Why GP beats Weak in the moderate-noise regime:
    Empirical sweep shows GP R²(clean) ≈ 0.95 vs Weak R²(clean) ≈ 0.885
    at 5% noise, and 0.92 vs 0.89 at 10%. GP wins because RBF length-scale
    learning effectively denoises while preserving curvature for d²V/dS².
    Weak SINDy still wins at very high noise (≥15%) where GP overfits the
    learned length-scale to noise.

    Why 'neural' and 'weak' are still reachable via force_strategy:
    Kept for completeness and ablation studies. Neural derivatives
    (autograd of a fitted NN) introduce approximation bias that exceeds
    noise reduction benefit at every level tested, so they are not in the
    default selection map.

    Parameters
    ----------
    estimated_noise_pct : float
    n_data_points : int or None
        Optional total data-point count of the surface.  When provided and
        less than 300, forces ``strategy='savgol'`` even if the noise level
        would otherwise select GP -- a GP fit on a sparse surface tends to
        over-smooth (see real-data MSFT regression where 172 points gave
        GP R^2 = 0.58 vs SavGol 0.82).

    Returns
    -------
    strategy : str
        One of 'fd', 'savgol', 'gp', 'weak', 'unreliable'.
        ('neural' and 'weak' are accessible via force_strategy in
        adaptive_sindy_discover but not selected by default.)
    params : dict
        Recommended hyperparameters for the selected strategy.
    """
    if estimated_noise_pct < 0.005:
        strategy = 'fd'
        params = {'smooth': False}
    elif estimated_noise_pct < GP_CROSSOVER:
        strategy = 'savgol'
        params = {'smooth': True, 'savgol_window': 21, 'savgol_poly': 5}
    elif estimated_noise_pct < WEAK_CROSSOVER:
        strategy = 'gp'
        params = {'n_subsample': 500}
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

    # Density-aware override: GP is unreliable on very sparse surfaces.
    if (n_data_points is not None and n_data_points < 300
            and strategy == 'gp'):
        logger.info(
            f"Density-aware override: n_data_points={n_data_points} < 300, "
            f"forcing strategy='savgol' instead of '{strategy}'."
        )
        strategy = 'savgol'
        params = {'smooth': True, 'savgol_window': 21, 'savgol_poly': 5}

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
        Override auto-detection. Accepts 'fd', 'savgol', 'gp', 'weak',
        'neural', or 'unreliable'.
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
    from src.gp_derivatives import sindy_with_gp_derivatives

    set_all_seeds(seed)

    # Step 1: Estimate noise
    estimated_noise = estimate_noise_level(V, S_grid, t_grid)

    # Step 2: Select strategy
    if force_strategy is not None:
        strategy = force_strategy
        _, params = select_derivative_strategy(estimated_noise)
        # When forcing, override params with sensible defaults for that strategy
        # if the auto-selected params don't match.
        if strategy == 'fd':
            params = {'smooth': False}
        elif strategy == 'savgol':
            params = {'smooth': True, 'savgol_window': 21, 'savgol_poly': 5}
        elif strategy == 'gp':
            params = {'n_subsample': 500}
        elif strategy in ('weak', 'unreliable'):
            params = {'n_test_functions': 100}
        elif strategy == 'neural':
            params = {'fit_epochs': 1500}
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
    elif strategy == 'gp':
        n_sub = params.get('n_subsample', 500)
        result = sindy_with_gp_derivatives(
            V, S_grid, t_grid, n_subsample=n_sub, seed=seed,
            K=K, T=T, option_type=option_type,
            true_sigma=true_sigma, true_r=true_r,
            sigma=true_sigma if true_sigma is not None else 0.2,
            r=true_r if true_r is not None else 0.05,
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


def recalibrate_adaptive_with_gp(K=100, r=0.05, sigma=0.2, T=1.0,
                                   n_S=50, n_t=50, seed=42,
                                   noise_levels=None, n_subsample=500):
    """
    Run a fine-grained noise sweep comparing FD, SavGol, GP, and Weak SINDy.

    For each method at each noise level, compute R²(clean). Identifies the
    GP crossover threshold — the lowest noise level at which GP first beats
    SavGol consistently — and the GP -> Weak crossover where GP collapses.

    Default noise levels: ``[0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07,
    0.10, 0.15, 0.20, 0.25, 0.30]``.

    Parameters
    ----------
    K, r, sigma, T : float
        Black-Scholes parameters.
    n_S, n_t : int
        Grid size. 50x50 keeps runtime under ~3 minutes on CPU.
    seed : int
    noise_levels : list[float] or None
        Override default noise sweep.
    n_subsample : int
        GP training subsample size.

    Returns
    -------
    df : pandas.DataFrame
        Columns: noise_pct, r2_fd, r2_savgol, r2_gp, r2_weak,
                 runtime_fd_s, runtime_savgol_s, runtime_gp_s, runtime_weak_s.
    recommended : dict
        Keys: 'gp_crossover' (SavGol -> GP), 'weak_crossover' (GP -> Weak),
        'thresholds' (full strategy map), 'notes'.
    """
    # Local imports to avoid circulars during module load
    from src.data_generation import generate_price_surface, add_noise
    from src.sindy_discovery import discover_pde, compute_r2_clean
    from src.weak_sindy import weak_sindy_discover
    from src.gp_derivatives import sindy_with_gp_derivatives

    set_all_seeds(seed)

    if noise_levels is None:
        noise_levels = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07,
                        0.10, 0.15, 0.20, 0.25, 0.30]

    V_clean, S_grid, t_grid = generate_price_surface(
        n_S=n_S, n_t=n_t, K=K, r=r, sigma=sigma, T=T,
    )

    rows = []
    for nl in noise_levels:
        V = add_noise(V_clean, nl, seed=seed) if nl > 0 else V_clean.copy()
        row = {'noise_pct': float(nl)}

        # FD
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = discover_pde(
                    V, S_grid, t_grid, true_sigma=sigma, true_r=r,
                    smooth=False, K=K, T=T,
                )
                row['r2_fd'] = float(compute_r2_clean(
                    res['discovered_coefficients'], S_grid, t_grid,
                    K=K, r=r, sigma=sigma, T=T,
                ))
        except Exception as e:
            logger.warning(f"FD failed at noise={nl}: {e}")
            row['r2_fd'] = float('nan')
        row['runtime_fd_s'] = time.perf_counter() - t0

        # SavGol
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = discover_pde(
                    V, S_grid, t_grid, true_sigma=sigma, true_r=r,
                    smooth=True, savgol_window=21, savgol_poly=5,
                    K=K, T=T,
                )
                row['r2_savgol'] = float(compute_r2_clean(
                    res['discovered_coefficients'], S_grid, t_grid,
                    K=K, r=r, sigma=sigma, T=T,
                ))
        except Exception as e:
            logger.warning(f"SavGol failed at noise={nl}: {e}")
            row['r2_savgol'] = float('nan')
        row['runtime_savgol_s'] = time.perf_counter() - t0

        # GP
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sindy_with_gp_derivatives(
                    V, S_grid, t_grid, n_subsample=n_subsample, seed=seed,
                    K=K, r=r, sigma=sigma, T=T,
                    true_sigma=sigma, true_r=r,
                )
                row['r2_gp'] = float(res['r2_clean'])
        except Exception as e:
            logger.warning(f"GP failed at noise={nl}: {e}")
            row['r2_gp'] = float('nan')
        row['runtime_gp_s'] = time.perf_counter() - t0

        # Weak
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = weak_sindy_discover(
                    V, S_grid, t_grid, n_test_functions=100,
                    true_sigma=sigma, true_r=r, seed=seed,
                )
                row['r2_weak'] = float(compute_r2_clean(
                    res['discovered_coefficients'], S_grid, t_grid,
                    K=K, r=r, sigma=sigma, T=T,
                ))
        except Exception as e:
            logger.warning(f"Weak failed at noise={nl}: {e}")
            row['r2_weak'] = float('nan')
        row['runtime_weak_s'] = time.perf_counter() - t0

        rows.append(row)

    df = pd.DataFrame(rows)

    # Identify GP crossover: lowest noise > 0 where GP first beats SavGol.
    gp_crossover = None
    for _, r_ in df.iterrows():
        nl = r_['noise_pct']
        if nl <= 0:
            continue
        if (not np.isnan(r_['r2_gp']) and not np.isnan(r_['r2_savgol'])
                and r_['r2_gp'] > r_['r2_savgol']):
            gp_crossover = float(nl)
            break

    # Identify GP -> Weak crossover: lowest noise where Weak first beats GP.
    weak_crossover = None
    for _, r_ in df.iterrows():
        nl = r_['noise_pct']
        if not np.isnan(r_['r2_gp']) and not np.isnan(r_['r2_weak']) \
                and r_['r2_weak'] > r_['r2_gp']:
            weak_crossover = float(nl)
            break

    # Fall back to currently-calibrated values if sweep can't identify a clean
    # crossover (e.g. all-NaN row).
    if gp_crossover is None:
        gp_crossover = GP_CROSSOVER
    if weak_crossover is None:
        weak_crossover = WEAK_CROSSOVER

    recommended = {
        'gp_crossover': gp_crossover,
        'weak_crossover': weak_crossover,
        'thresholds': {
            'fd': (0.0, 0.005),
            'savgol': (0.005, gp_crossover),
            'gp': (gp_crossover, weak_crossover),
            'weak': (weak_crossover, 0.50),
            'unreliable': (0.50, float('inf')),
        },
        'notes': (
            f"GP first beats SavGol at noise={gp_crossover:.3f}; "
            f"Weak first beats GP at noise={weak_crossover:.3f}."
        ),
    }

    logger.info(
        f"Recalibration complete:\n{df.to_string(index=False)}\n"
        f"Recommended: gp_crossover={gp_crossover:.3f}, "
        f"weak_crossover={weak_crossover:.3f}"
    )

    return df, recommended
