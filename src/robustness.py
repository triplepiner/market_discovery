"""
Noise robustness and parameter generalization experiments.

Evaluates how well SINDy-based PDE discovery recovers the Black-Scholes
equation under increasing observation noise and across different market
parameter regimes (volatility, risk-free rate).
"""

import numpy as np
import pandas as pd
from src.utils import set_all_seeds, setup_logging, safe_relative_error, Timer
from src.data_generation import generate_price_surface, add_noise
from src.sindy_discovery import discover_pde, TERM_NAMES

try:
    from src.diagnostics import check_sindy_sparsity_stability
    _HAS_BOOTSTRAP = True
except ImportError:
    _HAS_BOOTSTRAP = False

logger = setup_logging(__name__)

# Indices into the TERM_NAMES / coefficient arrays
_IDX_V = 0          # 'V'
_IDX_DVDS = 1       # 'dV/dS'
_IDX_D2VDS2 = 2     # 'd2V/dS2'
_IDX_SDVDS = 3      # 'S*dV/dS'
_IDX_S2D2VDS2 = 4   # 'S2*d2V/dS2'

# The three terms that should be active in the true Black-Scholes PDE
_TRUE_ACTIVE_INDICES = {_IDX_V, _IDX_SDVDS, _IDX_S2D2VDS2}


def _true_coefficients(r, sigma):
    """Return the true Black-Scholes PDE coefficients for given r and sigma.

    The PDE is:
        dV/dt = r*V - r*S*dV/dS - 0.5*sigma^2*S^2*d2V/dS^2

    Parameters
    ----------
    r : float
        Risk-free rate.
    sigma : float
        Volatility.

    Returns
    -------
    ndarray, shape (5,)
        Coefficients for [V, dV/dS, d2V/dS2, S*dV/dS, S2*d2V/dS2].
    """
    return np.array([
        r,                      # V
        0.0,                    # dV/dS
        0.0,                    # d2V/dS2
        -r,                     # S*dV/dS
        -0.5 * sigma ** 2,     # S2*d2V/dS2
    ])


def _check_correct_structure(active_mask):
    """Return True if exactly the 3 true BS terms are active.

    Parameters
    ----------
    active_mask : ndarray of bool, shape (5,)

    Returns
    -------
    bool
    """
    discovered_active = set(np.where(active_mask)[0])
    return discovered_active == _TRUE_ACTIVE_INDICES


def _count_false_positives(active_mask):
    """Count spurious terms that are active but should not be.

    Parameters
    ----------
    active_mask : ndarray of bool, shape (5,)

    Returns
    -------
    int
    """
    discovered_active = set(np.where(active_mask)[0])
    return len(discovered_active - _TRUE_ACTIVE_INDICES)


def _count_false_negatives(active_mask):
    """Count true terms that are missing from the discovered set.

    Parameters
    ----------
    active_mask : ndarray of bool, shape (5,)

    Returns
    -------
    int
    """
    discovered_active = set(np.where(active_mask)[0])
    return len(_TRUE_ACTIVE_INDICES - discovered_active)


def _false_positive_term_names(active_mask):
    """Return names of spurious active terms.

    Parameters
    ----------
    active_mask : ndarray of bool, shape (5,)

    Returns
    -------
    str
        Comma-separated term names, or empty string if none.
    """
    discovered_active = set(np.where(active_mask)[0])
    fp_indices = sorted(discovered_active - _TRUE_ACTIVE_INDICES)
    return ','.join(TERM_NAMES[i] for i in fp_indices)


def _run_bootstrap_stability(V, S_grid, t_grid, smooth, sigma, r, K, T):
    """Run bootstrap stability check if available.

    Parameters
    ----------
    V : ndarray
        Price surface (possibly noisy).
    S_grid, t_grid : ndarray
        Grid arrays.
    smooth : bool
        Whether smoothing is enabled.
    sigma, r : float
        True parameters.
    K, T : float
        Strike and maturity.

    Returns
    -------
    float
        Bootstrap stability percentage, or -1.0 if unavailable.
    """
    if not _HAS_BOOTSTRAP:
        return -1.0
    try:
        stability_result = check_sindy_sparsity_stability(
            V, S_grid, t_grid,
            n_bootstrap=10,
            K=K, r=r, sigma=sigma, T=T,
        )
        if isinstance(stability_result, dict):
            # Fraction of bootstraps with consistent term selection
            freq = stability_result.get('selection_frequency', np.array([]))
            if len(freq) > 0:
                # Stability: % of bootstraps where the 3 true terms were selected
                true_term_freq = np.mean([freq[i] for i in _TRUE_ACTIVE_INDICES])
                return float(true_term_freq * 100)
            return float(100.0 if stability_result.get('stable', False) else 0.0)
        return float(stability_result)
    except Exception as e:
        logger.warning(f"Bootstrap stability check failed: {e}")
        return -1.0


def run_noise_robustness(noise_levels=None, K=100, r=0.05, sigma=0.2, T=1.0,
                         quick_pinn_epochs=3000):
    """
    Run noise robustness experiments across multiple noise levels.

    For each noise level, generates a clean BS price surface, adds Gaussian
    noise, runs SINDy discovery, and records how well the true PDE is
    recovered.

    Parameters
    ----------
    noise_levels : list of float or None
        Noise levels as fractions of the surface standard deviation.
        Default: [0, 0.01, 0.05, 0.10, 0.20].
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Option maturity.
    quick_pinn_epochs : int
        Number of PINN epochs (reserved for future use; not used by SINDy).

    Returns
    -------
    pandas.DataFrame
        One row per noise level with columns:
        noise_level, coeff_V, coeff_dVdS, coeff_d2VdS2, coeff_SdVdS,
        coeff_S2d2VdS2, rel_error_V, rel_error_SdVdS, rel_error_S2d2VdS2,
        n_active_terms, correct_structure, r2, bic, false_positive_terms,
        bootstrap_stability_pct
    """
    if noise_levels is None:
        noise_levels = [0, 0.01, 0.05, 0.10, 0.20]

    set_all_seeds(42)

    true_coeffs = _true_coefficients(r, sigma)

    logger.info(
        f"Starting noise robustness experiment: {len(noise_levels)} levels, "
        f"K={K}, r={r}, sigma={sigma}, T={T}"
    )
    logger.info(
        f"True coefficients: V={true_coeffs[0]:.4f}, "
        f"S*dV/dS={true_coeffs[3]:.4f}, "
        f"S2*d2V/dS2={true_coeffs[4]:.4f}"
    )

    # Generate clean surface once
    V_clean, S_grid, t_grid = generate_price_surface(
        K=K, r=r, sigma=sigma, T=T
    )

    rows = []
    for noise_level in noise_levels:
        logger.info(f"--- Noise level: {noise_level:.2%} ---")

        # Add noise
        V_noisy = add_noise(V_clean, noise_level, seed=42)

        # Use smoothing when there is noise
        use_smooth = noise_level > 0

        # Run SINDy discovery
        result = discover_pde(
            V_noisy, S_grid, t_grid,
            true_sigma=sigma, true_r=r,
            smooth=use_smooth,
            K=K, T=T, option_type='call',
        )

        disc_coeffs = result['discovered_coefficients']
        active_mask = result['active_mask']

        # Relative errors for the three true active terms
        rel_err_V = float(safe_relative_error(disc_coeffs[_IDX_V], true_coeffs[_IDX_V]))
        rel_err_SdVdS = float(safe_relative_error(disc_coeffs[_IDX_SDVDS], true_coeffs[_IDX_SDVDS]))
        rel_err_S2d2VdS2 = float(safe_relative_error(disc_coeffs[_IDX_S2D2VDS2], true_coeffs[_IDX_S2D2VDS2]))

        correct = _check_correct_structure(active_mask)
        n_active = int(result['n_active'])
        fp_terms = _false_positive_term_names(active_mask)
        n_fp = _count_false_positives(active_mask)
        n_fn = _count_false_negatives(active_mask)

        # Bootstrap stability
        bootstrap_pct = _run_bootstrap_stability(
            V_noisy, S_grid, t_grid,
            smooth=use_smooth,
            sigma=sigma, r=r, K=K, T=T,
        )

        logger.info(
            f"  Discovered: {result['human_readable_pde']}"
        )
        logger.info(
            f"  R2={result['r2_score']:.6f}, BIC={result['bic']:.1f}, "
            f"active={n_active}, correct_structure={correct}"
        )
        logger.info(
            f"  Rel errors: V={rel_err_V:.4f}, S*dV/dS={rel_err_SdVdS:.4f}, "
            f"S2*d2V/dS2={rel_err_S2d2VdS2:.4f}"
        )
        if n_fp > 0:
            logger.warning(f"  False positives ({n_fp}): {fp_terms}")
        if n_fn > 0:
            logger.warning(
                f"  False negatives ({n_fn}): missing true terms"
            )

        rows.append({
            'noise_level': noise_level,
            'coeff_V': disc_coeffs[_IDX_V],
            'coeff_dVdS': disc_coeffs[_IDX_DVDS],
            'coeff_d2VdS2': disc_coeffs[_IDX_D2VDS2],
            'coeff_SdVdS': disc_coeffs[_IDX_SDVDS],
            'coeff_S2d2VdS2': disc_coeffs[_IDX_S2D2VDS2],
            'rel_error_V': rel_err_V,
            'rel_error_SdVdS': rel_err_SdVdS,
            'rel_error_S2d2VdS2': rel_err_S2d2VdS2,
            'n_active_terms': n_active,
            'correct_structure': correct,
            'r2': result['r2_score'],
            'bic': result['bic'],
            'false_positive_terms': fp_terms,
            'bootstrap_stability_pct': bootstrap_pct,
        })

    df = pd.DataFrame(rows)

    logger.info("=== Noise Robustness Summary ===")
    logger.info(f"\n{df.to_string(index=False)}")

    return df


def run_parameter_generalization(sigma_list=None, r_list=None, K=100, T=1.0):
    """
    Run parameter generalization experiments across volatility and rate grids.

    For each (sigma, r) combination, generates a clean BS price surface, runs
    SINDy discovery, and compares discovered coefficients to the true values
    for that parameter set.

    Parameters
    ----------
    sigma_list : list of float or None
        Volatility values to test. Default: [0.1, 0.2, 0.3, 0.4].
    r_list : list of float or None
        Risk-free rate values to test. Default: [0.01, 0.05, 0.10].
    K : float
        Strike price.
    T : float
        Option maturity.

    Returns
    -------
    pandas.DataFrame
        One row per (sigma, r) combination with columns:
        sigma, r, true_coeff_V, disc_coeff_V, rel_error_V,
        true_coeff_SdVdS, disc_coeff_SdVdS, rel_error_SdVdS,
        true_coeff_S2d2VdS2, disc_coeff_S2d2VdS2, rel_error_S2d2VdS2,
        correct_structure, r2
    """
    if sigma_list is None:
        sigma_list = [0.1, 0.2, 0.3, 0.4]
    if r_list is None:
        r_list = [0.01, 0.05, 0.10]

    set_all_seeds(42)

    n_combos = len(sigma_list) * len(r_list)
    logger.info(
        f"Starting parameter generalization experiment: "
        f"{len(sigma_list)} sigmas x {len(r_list)} rates = {n_combos} combinations"
    )

    rows = []
    for sigma in sigma_list:
        for r_val in r_list:
            logger.info(f"--- sigma={sigma:.2f}, r={r_val:.2f} ---")

            true_coeffs = _true_coefficients(r_val, sigma)

            # Generate clean surface for this parameter combination
            V, S_grid, t_grid = generate_price_surface(
                K=K, r=r_val, sigma=sigma, T=T
            )

            # Run SINDy discovery (no noise, no smoothing needed)
            result = discover_pde(
                V, S_grid, t_grid,
                true_sigma=sigma, true_r=r_val,
                smooth=False,
                K=K, T=T, option_type='call',
            )

            disc_coeffs = result['discovered_coefficients']
            active_mask = result['active_mask']

            # Relative errors for the three true active terms
            rel_err_V = float(safe_relative_error(
                disc_coeffs[_IDX_V], true_coeffs[_IDX_V]
            ))
            rel_err_SdVdS = float(safe_relative_error(
                disc_coeffs[_IDX_SDVDS], true_coeffs[_IDX_SDVDS]
            ))
            rel_err_S2d2VdS2 = float(safe_relative_error(
                disc_coeffs[_IDX_S2D2VDS2], true_coeffs[_IDX_S2D2VDS2]
            ))

            correct = _check_correct_structure(active_mask)

            logger.info(
                f"  Discovered: {result['human_readable_pde']}"
            )
            logger.info(
                f"  R2={result['r2_score']:.6f}, correct_structure={correct}"
            )
            logger.info(
                f"  True:  V={true_coeffs[_IDX_V]:.4f}, "
                f"S*dV/dS={true_coeffs[_IDX_SDVDS]:.4f}, "
                f"S2*d2V/dS2={true_coeffs[_IDX_S2D2VDS2]:.4f}"
            )
            logger.info(
                f"  Disc:  V={disc_coeffs[_IDX_V]:.4f}, "
                f"S*dV/dS={disc_coeffs[_IDX_SDVDS]:.4f}, "
                f"S2*d2V/dS2={disc_coeffs[_IDX_S2D2VDS2]:.4f}"
            )
            logger.info(
                f"  Rel errors: V={rel_err_V:.4f}, "
                f"S*dV/dS={rel_err_SdVdS:.4f}, "
                f"S2*d2V/dS2={rel_err_S2d2VdS2:.4f}"
            )

            rows.append({
                'sigma': sigma,
                'r': r_val,
                'true_coeff_V': true_coeffs[_IDX_V],
                'disc_coeff_V': disc_coeffs[_IDX_V],
                'rel_error_V': rel_err_V,
                'true_coeff_SdVdS': true_coeffs[_IDX_SDVDS],
                'disc_coeff_SdVdS': disc_coeffs[_IDX_SDVDS],
                'rel_error_SdVdS': rel_err_SdVdS,
                'true_coeff_S2d2VdS2': true_coeffs[_IDX_S2D2VDS2],
                'disc_coeff_S2d2VdS2': disc_coeffs[_IDX_S2D2VDS2],
                'rel_error_S2d2VdS2': rel_err_S2d2VdS2,
                'correct_structure': correct,
                'r2': result['r2_score'],
            })

    df = pd.DataFrame(rows)

    logger.info("=== Parameter Generalization Summary ===")
    logger.info(f"\n{df.to_string(index=False)}")

    # Summary statistics
    n_correct = df['correct_structure'].sum()
    logger.info(
        f"Correct structure recovered in {n_correct}/{n_combos} "
        f"({n_correct / n_combos:.0%}) of parameter combinations"
    )
    logger.info(
        f"Mean relative errors: V={df['rel_error_V'].mean():.4f}, "
        f"S*dV/dS={df['rel_error_SdVdS'].mean():.4f}, "
        f"S2*d2V/dS2={df['rel_error_S2d2VdS2'].mean():.4f}"
    )

    return df


# ---------------------------------------------------------------------------
# Fix 2: Noise-vs-smoothing experiments
# ---------------------------------------------------------------------------


def run_smoothing_ablation(noise_pct=0.05, K=100, r=0.05, sigma=0.2, T=1.0):
    """
    Ablation over Savitzky-Golay smoothing settings on a noisy BS surface.

    Takes a clean Black-Scholes surface, adds *noise_pct* noise, and runs
    SINDy discovery once without smoothing and once for each of several
    Savitzky-Golay (window, poly) settings.  Returns a list of result dicts
    that can be passed directly to plotting utilities.

    Parameters
    ----------
    noise_pct : float
        Noise level as fraction of the surface standard deviation.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Option maturity.

    Returns
    -------
    list of dict
        Each dict contains:
        - smoothing : str  ("None" or "window,poly" e.g. "7,3")
        - r2 : float
        - coefficients : ndarray(5,)
        - rel_errors : ndarray or None
        - n_active : int
        - correct_structure : bool
        - human_readable_pde : str
    """
    set_all_seeds(42)

    logger.info(
        f"Starting smoothing ablation: noise_pct={noise_pct:.2%}, "
        f"K={K}, r={r}, sigma={sigma}, T={T}"
    )

    # Generate clean surface and add noise
    V_clean, S_grid, t_grid = generate_price_surface(K=K, r=r, sigma=sigma, T=T)
    V_noisy = add_noise(V_clean, noise_pct, seed=42)

    smoothing_settings = [
        None,          # no smoothing
        (5, 3),
        (7, 3),
        (11, 3),
        (11, 5),
        (15, 5),
        (21, 5),
    ]

    results = []
    for setting in smoothing_settings:
        if setting is None:
            label = "None"
            use_smooth = False
            w, p = 7, 3  # defaults (unused when smooth=False)
        else:
            w, p = setting
            label = f"{w},{p}"
            use_smooth = True

        logger.info(f"  Smoothing: {label}")

        with Timer(f"SINDy smoothing={label}"):
            result = discover_pde(
                V_noisy, S_grid, t_grid,
                true_sigma=sigma, true_r=r,
                smooth=use_smooth,
                savgol_window=w, savgol_poly=p,
            )

        active_mask = result['active_mask']
        correct = _check_correct_structure(active_mask)

        logger.info(
            f"    R2={result['r2_score']:.6f}, n_active={result['n_active']}, "
            f"correct={correct}, PDE: {result['human_readable_pde']}"
        )

        results.append({
            'smoothing': label,
            'r2': result['r2_score'],
            'coefficients': result['discovered_coefficients'],
            'rel_errors': result['relative_errors'],
            'n_active': int(result['n_active']),
            'correct_structure': correct,
            'human_readable_pde': result['human_readable_pde'],
        })

    logger.info("=== Smoothing Ablation Complete ===")
    return results


def run_grid_resolution_vs_noise(noise_pct=0.05, K=100, r=0.05, sigma=0.2,
                                  T=1.0):
    """
    Evaluate SINDy performance across grid resolutions with and without noise.

    For each grid size, generates a clean BS surface at that resolution,
    optionally adds noise, runs SINDy discovery, and records fit quality
    metrics.  This helps determine whether higher resolution compensates
    for noise.

    Parameters
    ----------
    noise_pct : float
        Noise level as fraction of the surface standard deviation.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Option maturity.

    Returns
    -------
    list of dict
        Each dict contains:
        - grid_size : int
        - noise : float
        - r2_clean : float
        - r2_noisy : float
        - n_active_clean : int
        - n_active_noisy : int
        - correct_structure_clean : bool
        - correct_structure_noisy : bool
    """
    set_all_seeds(42)

    grid_sizes = [30, 50, 100, 200]

    logger.info(
        f"Starting grid resolution vs noise experiment: "
        f"grids={grid_sizes}, noise_pct={noise_pct:.2%}"
    )

    results = []
    for n in grid_sizes:
        logger.info(f"  Grid size: {n}x{n}")

        with Timer(f"Grid {n}x{n} clean+noisy"):
            # Generate surface at this resolution
            V_clean, S_grid, t_grid = generate_price_surface(
                n_S=n, n_t=n, K=K, r=r, sigma=sigma, T=T
            )

            # --- Clean run ---
            res_clean = discover_pde(
                V_clean, S_grid, t_grid,
                true_sigma=sigma, true_r=r,
                smooth=False,
            )
            correct_clean = _check_correct_structure(res_clean['active_mask'])

            # --- Noisy run (no smoothing) ---
            V_noisy = add_noise(V_clean, noise_pct, seed=42)
            res_noisy = discover_pde(
                V_noisy, S_grid, t_grid,
                true_sigma=sigma, true_r=r,
                smooth=False,
            )
            correct_noisy = _check_correct_structure(res_noisy['active_mask'])

        logger.info(
            f"    Clean: R2={res_clean['r2_score']:.6f}, "
            f"active={res_clean['n_active']}, correct={correct_clean}"
        )
        logger.info(
            f"    Noisy: R2={res_noisy['r2_score']:.6f}, "
            f"active={res_noisy['n_active']}, correct={correct_noisy}"
        )

        results.append({
            'grid_size': n,
            'noise': noise_pct,
            'r2_clean': res_clean['r2_score'],
            'r2_noisy': res_noisy['r2_score'],
            'n_active_clean': int(res_clean['n_active']),
            'n_active_noisy': int(res_noisy['n_active']),
            'correct_structure_clean': correct_clean,
            'correct_structure_noisy': correct_noisy,
        })

    logger.info("=== Grid Resolution vs Noise Complete ===")
    return results


def run_noise_smoothing_matrix(K=100, r=0.05, sigma=0.2, T=1.0):
    """
    Full cross of noise levels and smoothing settings.

    Generates a clean BS surface, then for every combination of noise level
    and smoothing setting runs SINDy and records R², discovered coefficients,
    and coefficient biases relative to the true Black-Scholes values.

    Parameters
    ----------
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Option maturity.

    Returns
    -------
    list of dict
        Each dict contains:
        - noise_pct : float
        - smoothing : str  ("None" or "window,poly")
        - r2 : float
        - coeff_V : float
        - coeff_SdVdS : float
        - coeff_S2d2VdS2 : float
        - bias_V : float
        - bias_SdVdS : float
        - bias_S2d2VdS2 : float
        - correct_structure : bool
    """
    set_all_seeds(42)

    noise_levels = [0.0, 0.01, 0.05, 0.10]
    smoothing_settings = [None, (7, 3), (11, 5), (21, 5)]

    true_coeffs = _true_coefficients(r, sigma)
    true_V = true_coeffs[_IDX_V]               # r
    true_SdVdS = true_coeffs[_IDX_SDVDS]       # -r
    true_S2d2VdS2 = true_coeffs[_IDX_S2D2VDS2] # -0.5*sigma^2

    logger.info(
        f"Starting noise x smoothing matrix: "
        f"{len(noise_levels)} noise levels x {len(smoothing_settings)} smoothing settings"
    )
    logger.info(
        f"True coefficients: V={true_V:.4f}, "
        f"S*dV/dS={true_SdVdS:.4f}, "
        f"S2*d2V/dS2={true_S2d2VdS2:.4f}"
    )

    # Generate clean surface once
    V_clean, S_grid, t_grid = generate_price_surface(K=K, r=r, sigma=sigma, T=T)

    results = []
    for noise_pct in noise_levels:
        V_noisy = add_noise(V_clean, noise_pct, seed=42) if noise_pct > 0 else V_clean.copy()

        for setting in smoothing_settings:
            if setting is None:
                label = "None"
                use_smooth = False
                w, p = 7, 3
            else:
                w, p = setting
                label = f"{w},{p}"
                use_smooth = True

            logger.info(f"  noise={noise_pct:.2%}, smoothing={label}")

            with Timer(f"noise={noise_pct:.2%} smooth={label}"):
                result = discover_pde(
                    V_noisy, S_grid, t_grid,
                    true_sigma=sigma, true_r=r,
                    smooth=use_smooth,
                    savgol_window=w, savgol_poly=p,
                )

            disc = result['discovered_coefficients']
            active_mask = result['active_mask']
            correct = _check_correct_structure(active_mask)

            bias_V = float(disc[_IDX_V] - true_V)
            bias_SdVdS = float(disc[_IDX_SDVDS] - true_SdVdS)
            bias_S2d2VdS2 = float(disc[_IDX_S2D2VDS2] - true_S2d2VdS2)

            logger.info(
                f"    R2={result['r2_score']:.6f}, correct={correct}, "
                f"bias_V={bias_V:.4e}, bias_SdVdS={bias_SdVdS:.4e}, "
                f"bias_S2d2VdS2={bias_S2d2VdS2:.4e}"
            )

            results.append({
                'noise_pct': noise_pct,
                'smoothing': label,
                'r2': result['r2_score'],
                'coeff_V': float(disc[_IDX_V]),
                'coeff_SdVdS': float(disc[_IDX_SDVDS]),
                'coeff_S2d2VdS2': float(disc[_IDX_S2D2VDS2]),
                'bias_V': bias_V,
                'bias_SdVdS': bias_SdVdS,
                'bias_S2d2VdS2': bias_S2d2VdS2,
                'correct_structure': correct,
            })

    logger.info("=== Noise x Smoothing Matrix Complete ===")
    return results
