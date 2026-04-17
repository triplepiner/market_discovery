"""
Weak-form SINDy for noise-robust PDE discovery.

Implements the integral form of SINDy (Messenger & Bortz, 2021) which avoids
pointwise differentiation by using integration by parts with smooth test
functions. All derivatives are moved onto the test functions (which are known
analytically), so only the raw data V appears in integrands.
"""

import time
import numpy as np
from src.utils import set_all_seeds, setup_logging, safe_relative_error
from src.sindy_discovery import stlsq_sweep, format_pde_string, TERM_NAMES

logger = setup_logging(__name__)


def create_test_functions(S_grid, t_grid, n_functions=100, width_S=None,
                           width_t=None, seed=42):
    """
    Generate Gaussian test functions with analytical derivatives.

    Each test function is a product of Gaussians:
      φ_k(S,t) = exp(-(S-S_k)²/(2·w_S²)) · exp(-(t-t_k)²/(2·w_t²))

    with (S_k, t_k) randomly chosen from the interior of the domain.

    Parameters
    ----------
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    n_functions : int
        Number of test functions.
    width_S : float or None
        Gaussian width in S direction. Default: (S_max-S_min)/10.
    width_t : float or None
        Gaussian width in t direction. Default: (t_max-t_min)/10.
    seed : int

    Returns
    -------
    list of dict
        Each dict contains: 'phi', 'dphi_dS', 'd2phi_dS2', 'dphi_dt'
        as 2D arrays of shape (n_S, n_t).
    """
    rng = np.random.RandomState(seed)

    S_min, S_max = S_grid[0], S_grid[-1]
    t_min, t_max = t_grid[0], t_grid[-1]

    if width_S is None:
        width_S = (S_max - S_min) / 20.0
    if width_t is None:
        width_t = (t_max - t_min) / 20.0

    # Random centers in the interior — ensure 3.5σ margin from boundaries
    # so that the truncated Gaussians truly vanish at domain edges
    margin_S = 3.5 * width_S
    margin_t = 3.5 * width_t
    S_lo = S_min + margin_S
    S_hi = S_max - margin_S
    t_lo = t_min + margin_t
    t_hi = t_max - margin_t
    # Clamp to ensure valid range
    if S_lo >= S_hi:
        S_lo = 0.3 * S_min + 0.7 * S_max
        S_hi = 0.7 * S_min + 0.3 * S_max
    if t_lo >= t_hi:
        t_lo = 0.3 * t_min + 0.7 * t_max
        t_hi = 0.7 * t_min + 0.3 * t_max
    S_centers = rng.uniform(S_lo, S_hi, size=n_functions)
    t_centers = rng.uniform(t_lo, t_hi, size=n_functions)

    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')

    test_functions = []
    for k in range(n_functions):
        S_k = S_centers[k]
        t_k = t_centers[k]
        wS2 = width_S ** 2
        wt2 = width_t ** 2

        # Gaussian components
        gS = np.exp(-0.5 * (S_mesh - S_k) ** 2 / wS2)
        gt = np.exp(-0.5 * (t_mesh - t_k) ** 2 / wt2)

        # Truncate to ~0 outside 3 sigma (compact support approximation)
        mask_S = np.abs(S_mesh - S_k) <= 3.0 * width_S
        mask_t = np.abs(t_mesh - t_k) <= 3.0 * width_t
        mask = mask_S & mask_t

        phi = gS * gt * mask

        # Analytical derivatives of φ (before masking the derivative values)
        # dφ/dS = -(S-S_k)/w_S² · φ
        dphi_dS = -(S_mesh - S_k) / wS2 * phi

        # d²φ/dS² = [(S-S_k)²/w_S⁴ - 1/w_S²] · φ
        d2phi_dS2 = ((S_mesh - S_k) ** 2 / (wS2 ** 2) - 1.0 / wS2) * phi

        # dφ/dt = -(t-t_k)/w_t² · φ
        dphi_dt = -(t_mesh - t_k) / wt2 * phi

        test_functions.append({
            'phi': phi,
            'dphi_dS': dphi_dS,
            'd2phi_dS2': d2phi_dS2,
            'dphi_dt': dphi_dt,
        })

    return test_functions


def _integrate_2d(f, S_grid, t_grid):
    """
    Compute ∫∫ f(S,t) dS dt using 2D trapezoidal rule.

    Parameters
    ----------
    f : ndarray, shape (n_S, n_t)
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)

    Returns
    -------
    float
    """
    # Integrate over S first (axis=0), then over t (axis=0 of result)
    try:
        inner = np.trapezoid(f, S_grid, axis=0)
        return float(np.trapezoid(inner, t_grid))
    except AttributeError:
        # numpy < 2.0 fallback
        inner = np.trapz(f, S_grid, axis=0)
        return float(np.trapz(inner, t_grid))


def weak_sindy_regression(V, S_grid, t_grid, test_functions, threshold=None):
    """
    Perform weak-form SINDy regression using pre-computed test functions.

    For each test function φ_k, computes integrals using integration by parts
    so that NO derivatives of V appear. All derivatives are transferred to
    the analytically known test functions.

    Parameters
    ----------
    V : ndarray, shape (n_S, n_t)
        Raw data (possibly noisy).
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    test_functions : list of dict
        From create_test_functions().
    threshold : float or None
        If None, uses stlsq_sweep.

    Returns
    -------
    best : dict (from stlsq_sweep)
    sweep_results : list of dict
    library : ndarray, shape (n_functions, 5)
    """
    S_mesh, _ = np.meshgrid(S_grid, t_grid, indexing='ij')

    n_func = len(test_functions)
    LHS = np.zeros(n_func)
    RHS = np.zeros((n_func, 5))

    for k, tf in enumerate(test_functions):
        phi = tf['phi']
        dphi_dS = tf['dphi_dS']
        d2phi_dS2 = tf['d2phi_dS2']
        dphi_dt = tf['dphi_dt']

        # LHS: -∫∫ (∂φ/∂t) · V dS dt
        LHS[k] = -_integrate_2d(dphi_dt * V, S_grid, t_grid)

        # RHS[0]: ∫∫ φ · V dS dt  (coefficient of V)
        RHS[k, 0] = _integrate_2d(phi * V, S_grid, t_grid)

        # RHS[1]: -∫∫ (∂φ/∂S) · V dS dt  (coefficient of dV/dS, IBP once)
        RHS[k, 1] = -_integrate_2d(dphi_dS * V, S_grid, t_grid)

        # RHS[2]: ∫∫ (∂²φ/∂S²) · V dS dt  (coefficient of d²V/dS², IBP twice)
        RHS[k, 2] = _integrate_2d(d2phi_dS2 * V, S_grid, t_grid)

        # RHS[3]: -∫∫ (S·∂φ/∂S + φ) · V dS dt  (coefficient of S·dV/dS, IBP once with product rule)
        integrand_3 = (S_mesh * dphi_dS + phi) * V
        RHS[k, 3] = -_integrate_2d(integrand_3, S_grid, t_grid)

        # RHS[4]: ∫∫ (S²·∂²φ/∂S² + 4S·∂φ/∂S + 2φ) · V dS dt
        # (coefficient of S²·d²V/dS², IBP twice with product rule)
        integrand_4 = (S_mesh**2 * d2phi_dS2 + 4.0 * S_mesh * dphi_dS + 2.0 * phi) * V
        RHS[k, 4] = _integrate_2d(integrand_4, S_grid, t_grid)

    # Run STLSQ sweep on the weak-form system
    best, sweep_results = stlsq_sweep(RHS, LHS)

    return best, sweep_results, RHS


def weak_sindy_discover(V, S_grid, t_grid, n_test_functions=100,
                         true_sigma=None, true_r=None, seed=42,
                         width_S=None, width_t=None):
    """
    Top-level weak-form SINDy PDE discovery.

    Parameters
    ----------
    V : ndarray, shape (n_S, n_t)
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    n_test_functions : int
    true_sigma : float or None
    true_r : float or None
    seed : int
    width_S : float or None
    width_t : float or None

    Returns
    -------
    dict with same keys as discover_pde()
    """
    set_all_seeds(seed)

    # Create test functions
    test_functions = create_test_functions(
        S_grid, t_grid, n_functions=n_test_functions,
        width_S=width_S, width_t=width_t, seed=seed
    )

    # Run weak-form regression
    best, sweep_results, library = weak_sindy_regression(
        V, S_grid, t_grid, test_functions
    )

    discovered = best['coefficients']
    cond_number = float(np.linalg.cond(library))

    # True coefficients
    true_coeffs = None
    rel_errors = None
    if true_sigma is not None and true_r is not None:
        true_coeffs = np.array([
            true_r,
            0.0,
            0.0,
            -true_r,
            -0.5 * true_sigma ** 2,
        ])
        rel_errors = safe_relative_error(discovered, true_coeffs)

    active_terms = [TERM_NAMES[i] for i in range(5) if best['active_mask'][i]]
    pde_str = format_pde_string(discovered, TERM_NAMES)

    logger.info(
        f"Weak SINDy: R²={best['r2']:.6f}, active={best['n_active']}, "
        f"PDE: {pde_str}, cond#={cond_number:.2e}"
    )

    return {
        'discovered_coefficients': discovered,
        'true_coefficients': true_coeffs,
        'active_terms': active_terms,
        'term_names': TERM_NAMES,
        'relative_errors': rel_errors,
        'best_threshold': best['threshold'],
        'r2_score': best['r2'],
        'bic': best['bic'],
        'condition_number': cond_number,
        'derivative_quality': {},
        'sweep_results': sweep_results,
        'human_readable_pde': pde_str,
        'active_mask': best['active_mask'],
        'n_active': best['n_active'],
    }


def tune_weak_sindy(V_clean, S_grid, t_grid, true_sigma, true_r,
                     n_functions_list=None, width_factors=None,
                     K=100, T=1.0, seed=42):
    """
    Test different weak SINDy hyperparameters on clean data.

    Sweeps over (n_test_functions, width_factor) combinations where
    width_S = (S_max - S_min) / width_factor.

    Parameters
    ----------
    V_clean : ndarray, shape (n_S, n_t)
    S_grid, t_grid : ndarray
    true_sigma, true_r : float
    n_functions_list : list of int or None
    width_factors : list of float or None
    K, T : float
    seed : int

    Returns
    -------
    dict with keys:
        'results_df': DataFrame with per-config metrics
        'best_config': dict with best n_functions, width_factor
        'best_r2_clean': float
    """
    import pandas as pd
    from src.sindy_discovery import compute_r2_clean, compute_coefficient_metrics

    if n_functions_list is None:
        n_functions_list = [30, 50, 100, 150, 200, 300]
    if width_factors is None:
        width_factors = [5, 8, 10, 15, 20]

    S_range = float(S_grid[-1] - S_grid[0])
    t_range = float(t_grid[-1] - t_grid[0])

    rows = []
    for nf in n_functions_list:
        for wf in width_factors:
            wS = S_range / wf
            wt = t_range / wf

            t_start = time.perf_counter()
            try:
                result = weak_sindy_discover(
                    V_clean, S_grid, t_grid,
                    n_test_functions=nf,
                    true_sigma=true_sigma, true_r=true_r,
                    seed=seed, width_S=wS, width_t=wt,
                )
                coeffs = result['discovered_coefficients']
                r2_clean = compute_r2_clean(
                    coeffs, S_grid, t_grid,
                    K=K, r=true_r, sigma=true_sigma, T=T,
                )
                cm = compute_coefficient_metrics(coeffs, true_r=true_r, true_sigma=true_sigma)
                elapsed = time.perf_counter() - t_start

                rows.append({
                    'n_functions': nf,
                    'width_factor': wf,
                    'width_S': wS,
                    'width_t': wt,
                    'r2_clean': r2_clean,
                    'r2_noisy': result['r2_score'],
                    'max_coeff_err': cm['max_coeff_rel_error'],
                    'mean_coeff_err': cm['mean_coeff_rel_error'],
                    'correct_structure': cm['correct_structure'],
                    'n_active': result['n_active'],
                    'condition_number': result['condition_number'],
                    'time_s': elapsed,
                })
            except Exception as e:
                elapsed = time.perf_counter() - t_start
                logger.warning(f"Weak SINDy tuning failed nf={nf}, wf={wf}: {e}")
                rows.append({
                    'n_functions': nf, 'width_factor': wf,
                    'width_S': wS, 'width_t': wt,
                    'r2_clean': float('nan'), 'r2_noisy': float('nan'),
                    'max_coeff_err': float('nan'), 'mean_coeff_err': float('nan'),
                    'correct_structure': False, 'n_active': 0,
                    'condition_number': float('nan'), 'time_s': elapsed,
                })

    results_df = pd.DataFrame(rows)
    valid = results_df.dropna(subset=['r2_clean'])
    if len(valid) > 0:
        best_idx = valid['r2_clean'].idxmax()
        best_row = valid.loc[best_idx]
        best_config = {
            'n_functions': int(best_row['n_functions']),
            'width_factor': int(best_row['width_factor']),
        }
        best_r2 = float(best_row['r2_clean'])
    else:
        best_config = {'n_functions': 100, 'width_factor': 20}
        best_r2 = float('nan')

    logger.info(
        f"Best weak SINDy config: nf={best_config['n_functions']}, "
        f"wf={best_config['width_factor']}, R²(clean)={best_r2:.4f}"
    )

    return {
        'results_df': results_df,
        'best_config': best_config,
        'best_r2_clean': best_r2,
    }
