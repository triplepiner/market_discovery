"""
PDE discovery via Sparse Identification of Nonlinear Dynamics (SINDy).

Discovers the Black-Scholes PDE from option price surface data using:
1. Numerical differentiation to compute partial derivatives
2. A candidate library of PDE terms
3. Sequential Thresholded Least Squares (STLSQ) sparse regression
"""

import numpy as np
from sklearn.linear_model import Ridge
from src.utils import set_all_seeds, setup_logging, NumericalDifferentiator, safe_relative_error
from src.data_generation import (
    bs_theta_call, bs_theta_put, bs_call_delta, bs_put_delta, bs_gamma
)

logger = setup_logging(__name__)

TERM_NAMES = ['V', 'dV/dS', 'd2V/dS2', 'S*dV/dS', 'S2*d2V/dS2']
REDUCED_TERM_NAMES = ['V', 'S*dV/dS', 'S2*d2V/dS2']


def compute_derivatives(V, S_grid, t_grid, smooth=False, savgol_window=7,
                        savgol_poly=3, trim=5):
    """
    Compute partial derivatives dV/dt, dV/dS, d2V/dS2 from a price surface.

    Trims boundary rows/columns to avoid finite difference artifacts.

    Parameters
    ----------
    V : ndarray, shape (n_S, n_t)
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    smooth : bool
        Apply Savitzky-Golay smoothing before differentiation.
    savgol_window : int
    savgol_poly : int
    trim : int
        Number of boundary rows/columns to trim from each edge.

    Returns
    -------
    dict with keys: 'V', 'dVdt', 'dVdS', 'd2VdS2', 'S_grid', 't_grid',
                    'S_mesh', 't_mesh'
    """
    dS = S_grid[1] - S_grid[0]
    dt = t_grid[1] - t_grid[0]

    diff = NumericalDifferentiator(
        order=2, smooth=smooth,
        savgol_window=savgol_window, savgol_poly=savgol_poly
    )

    # dV/dt: derivative along axis=1 (time)
    dVdt = diff.first_derivative(V, dt, axis=1)

    # dV/dS: derivative along axis=0 (stock price)
    dVdS = diff.first_derivative(V, dS, axis=0)

    # d2V/dS2: second derivative along axis=0 using direct stencil
    d2VdS2 = diff.second_derivative(V, dS, axis=0)

    # Trim boundaries
    s = slice(trim, -trim) if trim > 0 else slice(None)
    V_tr = V[s, s]
    dVdt_tr = dVdt[s, s]
    dVdS_tr = dVdS[s, s]
    d2VdS2_tr = d2VdS2[s, s]
    S_tr = S_grid[s]
    t_tr = t_grid[s]

    logger.info(
        f"Derivatives computed. Grid after trim: {V_tr.shape[0]}x{V_tr.shape[1]} "
        f"(trimmed {trim} from each edge)"
    )

    S_mesh_tr, t_mesh_tr = np.meshgrid(S_tr, t_tr, indexing='ij')

    return {
        'V': V_tr,
        'dVdt': dVdt_tr,
        'dVdS': dVdS_tr,
        'd2VdS2': d2VdS2_tr,
        'S_grid': S_tr,
        't_grid': t_tr,
        'S_mesh': S_mesh_tr,
        't_mesh': t_mesh_tr,
    }


def check_derivative_quality(deriv_dict, K, r, sigma, T, option_type='call'):
    """
    Compare numerical derivatives against analytical values.

    Returns dict with relative L2 errors for each derivative.
    """
    S_mesh = deriv_dict['S_mesh']
    t_mesh = deriv_dict['t_mesh']
    tau_mesh = T - t_mesh

    if option_type == 'call':
        theta_analytical = bs_theta_call(S_mesh, K, r, sigma, tau_mesh)
        delta_analytical = bs_call_delta(S_mesh, K, r, sigma, tau_mesh)
    else:
        theta_analytical = bs_theta_put(S_mesh, K, r, sigma, tau_mesh)
        delta_analytical = bs_put_delta(S_mesh, K, r, sigma, tau_mesh)
    gamma_analytical = bs_gamma(S_mesh, K, r, sigma, tau_mesh)

    def rel_l2(num, ana):
        denom = np.linalg.norm(ana)
        if denom < 1e-15:
            return 0.0
        return np.linalg.norm(num - ana) / denom

    errors = {
        'dVdt_rel_L2': rel_l2(deriv_dict['dVdt'], theta_analytical),
        'dVdS_rel_L2': rel_l2(deriv_dict['dVdS'], delta_analytical),
        'd2VdS2_rel_L2': rel_l2(deriv_dict['d2VdS2'], gamma_analytical),
    }

    for name, err in errors.items():
        if err > 0.10:
            logger.warning(f"Derivative quality poor: {name} = {err:.4f} (>10%)")
        else:
            logger.info(f"Derivative quality: {name} = {err:.6f}")

    return errors


def build_candidate_library(V_trimmed, dVdS, d2VdS2, S_mesh):
    """
    Build the SINDy candidate library matrix.

    Columns:
        0: V
        1: dV/dS
        2: d2V/dS2
        3: S * dV/dS
        4: S^2 * d2V/dS2

    Parameters
    ----------
    V_trimmed : ndarray
    dVdS : ndarray
    d2VdS2 : ndarray
    S_mesh : ndarray
        Stock price values at each grid point (same shape as V_trimmed).

    Returns
    -------
    library : ndarray, shape (n_points, 5)
    """
    n_points = V_trimmed.size

    library = np.column_stack([
        V_trimmed.ravel(),
        dVdS.ravel(),
        d2VdS2.ravel(),
        (S_mesh * dVdS).ravel(),
        (S_mesh ** 2 * d2VdS2).ravel(),
    ])

    # Condition number check
    cond = np.linalg.cond(library)
    logger.info(f"Library condition number: {cond:.2e}")
    if cond > 1e10:
        logger.warning(
            f"Library is ill-conditioned (cond={cond:.2e}). "
            "Results may be unreliable."
        )

    # Pairwise correlation check
    corr_matrix = np.corrcoef(library.T)
    for i in range(5):
        for j in range(i + 1, 5):
            if abs(corr_matrix[i, j]) > 0.95:
                logger.warning(
                    f"High correlation ({corr_matrix[i, j]:.3f}) between "
                    f"'{TERM_NAMES[i]}' and '{TERM_NAMES[j]}'"
                )

    return library


def stlsq(library, target, threshold, max_iter=20):
    """
    Sequential Thresholded Least Squares for a single threshold.

    Parameters
    ----------
    library : ndarray, shape (n, p)
    target : ndarray, shape (n,)
    threshold : float
    max_iter : int

    Returns
    -------
    coefficients : ndarray, shape (p,)
    active_mask : ndarray of bool, shape (p,)
    """
    n, p = library.shape
    active = np.ones(p, dtype=bool)
    coeffs = np.zeros(p)

    for iteration in range(max_iter):
        if not np.any(active):
            # All terms zeroed out — return zeros
            coeffs = np.zeros(p)
            break
        # Solve least squares on active columns
        lib_active = library[:, active]
        c, _, _, _ = np.linalg.lstsq(lib_active, target, rcond=None)
        coeffs = np.zeros(p)
        coeffs[active] = c

        # Threshold
        new_active = np.abs(coeffs) >= threshold
        if np.array_equal(new_active, active):
            break
        active = new_active

    # Final solve on active columns
    if np.any(active):
        lib_active = library[:, active]
        c, _, _, _ = np.linalg.lstsq(lib_active, target, rcond=None)
        coeffs = np.zeros(p)
        coeffs[active] = c
    else:
        coeffs = np.zeros(p)

    return coeffs, active


def stlsq_sweep(library, target, thresholds=None, r2_min=0.99):
    """
    Run STLSQ over a range of thresholds and select the best.

    Selection: among thresholds with R^2 > r2_min, pick the one with
    fewest active terms. Ties broken by BIC.

    Parameters
    ----------
    library : ndarray, shape (n, p)
    target : ndarray, shape (n,)
    thresholds : array-like or None
    r2_min : float

    Returns
    -------
    best : dict with keys 'coefficients', 'active_mask', 'threshold',
           'r2', 'bic', 'n_active'
    sweep_results : list of dicts
    """
    if thresholds is None:
        # Dense sampling to find intermediate sparsity solutions
        thresholds = np.sort(np.unique(np.concatenate([
            np.logspace(-3, np.log10(2.0), 30),
            np.linspace(0.001, 0.1, 20),
        ])))

    n = len(target)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    if ss_tot < 1e-30:
        ss_tot = 1e-30

    sweep_results = []
    for thr in thresholds:
        coeffs, active = stlsq(library, target, thr)
        residual = target - library @ coeffs
        rss = np.sum(residual ** 2)
        r2 = 1.0 - rss / ss_tot
        k = np.sum(active)
        bic = n * np.log(max(rss / n, 1e-30)) + k * np.log(n)

        sweep_results.append({
            'threshold': thr,
            'coefficients': coeffs.copy(),
            'active_mask': active.copy(),
            'n_active': int(k),
            'r2': r2,
            'bic': bic,
            'rss': rss,
        })

    # Select best: among those with R^2 > r2_min and n_active > 0, use BIC
    candidates = [r for r in sweep_results if r['r2'] > r2_min and r['n_active'] > 0]
    if not candidates:
        logger.warning(
            f"No threshold achieves R^2 > {r2_min} with active terms. Relaxing to R^2 > 0.95."
        )
        candidates = [r for r in sweep_results if r['r2'] > 0.95 and r['n_active'] > 0]
    if not candidates:
        logger.warning("No threshold achieves R^2 > 0.95. Using best R^2.")
        candidates = sorted(
            [r for r in sweep_results if r['n_active'] > 0],
            key=lambda x: -x['r2']
        )[:5]
    if not candidates:
        # Fallback: pick the overall best R^2 even with 0 active
        candidates = sorted(sweep_results, key=lambda x: -x['r2'])[:1]

    # Primary criterion: lowest BIC (balances fit and sparsity)
    candidates.sort(key=lambda x: x['bic'])
    best = candidates[0]

    logger.info(
        f"Best threshold: {best['threshold']:.4f}, "
        f"R^2={best['r2']:.6f}, "
        f"active terms={best['n_active']}, "
        f"BIC={best['bic']:.1f}"
    )

    return best, sweep_results


def format_pde_string(coefficients, term_names=None, threshold=1e-6):
    """
    Format discovered PDE as a human-readable string.

    Parameters
    ----------
    coefficients : array-like
    term_names : list of str or None
    threshold : float
        Terms with |coeff| < threshold are omitted.

    Returns
    -------
    str
    """
    if term_names is None:
        term_names = TERM_NAMES

    rhs_parts = []
    for c, name in zip(coefficients, term_names):
        if abs(c) < threshold:
            continue
        sign = '+' if c >= 0 else '-'
        rhs_parts.append(f"{sign}{abs(c):.6f}*{name}")

    if not rhs_parts:
        return "dV/dt = 0"

    rhs = ' '.join(rhs_parts)
    # Clean up leading +
    if rhs.startswith('+'):
        rhs = rhs[1:]
    return f"dV/dt = {rhs}"


def discover_pde(V, S_grid, t_grid, true_sigma=None, true_r=None,
                 smooth=False, K=100, T=1.0, option_type='call',
                 savgol_window=7, savgol_poly=3, trim=5):
    """
    Top-level PDE discovery: derivatives -> library -> STLSQ -> results.

    Parameters
    ----------
    V : ndarray, shape (n_S, n_t)
    S_grid : ndarray
    t_grid : ndarray
    true_sigma : float or None
    true_r : float or None
    smooth : bool
    K : float
    T : float
    option_type : str
    savgol_window : int
    savgol_poly : int
    trim : int

    Returns
    -------
    dict with discovery results
    """
    set_all_seeds(42)

    # Compute derivatives
    derivs = compute_derivatives(
        V, S_grid, t_grid, smooth=smooth,
        savgol_window=savgol_window, savgol_poly=savgol_poly, trim=trim
    )

    # Check derivative quality if analytical params known
    deriv_quality = {}
    if true_sigma is not None and true_r is not None:
        deriv_quality = check_derivative_quality(
            derivs, K, true_r, true_sigma, T, option_type
        )

    # Build library
    library = build_candidate_library(
        derivs['V'], derivs['dVdS'], derivs['d2VdS2'], derivs['S_mesh']
    )
    target = derivs['dVdt'].ravel()
    cond_number = np.linalg.cond(library)

    # Run STLSQ sweep
    best, sweep_results = stlsq_sweep(library, target)

    discovered = best['coefficients']

    # True coefficients
    true_coeffs = None
    rel_errors = None
    if true_sigma is not None and true_r is not None:
        true_coeffs = np.array([
            true_r,       # V
            0.0,          # dV/dS
            0.0,          # d2V/dS2
            -true_r,      # S*dV/dS
            -0.5 * true_sigma ** 2,  # S^2*d2V/dS2
        ])
        rel_errors = safe_relative_error(discovered, true_coeffs)

    active_terms = [TERM_NAMES[i] for i in range(5) if best['active_mask'][i]]
    pde_str = format_pde_string(discovered, TERM_NAMES)

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
        'derivative_quality': deriv_quality,
        'sweep_results': sweep_results,
        'human_readable_pde': pde_str,
        'active_mask': best['active_mask'],
        'n_active': best['n_active'],
    }


def post_process_coefficients(coefficients, term_names=None, relative_threshold=0.1):
    """
    Apply a secondary relative threshold to SINDy coefficients.

    Zeros out any coefficient whose magnitude is less than
    ``relative_threshold * max(|coefficients|)``, then checks whether the
    surviving active terms match the true Black-Scholes structure (indices
    0, 3, 4: V, S*dV/dS, S^2*d2V/dS2).

    Parameters
    ----------
    coefficients : ndarray, shape (p,)
        Coefficient vector from SINDy (length 5 for the standard library).
    term_names : list of str or None
        Human-readable names for each library column.  Defaults to
        ``TERM_NAMES``.
    relative_threshold : float
        Fraction of the largest absolute coefficient below which a term is
        zeroed out.

    Returns
    -------
    dict
        Keys: original_coefficients, post_processed_coefficients,
        original_active, post_processed_active, original_n_active,
        post_processed_n_active, correct_structure, threshold_used,
        relative_threshold, removed_terms.
    """
    if term_names is None:
        term_names = TERM_NAMES

    coefficients = np.asarray(coefficients, dtype=float)
    max_abs = np.max(np.abs(coefficients))
    threshold_used = relative_threshold * max_abs

    # Original active terms (anything non-zero)
    original_active_mask = np.abs(coefficients) > 0
    original_active = [term_names[i] for i in range(len(coefficients)) if original_active_mask[i]]

    # Post-processed coefficients
    pp_coefficients = coefficients.copy()
    pp_coefficients[np.abs(pp_coefficients) < threshold_used] = 0.0

    # Post-processed active terms
    pp_active_mask = np.abs(pp_coefficients) > 0
    pp_active = [term_names[i] for i in range(len(pp_coefficients)) if pp_active_mask[i]]

    # Terms that were removed
    removed = [term_names[i] for i in range(len(coefficients))
               if original_active_mask[i] and not pp_active_mask[i]]

    # Check correct structure: exactly indices 0, 3, 4 active
    pp_active_indices = set(np.where(pp_active_mask)[0])
    correct_structure = pp_active_indices == {0, 3, 4}

    logger.info(
        f"Post-processing: {int(original_active_mask.sum())} -> "
        f"{int(pp_active_mask.sum())} active terms "
        f"(threshold={threshold_used:.6f}, correct_structure={correct_structure})"
    )

    return {
        'original_coefficients': coefficients.copy(),
        'post_processed_coefficients': pp_coefficients,
        'original_active': original_active,
        'post_processed_active': pp_active,
        'original_n_active': int(original_active_mask.sum()),
        'post_processed_n_active': int(pp_active_mask.sum()),
        'correct_structure': correct_structure,
        'threshold_used': threshold_used,
        'relative_threshold': relative_threshold,
        'removed_terms': removed,
    }


def compute_library_correlations(library, term_names=None):
    """
    Compute the correlation matrix and condition number for a SINDy library.

    Parameters
    ----------
    library : ndarray, shape (n, p)
        The candidate library matrix.
    term_names : list of str or None
        Human-readable names for each column.  Defaults to ``TERM_NAMES``.

    Returns
    -------
    dict
        Keys: correlation_matrix, term_names, high_correlations,
        condition_number.
    """
    if term_names is None:
        term_names = TERM_NAMES

    corr_matrix = np.corrcoef(library.T)
    cond = np.linalg.cond(library)
    p = library.shape[1]

    high_correlations = []
    for i in range(p):
        for j in range(i + 1, p):
            if abs(corr_matrix[i, j]) > 0.9:
                high_correlations.append(
                    (term_names[i], term_names[j], float(corr_matrix[i, j]))
                )

    logger.info(
        f"Library correlations: {len(high_correlations)} pairs with |corr| > 0.9, "
        f"condition number = {cond:.2e}"
    )

    return {
        'correlation_matrix': corr_matrix,
        'term_names': list(term_names),
        'high_correlations': high_correlations,
        'condition_number': float(cond),
    }


def compute_r2_clean(discovered_coefficients, S_grid, t_grid, K, r, sigma, T,
                     option_type='call', trim=5):
    """
    Compute R²(clean): how well discovered PDE predicts analytical dV/dt
    using CLEAN analytical derivatives.

    R²(noisy) measures fit to the noisy target dV/dt, which misleadingly
    increases with noise. R²(clean) measures prediction accuracy against
    the true analytical dV/dt — the real accuracy metric.

    Parameters
    ----------
    discovered_coefficients : ndarray, shape (5,)
        Coefficients for [V, dV/dS, d2V/dS2, S*dV/dS, S2*d2V/dS2].
    S_grid, t_grid : ndarray
        Grid arrays (same grid the experiment used).
    K, r, sigma, T : float
        Black-Scholes parameters.
    option_type : str
    trim : int
        Boundary trim (must match what the SINDy run used).

    Returns
    -------
    float
        R²(clean) score.
    """
    from src.data_generation import (
        generate_price_surface, bs_theta_call, bs_theta_put,
        bs_call_delta, bs_put_delta, bs_gamma
    )

    # Generate clean surface on the same grid
    V_clean, _, _ = generate_price_surface(
        S_min=float(S_grid[0]), S_max=float(S_grid[-1]), n_S=len(S_grid),
        t_min=float(t_grid[0]), t_max=float(t_grid[-1]), n_t=len(t_grid),
        K=K, r=r, sigma=sigma, T=T, option_type=option_type,
    )

    # Trim boundaries
    s = slice(trim, -trim) if trim > 0 else slice(None)
    S_tr = S_grid[s]
    t_tr = t_grid[s]
    S_mesh, t_mesh = np.meshgrid(S_tr, t_tr, indexing='ij')
    tau_mesh = T - t_mesh

    # Clean analytical derivatives
    if option_type == 'call':
        theta_clean = bs_theta_call(S_mesh, K, r, sigma, tau_mesh)
        delta_clean = bs_call_delta(S_mesh, K, r, sigma, tau_mesh)
    else:
        theta_clean = bs_theta_put(S_mesh, K, r, sigma, tau_mesh)
        delta_clean = bs_put_delta(S_mesh, K, r, sigma, tau_mesh)
    gamma_clean = bs_gamma(S_mesh, K, r, sigma, tau_mesh)
    V_tr_clean = V_clean[s, s]

    # Build predicted dV/dt from discovered coefficients + clean library
    c = np.asarray(discovered_coefficients)
    dVdt_pred = (c[0] * V_tr_clean +
                 c[1] * delta_clean +
                 c[2] * gamma_clean +
                 c[3] * S_mesh * delta_clean +
                 c[4] * S_mesh**2 * gamma_clean)

    # R²(clean) = 1 - SS_res / SS_tot
    pred_flat = dVdt_pred.ravel()
    true_flat = theta_clean.ravel()
    ss_res = np.sum((pred_flat - true_flat)**2)
    ss_tot = np.sum((true_flat - np.mean(true_flat))**2)

    if ss_tot < 1e-30:
        return 0.0

    return float(1.0 - ss_res / ss_tot)


def compute_coefficient_metrics(discovered_coefficients, true_r, true_sigma):
    """
    Compute per-coefficient accuracy metrics for discovered PDE.

    Parameters
    ----------
    discovered_coefficients : ndarray, shape (5,)
    true_r : float
    true_sigma : float

    Returns
    -------
    dict with keys:
        coeff_V, coeff_SdVdS, coeff_S2d2VdS2 : discovered values
        true_V, true_SdVdS, true_S2d2VdS2 : true values
        rel_err_V, rel_err_SdVdS, rel_err_S2d2VdS2 : relative errors
        max_coeff_rel_error, mean_coeff_rel_error : summary stats
        correct_structure : bool
    """
    c = np.asarray(discovered_coefficients)
    true_coeffs = np.array([true_r, 0.0, 0.0, -true_r, -0.5 * true_sigma**2])

    # Per-coefficient relative errors for the 3 true active terms
    def _rel_err(disc, true):
        if abs(true) < 1e-15:
            return abs(disc)  # absolute error when true is zero
        return abs(disc - true) / abs(true)

    rel_V = _rel_err(c[0], true_coeffs[0])
    rel_SdVdS = _rel_err(c[3], true_coeffs[3])
    rel_S2d2VdS2 = _rel_err(c[4], true_coeffs[4])

    active_mask = np.abs(c) > 1e-10
    discovered_active = set(np.where(active_mask)[0])
    correct_structure = discovered_active == {0, 3, 4}

    return {
        'coeff_V': float(c[0]),
        'coeff_SdVdS': float(c[3]),
        'coeff_S2d2VdS2': float(c[4]),
        'true_V': float(true_coeffs[0]),
        'true_SdVdS': float(true_coeffs[3]),
        'true_S2d2VdS2': float(true_coeffs[4]),
        'rel_err_V': float(rel_V),
        'rel_err_SdVdS': float(rel_SdVdS),
        'rel_err_S2d2VdS2': float(rel_S2d2VdS2),
        'max_coeff_rel_error': float(max(rel_V, rel_SdVdS, rel_S2d2VdS2)),
        'mean_coeff_rel_error': float(np.mean([rel_V, rel_SdVdS, rel_S2d2VdS2])),
        'correct_structure': correct_structure,
    }


def analyze_full_library_result(result_dict, true_sigma, true_r):
    """
    Reframe a 5-term library SINDy result separating true-term accuracy
    from false positives, with dimensional analysis note.

    Parameters
    ----------
    result_dict : dict
        Output from discover_pde() or sindy_with_neural_derivatives().
    true_sigma : float
    true_r : float

    Returns
    -------
    dict with keys:
        true_term_coefficients : dict mapping term name -> discovered value
        true_term_errors : dict mapping term name -> relative error
        spurious_term_coefficients : dict mapping term name -> discovered value
        correlation_matrix : ndarray (5x5)
        dimensional_analysis_note : str
    """
    c = np.asarray(result_dict['discovered_coefficients'])
    true_coeffs = np.array([true_r, 0.0, 0.0, -true_r, -0.5 * true_sigma**2])

    # True terms: indices 0 (V), 3 (S*dV/dS), 4 (S2*d2V/dS2)
    true_indices = [0, 3, 4]
    spurious_indices = [1, 2]

    true_term_coefficients = {}
    true_term_errors = {}
    for idx in true_indices:
        name = TERM_NAMES[idx]
        true_term_coefficients[name] = float(c[idx])
        true_val = true_coeffs[idx]
        if abs(true_val) > 1e-15:
            true_term_errors[name] = float(abs(c[idx] - true_val) / abs(true_val))
        else:
            true_term_errors[name] = float(abs(c[idx]))

    spurious_term_coefficients = {}
    for idx in spurious_indices:
        name = TERM_NAMES[idx]
        if abs(c[idx]) > 1e-10:
            spurious_term_coefficients[name] = float(c[idx])

    # Build correlation matrix from the library if available
    correlation_matrix = None
    if 'sweep_results' in result_dict and result_dict['sweep_results']:
        # We don't have the library directly, but condition_number hints at it
        correlation_matrix = None  # caller can compute from library

    dimensional_note = (
        "Note on spurious terms: dV/dS has units [$/S] while S*dV/dS has units [$]. "
        "Similarly d2V/dS2 has units [$/S^2] while S^2*d2V/dS2 has units [$]. "
        "High correlation (>0.96) between bare and S-weighted derivatives makes "
        "the 5-term library ill-conditioned. Spurious terms with small coefficients "
        "are regression artifacts, not physical contributions to the PDE."
    )

    logger.info(
        f"Full library analysis: true-term errors = "
        f"{', '.join(f'{k}={v:.4f}' for k, v in true_term_errors.items())}, "
        f"spurious terms = {list(spurious_term_coefficients.keys())}"
    )

    return {
        'true_term_coefficients': true_term_coefficients,
        'true_term_errors': true_term_errors,
        'spurious_term_coefficients': spurious_term_coefficients,
        'correlation_matrix': correlation_matrix,
        'dimensional_analysis_note': dimensional_note,
    }


def build_reduced_library(V_trimmed, dVdS, d2VdS2, S_mesh):
    """
    Build a reduced 3-term SINDy library excluding bare derivative terms.

    Removes dV/dS and d2V/dS2 (which are highly correlated with their
    S-weighted counterparts), keeping only the three terms present in the
    true Black-Scholes PDE: V, S*dV/dS, S^2*d2V/dS2.

    Returns
    -------
    library : ndarray, shape (n_points, 3)
    """
    library = np.column_stack([
        V_trimmed.ravel(),
        (S_mesh * dVdS).ravel(),
        (S_mesh ** 2 * d2VdS2).ravel(),
    ])

    cond = np.linalg.cond(library)
    logger.info(f"Reduced library condition number: {cond:.2e}")

    corr_matrix = np.corrcoef(library.T)
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(corr_matrix[i, j]) > 0.95:
                logger.warning(
                    f"High correlation ({corr_matrix[i, j]:.3f}) between "
                    f"'{REDUCED_TERM_NAMES[i]}' and '{REDUCED_TERM_NAMES[j]}'"
                )

    return library


def discover_pde_reduced(V, S_grid, t_grid, true_sigma=None, true_r=None,
                         smooth=False, K=100, T=1.0, option_type='call',
                         savgol_window=7, savgol_poly=3, trim=5):
    """
    PDE discovery using the reduced 3-term library (no bare derivatives).

    This eliminates multicollinearity by removing dV/dS and d2V/dS2,
    keeping only the three terms that appear in the true Black-Scholes PDE.

    Returns
    -------
    dict with discovery results (same structure as discover_pde but with 3 terms)
    """
    set_all_seeds(42)

    derivs = compute_derivatives(
        V, S_grid, t_grid, smooth=smooth,
        savgol_window=savgol_window, savgol_poly=savgol_poly, trim=trim
    )

    library = build_reduced_library(
        derivs['V'], derivs['dVdS'], derivs['d2VdS2'], derivs['S_mesh']
    )
    target = derivs['dVdt'].ravel()
    cond_number = np.linalg.cond(library)

    best, sweep_results = stlsq_sweep(library, target)

    discovered = best['coefficients']

    true_coeffs = None
    rel_errors = None
    if true_sigma is not None and true_r is not None:
        true_coeffs = np.array([
            true_r,                      # V
            -true_r,                     # S*dV/dS
            -0.5 * true_sigma ** 2,      # S^2*d2V/dS2
        ])
        rel_errors = safe_relative_error(discovered, true_coeffs)

    active_terms = [REDUCED_TERM_NAMES[i] for i in range(3) if best['active_mask'][i]]
    pde_str = format_pde_string(discovered, REDUCED_TERM_NAMES)

    return {
        'discovered_coefficients': discovered,
        'true_coefficients': true_coeffs,
        'active_terms': active_terms,
        'term_names': REDUCED_TERM_NAMES,
        'relative_errors': rel_errors,
        'best_threshold': best['threshold'],
        'r2_score': best['r2'],
        'bic': best['bic'],
        'condition_number': cond_number,
        'sweep_results': sweep_results,
        'human_readable_pde': pde_str,
        'active_mask': best['active_mask'],
        'n_active': best['n_active'],
    }


# ---------------------------------------------------------------------------
# Advanced SINDy: ensemble, PCA, time-varying, CV thresholds, bootstrap CIs
# ---------------------------------------------------------------------------


def _build_library_and_target(V, S_grid, t_grid, smooth=True, trim=5,
                              savgol_window=7, savgol_poly=3):
    """Helper: compute derivatives, build the standard 5-term library and target."""
    derivs = compute_derivatives(
        V, S_grid, t_grid, smooth=smooth,
        savgol_window=savgol_window, savgol_poly=savgol_poly, trim=trim,
    )
    library = build_candidate_library(
        derivs['V'], derivs['dVdS'], derivs['d2VdS2'], derivs['S_mesh']
    )
    target = derivs['dVdt'].ravel()
    return library, target, derivs


def ensemble_sindy(V, S_grid, t_grid, threshold=0.1, n_bootstraps=50,
                   subsample_frac=0.7, smooth=True, seed=42):
    """
    Ensemble SINDy via subsampling.

    Run STLSQ on ``n_bootstraps`` random subsamples (without replacement) of
    the rows of the candidate library/target.  Record which terms are active
    and the coefficient values.  Compute per-term inclusion probability and
    coefficient quantiles (median, 2.5%, 97.5%).  Final selected terms are
    those with inclusion_probability > 0.6.

    Returns
    -------
    dict with keys: term_names, inclusion_probabilities, median_coefficients,
                    ci_low, ci_high, selected_terms, n_bootstraps
    """
    library, target, _ = _build_library_and_target(
        V, S_grid, t_grid, smooth=smooth
    )
    n, p = library.shape
    n_sub = max(p + 1, int(round(subsample_frac * n)))
    n_sub = min(n_sub, n)

    rng = np.random.default_rng(seed)
    coeffs_runs = np.zeros((n_bootstraps, p))
    active_runs = np.zeros((n_bootstraps, p), dtype=bool)

    for b in range(n_bootstraps):
        idx = rng.choice(n, size=n_sub, replace=False)
        lib_b = library[idx]
        tgt_b = target[idx]
        coeffs, active = stlsq(lib_b, tgt_b, threshold)
        coeffs_runs[b] = coeffs
        active_runs[b] = active

    inclusion_prob = active_runs.mean(axis=0)
    median_coeffs = np.median(coeffs_runs, axis=0)
    ci_low = np.percentile(coeffs_runs, 2.5, axis=0)
    ci_high = np.percentile(coeffs_runs, 97.5, axis=0)

    selected_terms = [TERM_NAMES[i] for i in range(p) if inclusion_prob[i] > 0.6]

    logger.info(
        f"Ensemble SINDy ({n_bootstraps} runs): inclusion probabilities = "
        + ", ".join(f"{TERM_NAMES[i]}={inclusion_prob[i]:.2f}" for i in range(p))
    )

    return {
        'term_names': list(TERM_NAMES),
        'inclusion_probabilities': inclusion_prob,
        'median_coefficients': median_coeffs,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'selected_terms': selected_terms,
        'n_bootstraps': n_bootstraps,
    }


def pca_sindy(V, S_grid, t_grid, threshold=0.1, secondary_threshold=0.05,
              smooth=True):
    """
    PCA-SINDy: standardize the library, project to PCA space, run STLSQ in
    that orthogonal basis, then transform coefficients back to the original
    library basis.

    Returns
    -------
    dict with keys: discovered_coefficients, active_terms, r2_score,
                    term_names, n_active
    """
    from sklearn.decomposition import PCA

    library, target, _ = _build_library_and_target(
        V, S_grid, t_grid, smooth=smooth
    )
    n, p = library.shape

    # Standardize columns
    mean = library.mean(axis=0)
    std = library.std(axis=0)
    std_safe = np.where(std < 1e-15, 1.0, std)
    lib_std = (library - mean) / std_safe

    # PCA
    pca = PCA(n_components=p)
    lib_pca = pca.fit_transform(lib_std)

    # STLSQ in PCA space
    pca_coeffs, _ = stlsq(lib_pca, target, threshold)

    # Transform back to standardized space, then to original space
    # lib_pca = (lib_std - pca.mean_) @ components_.T  (here pca.mean_ ~ 0)
    # target ~= lib_pca @ pca_coeffs
    #        = (lib_std - pca.mean_) @ components_.T @ pca_coeffs
    # So coefficients in standardized basis: components_.T @ pca_coeffs
    std_basis_coeffs = pca.components_.T @ pca_coeffs
    # Map back to original (unstandardized) basis: c_orig_i = c_std_i / std_i
    orig_coeffs = std_basis_coeffs / std_safe

    # Apply secondary threshold (relative to max absolute coefficient)
    max_abs = np.max(np.abs(orig_coeffs))
    if max_abs > 0:
        orig_coeffs[np.abs(orig_coeffs) < secondary_threshold * max_abs] = 0.0

    # R^2
    pred = library @ orig_coeffs
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0

    active_mask = np.abs(orig_coeffs) > 0
    active_terms = [TERM_NAMES[i] for i in range(p) if active_mask[i]]

    logger.info(
        f"PCA-SINDy: n_active={int(active_mask.sum())}, R^2={r2:.6f}"
    )

    return {
        'discovered_coefficients': orig_coeffs,
        'active_terms': active_terms,
        'r2_score': float(r2),
        'term_names': list(TERM_NAMES),
        'n_active': int(active_mask.sum()),
    }


def time_varying_sindy(V, S_grid, t_grid, window_size=20, stride=5,
                       threshold=0.1, smooth=True):
    """
    Time-varying SINDy: slide a window across the time axis and fit a SINDy
    model on each slice.  Useful for detecting non-autonomous dynamics.

    Returns
    -------
    dict with keys: window_centers, coefficients_per_window (n_windows x n_terms),
                    r2_per_window, is_autonomous
    """
    n_t = len(t_grid)
    if window_size > n_t:
        raise ValueError(
            f"window_size={window_size} exceeds n_t={n_t}"
        )

    starts = list(range(0, n_t - window_size + 1, stride))
    window_centers = []
    coeffs_list = []
    r2_list = []

    for start in starts:
        end = start + window_size
        V_win = V[:, start:end]
        t_win = t_grid[start:end]
        center = 0.5 * (t_win[0] + t_win[-1])
        try:
            result = discover_pde(
                V_win, S_grid, t_win,
                smooth=smooth,
                # Use a smaller trim for short windows
                trim=min(5, max(1, window_size // 6)),
            )
            coeffs_list.append(result['discovered_coefficients'])
            r2_list.append(result['r2_score'])
            window_centers.append(center)
        except Exception as e:
            logger.warning(f"Window at t={center:.3f} failed: {e}")
            continue

    coefficients_per_window = np.array(coeffs_list)  # (n_windows, n_terms)
    r2_per_window = np.array(r2_list)
    window_centers = np.array(window_centers)

    # Autonomous score: max over the true-term indices (0, 3, 4) of std/|mean|.
    # Low score => coefficients stable across windows => autonomous.
    is_autonomous = True
    autonomous_threshold = 0.15
    if coefficients_per_window.size > 0:
        scores = []
        for idx in [0, 3, 4]:
            col = coefficients_per_window[:, idx]
            m = np.mean(col)
            if abs(m) > 1e-10:
                scores.append(float(np.std(col) / abs(m)))
        if scores:
            is_autonomous = bool(max(scores) < autonomous_threshold)

    logger.info(
        f"Time-varying SINDy: {len(coeffs_list)} windows, "
        f"is_autonomous={is_autonomous}"
    )

    return {
        'window_centers': window_centers,
        'coefficients_per_window': coefficients_per_window,
        'r2_per_window': r2_per_window,
        'is_autonomous': is_autonomous,
    }


def cv_threshold_select(V, S_grid, t_grid, candidate_thresholds=None,
                        n_folds=5, smooth=True, seed=42):
    """
    Cross-validated threshold selection.

    For each candidate threshold, split row indices into ``n_folds`` folds,
    train STLSQ on n_folds-1 folds and evaluate R^2 on the held-out fold.
    Pick the threshold whose mean CV R^2 is highest among those tied for
    the fewest active terms.

    Returns
    -------
    best_threshold : float
    cv_scores : dict {threshold -> mean_cv_r2}
    """
    if candidate_thresholds is None:
        candidate_thresholds = np.logspace(-3, 0, 10)

    library, target, _ = _build_library_and_target(
        V, S_grid, t_grid, smooth=smooth
    )
    n, p = library.shape

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_size = n // n_folds
    folds = [perm[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
    # Remainder goes into the last fold
    if n_folds * fold_size < n:
        folds[-1] = np.concatenate([folds[-1], perm[n_folds * fold_size:]])

    cv_scores = {}
    n_active_per_thr = {}
    for thr in candidate_thresholds:
        fold_r2s = []
        fold_active = []
        for k in range(n_folds):
            test_idx = folds[k]
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            lib_tr = library[train_mask]
            tgt_tr = target[train_mask]
            lib_te = library[test_idx]
            tgt_te = target[test_idx]

            coeffs, active = stlsq(lib_tr, tgt_tr, thr)
            pred = lib_te @ coeffs
            ss_res = np.sum((tgt_te - pred) ** 2)
            ss_tot = np.sum((tgt_te - tgt_te.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
            fold_r2s.append(r2)
            fold_active.append(int(np.sum(active)))

        cv_scores[float(thr)] = float(np.mean(fold_r2s))
        n_active_per_thr[float(thr)] = float(np.mean(fold_active))

    # Tie-break: fewest active terms, then highest CV score
    min_active = min(n_active_per_thr.values())
    # Allow a tolerance of 0.5 (since the fold averages can be fractional)
    tied = [t for t, k in n_active_per_thr.items() if k <= min_active + 0.5]
    best_threshold = max(tied, key=lambda t: cv_scores[t])

    logger.info(
        f"CV threshold selection: best={best_threshold:.4f} "
        f"(CV R^2={cv_scores[best_threshold]:.4f})"
    )

    return float(best_threshold), cv_scores


def bootstrap_confidence_intervals(V, S_grid, t_grid, threshold=0.1,
                                   n_bootstraps=100, smooth=True, seed=42):
    """
    Bootstrap 95% confidence intervals for each library coefficient at a
    fixed threshold.

    Returns
    -------
    pandas.DataFrame with columns: term, point_estimate, ci_low, ci_high,
                                   ci_contains_zero
    """
    import pandas as pd

    library, target, _ = _build_library_and_target(
        V, S_grid, t_grid, smooth=smooth
    )
    n, p = library.shape

    # Point estimate on the full dataset
    point_coeffs, _ = stlsq(library, target, threshold)

    rng = np.random.default_rng(seed)
    boot_coeffs = np.zeros((n_bootstraps, p))
    for b in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)  # with replacement
        lib_b = library[idx]
        tgt_b = target[idx]
        c, _ = stlsq(lib_b, tgt_b, threshold)
        boot_coeffs[b] = c

    ci_low = np.percentile(boot_coeffs, 2.5, axis=0)
    ci_high = np.percentile(boot_coeffs, 97.5, axis=0)

    df = pd.DataFrame({
        'term': TERM_NAMES,
        'point_estimate': point_coeffs,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'ci_contains_zero': (ci_low <= 0) & (ci_high >= 0),
    })

    logger.info(f"Bootstrap CIs computed from {n_bootstraps} resamples")
    return df
