"""
Baseline PDE discovery methods for comparison against SINDy STLSQ.

Implements four alternative regression/discovery approaches that all operate
on the same candidate library and target vector as the SINDy pipeline:

1. Dense OLS regression (no sparsity)
2. Lasso with cross-validated regularisation path
3. Ridge regression followed by hard thresholding with BIC selection
4. Symbolic regression via genetic programming (gplearn)

Each baseline returns a standardised result dict for easy comparison.
"""

import warnings
import signal
import numpy as np

from src.utils import set_all_seeds, setup_logging, safe_relative_error
from src.sindy_discovery import compute_derivatives, build_candidate_library, TERM_NAMES
from src.data_generation import add_noise

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_r2(library, target, coefficients):
    """
    Compute the coefficient of determination R².

    R² = 1 - SS_res / SS_tot

    Parameters
    ----------
    library : ndarray, shape (n, p)
    target : ndarray, shape (n,)
    coefficients : ndarray, shape (p,)

    Returns
    -------
    float
        R² value.  Returns 0.0 when SS_tot is effectively zero.
    """
    prediction = library @ coefficients
    ss_res = np.sum((target - prediction) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    if ss_tot < 1e-30:
        logger.warning("SS_tot near zero; R² undefined, returning 0.0")
        return 0.0
    return 1.0 - ss_res / ss_tot


def _compute_bic(n, k, sse):
    """
    Compute the Bayesian Information Criterion.

    BIC = n * log(SSE / n) + k * log(n)

    Parameters
    ----------
    n : int
        Number of observations.
    k : int
        Number of active (nonzero) parameters.
    sse : float
        Sum of squared errors (residuals).

    Returns
    -------
    float
        BIC value.  Returns +inf for degenerate cases.
    """
    if n <= 0:
        return np.inf
    if sse <= 0:
        sse = 1e-30  # avoid log(0)
    return n * np.log(sse / n) + k * np.log(n)


# ---------------------------------------------------------------------------
# BASELINE 1: Dense OLS regression
# ---------------------------------------------------------------------------

def dense_regression(library, target):
    """
    Ordinary least-squares regression (no sparsity enforcement).

    Parameters
    ----------
    library : ndarray, shape (n, p)
        Candidate library matrix.
    target : ndarray, shape (n,)
        Target vector (dV/dt).

    Returns
    -------
    dict
        coefficients : ndarray, shape (p,)
        r2 : float
        active_mask : ndarray of bool (|coeff| > 0.01)
        n_active : int
        method : 'dense'
    """
    logger.info("Running dense OLS regression")

    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(library, target, rcond=None)
    except np.linalg.LinAlgError as exc:
        logger.error(f"Dense regression failed (singular matrix): {exc}")
        p = library.shape[1]
        coeffs = np.zeros(p)

    r2 = _compute_r2(library, target, coeffs)
    active_mask = np.abs(coeffs) > 0.01
    n_active = int(np.sum(active_mask))

    logger.info(
        f"Dense OLS: R²={r2:.6f}, active terms={n_active}, "
        f"coeffs={np.array2string(coeffs, precision=6)}"
    )

    return {
        'coefficients': coeffs,
        'r2': r2,
        'active_mask': active_mask,
        'n_active': n_active,
        'method': 'dense',
    }


# ---------------------------------------------------------------------------
# BASELINE 2: Lasso with cross-validated alpha
# ---------------------------------------------------------------------------

def lasso_regression(library, target, n_alphas=50):
    """
    Lasso regression with cross-validated regularisation strength.

    Uses LassoCV with 50 alphas log-spaced from 1e-6 to 1.0, plus
    sklearn.linear_model.lasso_path for the full regularisation path
    (useful for plotting coefficient trajectories).

    Parameters
    ----------
    library : ndarray, shape (n, p)
    target : ndarray, shape (n,)
    n_alphas : int
        Number of alpha values to try.

    Returns
    -------
    dict
        coefficients : ndarray, shape (p,)
        r2 : float
        active_mask : ndarray of bool (nonzero coefficients)
        n_active : int
        best_alpha : float
        method : 'lasso'
        lasso_path : dict with 'alphas' and 'coefs' arrays for plotting
    """
    from sklearn.linear_model import LassoCV, lasso_path

    logger.info(f"Running Lasso regression with {n_alphas} alphas")

    alphas = np.logspace(-6, 0, n_alphas)

    try:
        model = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
        model.fit(library, target)
        coeffs = model.coef_
        best_alpha = model.alpha_
    except Exception as exc:
        logger.error(f"LassoCV failed: {exc}")
        p = library.shape[1]
        coeffs = np.zeros(p)
        best_alpha = np.nan

    # Compute full Lasso path for plotting
    lasso_path_dict = {'alphas': np.array([]), 'coefs': np.array([])}
    try:
        path_alphas, path_coefs, _ = lasso_path(
            library, target, alphas=alphas, max_iter=10000
        )
        lasso_path_dict = {
            'alphas': path_alphas,
            'coefs': path_coefs,  # shape (p, n_alphas)
        }
    except Exception as exc:
        logger.warning(f"lasso_path computation failed: {exc}")

    r2 = _compute_r2(library, target, coeffs)
    active_mask = np.abs(coeffs) > 0.0
    n_active = int(np.sum(active_mask))

    logger.info(
        f"Lasso: best_alpha={best_alpha:.2e}, R²={r2:.6f}, "
        f"active terms={n_active}, "
        f"coeffs={np.array2string(coeffs, precision=6)}"
    )

    return {
        'coefficients': coeffs,
        'r2': r2,
        'active_mask': active_mask,
        'n_active': n_active,
        'best_alpha': best_alpha,
        'method': 'lasso',
        'lasso_path': lasso_path_dict,
    }


# ---------------------------------------------------------------------------
# BASELINE 3: Ridge + hard thresholding with BIC selection
# ---------------------------------------------------------------------------

def ridge_threshold(library, target, thresholds=None):
    """
    Ridge regression followed by hard coefficient thresholding.

    Fits RidgeCV first, then sweeps a range of thresholds: for each
    threshold, coefficients with |c| < threshold are zeroed out and the
    resulting R² is computed.  The best threshold is chosen by BIC among
    candidates whose R² exceeds 0.99.

    Parameters
    ----------
    library : ndarray, shape (n, p)
    target : ndarray, shape (n,)
    thresholds : array-like or None
        Hard thresholds to sweep.  Defaults to 30 values log-spaced
        from 0.001 to 1.0.

    Returns
    -------
    dict
        coefficients : ndarray, shape (p,)  (after thresholding)
        r2 : float
        active_mask : ndarray of bool
        n_active : int
        best_threshold : float
        best_ridge_alpha : float
        method : 'ridge_threshold'
    """
    from sklearn.linear_model import RidgeCV

    logger.info("Running Ridge + threshold regression")

    if thresholds is None:
        thresholds = np.logspace(np.log10(0.001), np.log10(1.0), 30)

    # Fit RidgeCV
    ridge_alphas = np.logspace(-4, 4, 50)
    try:
        model = RidgeCV(alphas=ridge_alphas, cv=5)
        model.fit(library, target)
        base_coeffs = model.coef_
        best_ridge_alpha = model.alpha_
    except Exception as exc:
        logger.error(f"RidgeCV failed: {exc}")
        p = library.shape[1]
        base_coeffs = np.zeros(p)
        best_ridge_alpha = np.nan

    logger.info(
        f"Ridge base coeffs: {np.array2string(base_coeffs, precision=6)}, "
        f"alpha={best_ridge_alpha}"
    )

    n = len(target)

    # Sweep thresholds
    best_bic = np.inf
    best_thr = thresholds[0]
    best_coeffs = base_coeffs.copy()

    for thr in thresholds:
        coeffs_thr = base_coeffs.copy()
        coeffs_thr[np.abs(coeffs_thr) < thr] = 0.0
        k = int(np.sum(np.abs(coeffs_thr) > 0))

        r2_thr = _compute_r2(library, target, coeffs_thr)
        residual = target - library @ coeffs_thr
        sse = np.sum(residual ** 2)
        bic = _compute_bic(n, k, sse)

        # Only consider candidates with R² > 0.99 (or at least some active terms)
        if r2_thr > 0.99 and bic < best_bic:
            best_bic = bic
            best_thr = thr
            best_coeffs = coeffs_thr.copy()

    # If no threshold achieved R² > 0.99, fall back to the one with best BIC overall
    if np.isinf(best_bic):
        logger.warning(
            "No threshold achieved R² > 0.99 after Ridge; "
            "falling back to best BIC across all thresholds."
        )
        for thr in thresholds:
            coeffs_thr = base_coeffs.copy()
            coeffs_thr[np.abs(coeffs_thr) < thr] = 0.0
            k = int(np.sum(np.abs(coeffs_thr) > 0))
            residual = target - library @ coeffs_thr
            sse = np.sum(residual ** 2)
            bic = _compute_bic(n, k, sse)
            if bic < best_bic:
                best_bic = bic
                best_thr = thr
                best_coeffs = coeffs_thr.copy()

    r2 = _compute_r2(library, target, best_coeffs)
    active_mask = np.abs(best_coeffs) > 0.0
    n_active = int(np.sum(active_mask))

    logger.info(
        f"Ridge+threshold: best_threshold={best_thr:.4f}, "
        f"ridge_alpha={best_ridge_alpha:.2e}, R²={r2:.6f}, "
        f"active terms={n_active}, "
        f"coeffs={np.array2string(best_coeffs, precision=6)}"
    )

    return {
        'coefficients': best_coeffs,
        'r2': r2,
        'active_mask': active_mask,
        'n_active': n_active,
        'best_threshold': best_thr,
        'best_ridge_alpha': best_ridge_alpha,
        'method': 'ridge_threshold',
    }


# ---------------------------------------------------------------------------
# BASELINE 4: Symbolic regression (gplearn)
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    """Raised when symbolic regression exceeds the time budget."""
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("Symbolic regression exceeded time limit")


def symbolic_regression(library, target, feature_names=None, max_time=300):
    """
    Symbolic regression via genetic programming (gplearn).

    Attempts to discover a closed-form expression relating the library
    features to the target.  Wraps execution in a try/except with a
    timeout guard.

    Parameters
    ----------
    library : ndarray, shape (n, p)
    target : ndarray, shape (n,)
    feature_names : list of str or None
        Names for each library column.  Defaults to TERM_NAMES.
    max_time : int
        Maximum wall-clock seconds before aborting.

    Returns
    -------
    dict or None
        If successful:
            best_program : str
            r2 : float
            complexity : int
            method : 'symbolic'
        If it fails or times out, returns None.
    """
    logger.info(f"Running symbolic regression (max_time={max_time}s)")

    if feature_names is None:
        feature_names = TERM_NAMES[:library.shape[1]]

    try:
        from gplearn.genetic import SymbolicRegressor
    except ImportError:
        logger.warning(
            "gplearn is not installed; skipping symbolic regression. "
            "Install with: pip install gplearn"
        )
        return None

    # Set up timeout (SIGALRM is Unix-only)
    use_alarm = hasattr(signal, 'SIGALRM')
    old_handler = None
    if use_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(max_time)

    try:
        sr = SymbolicRegressor(
            population_size=1000,
            generations=20,
            function_set=['add', 'mul', 'sub', 'neg'],
            parsimony_coefficient=0.01,
            max_samples=0.8,
            random_state=42,
            verbose=0,
            n_jobs=1,
            feature_names=feature_names,
        )
        sr.fit(library, target)

        best_program = str(sr._program)
        prediction = sr.predict(library)
        ss_res = np.sum((target - prediction) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        complexity = sr._program.length_

        logger.info(
            f"Symbolic regression: R²={r2:.6f}, complexity={complexity}, "
            f"program={best_program}"
        )

        result = {
            'best_program': best_program,
            'r2': r2,
            'complexity': complexity,
            'method': 'symbolic',
        }

    except _TimeoutError:
        logger.warning(
            f"Symbolic regression timed out after {max_time}s"
        )
        result = None

    except Exception as exc:
        logger.warning(f"Symbolic regression failed: {exc}")
        result = None

    finally:
        # Restore signal handler
        if use_alarm:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    return result


# ---------------------------------------------------------------------------
# BASELINE 5: Elastic Net (PRD improvement #7)
# ---------------------------------------------------------------------------

def elastic_net_regression(V, S_grid, t_grid, smooth=True, seed=42):
    """
    Elastic Net regression with cross-validated alpha and l1_ratio.

    Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties.  The L2 part
    handles correlated predictors more gracefully than pure Lasso while
    L1 still enforces sparsity.  Uses ``sklearn.linear_model.ElasticNetCV``
    with a 2-D grid over (l1_ratio, alpha).

    Builds the same candidate library as the SINDy pipeline via
    ``compute_derivatives`` and ``build_candidate_library``.

    Parameters
    ----------
    V : ndarray, shape (n_S, n_t)
        Option price surface.
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    smooth : bool
        Apply Savitzky-Golay smoothing before differentiation.
    seed : int
        Random seed for the CV splitter.

    Returns
    -------
    dict
        coefficients : ndarray, shape (5,)
        active_terms : list of str
        r2_score : float
        term_names : list of str
        best_alpha : float
        best_l1_ratio : float
        n_active : int
        method : 'elastic_net'
    """
    from sklearn.linear_model import ElasticNetCV

    set_all_seeds(seed)
    logger.info(f"Running Elastic Net (smooth={smooth}, seed={seed})")

    derivs = compute_derivatives(V, S_grid, t_grid, smooth=smooth, trim=5)
    library = build_candidate_library(
        derivs['V'], derivs['dVdS'], derivs['d2VdS2'], derivs['S_mesh']
    )
    target = derivs['dVdt'].ravel()

    l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
    alphas = np.logspace(-4, 1, 50)

    try:
        model = ElasticNetCV(
            l1_ratio=l1_ratios,
            alphas=alphas,
            cv=5,
            max_iter=20000,
            random_state=seed,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(library, target)
        coeffs = model.coef_
        best_alpha = float(model.alpha_)
        best_l1 = float(model.l1_ratio_)
    except Exception as exc:
        logger.error(f"ElasticNetCV failed: {exc}")
        coeffs = np.zeros(library.shape[1])
        best_alpha = float('nan')
        best_l1 = float('nan')

    r2 = _compute_r2(library, target, coeffs)
    active_mask = np.abs(coeffs) > 1e-8
    n_active = int(np.sum(active_mask))
    active_terms = [TERM_NAMES[i] for i in range(len(TERM_NAMES))
                    if i < len(active_mask) and active_mask[i]]

    logger.info(
        f"Elastic Net: best_alpha={best_alpha:.4e}, l1_ratio={best_l1:.3f}, "
        f"R²={r2:.6f}, active={n_active}, "
        f"coeffs={np.array2string(coeffs, precision=6)}"
    )

    return {
        'coefficients': coeffs,
        'active_terms': active_terms,
        'r2_score': r2,
        'term_names': list(TERM_NAMES),
        'best_alpha': best_alpha,
        'best_l1_ratio': best_l1,
        'n_active': n_active,
        'active_mask': active_mask,
        'method': 'elastic_net',
    }


# ---------------------------------------------------------------------------
# BASELINE 6: PySR symbolic regression (PRD improvement #9)
# ---------------------------------------------------------------------------

def pysr_symbolic_regression(V, S_grid, t_grid, smooth=True,
                             n_iterations=20, populations=15, max_size=15,
                             timeout_minutes=5, seed=42):
    """
    PySR-based symbolic regression for the Black-Scholes PDE.

    Attempts to discover a closed-form expression for ``dV/dt`` in terms
    of ``S``, ``dV/dS``, and ``d2V/dS2``.  If PySR is not installed or
    the underlying Julia backend is unavailable, the function returns a
    skip dict instead of raising.

    Parameters
    ----------
    V : ndarray, shape (n_S, n_t)
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    smooth : bool
        Apply Savitzky-Golay smoothing before differentiation.
    n_iterations : int
        Number of PySR iterations.
    populations : int
        Number of populations to evolve in parallel.
    max_size : int
        Maximum complexity of discovered expressions.
    timeout_minutes : int
        Hard wall-clock timeout for the search (Unix only via SIGALRM).
    seed : int
        Random seed for PySR.

    Returns
    -------
    dict
        On success: {status: 'completed', symbolic_expression: str,
                     r2_score: float, n_terms: int, method: 'pysr'}
        On skip/failure: {status: 'skipped', reason: str, method: 'pysr'}
    """
    set_all_seeds(seed)
    logger.info(
        f"Running PySR symbolic regression "
        f"(iterations={n_iterations}, populations={populations}, "
        f"max_size={max_size}, timeout={timeout_minutes}min)"
    )

    # Try import; if missing, skip gracefully
    try:
        from pysr import PySRRegressor
    except Exception as exc:
        logger.warning(f"PySR import failed: {exc}. Skipping PySR baseline.")
        return {
            'status': 'skipped',
            'reason': f'PySR unavailable: {exc}',
            'method': 'pysr',
        }

    # Build features
    try:
        derivs = compute_derivatives(V, S_grid, t_grid, smooth=smooth, trim=5)
    except Exception as exc:
        logger.warning(f"Derivative computation failed: {exc}")
        return {
            'status': 'skipped',
            'reason': f'Derivative computation failed: {exc}',
            'method': 'pysr',
        }

    S_flat = derivs['S_mesh'].ravel()
    dVdS_flat = derivs['dVdS'].ravel()
    d2VdS2_flat = derivs['d2VdS2'].ravel()
    V_flat = derivs['V'].ravel()
    target = derivs['dVdt'].ravel()

    # Feature matrix: columns S, V, dV/dS, d2V/dS2
    X = np.column_stack([S_flat, V_flat, dVdS_flat, d2VdS2_flat])
    variable_names = ['S', 'V', 'dV_dS', 'd2V_dS2']

    # Timeout setup
    timeout_secs = int(timeout_minutes * 60)
    use_alarm = hasattr(signal, 'SIGALRM')
    old_handler = None
    if use_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_secs)

    result = None
    try:
        try:
            model = PySRRegressor(
                niterations=n_iterations,
                populations=populations,
                maxsize=max_size,
                binary_operators=['+', '-', '*', '/'],
                unary_operators=[],
                random_state=seed,
                deterministic=True,
                parallelism='serial',
                progress=False,
                verbosity=0,
            )
        except Exception as exc:
            logger.warning(
                f"PySRRegressor instantiation failed (likely Julia missing): {exc}"
            )
            return {
                'status': 'skipped',
                'reason': f'PySR instantiation failed: {exc}',
                'method': 'pysr',
            }

        try:
            model.fit(X, target, variable_names=variable_names)
        except _TimeoutError:
            logger.warning(f"PySR timed out after {timeout_minutes} min")
            return {
                'status': 'skipped',
                'reason': f'Timed out after {timeout_minutes} min',
                'method': 'pysr',
            }
        except Exception as exc:
            logger.warning(f"PySR fit failed: {exc}")
            return {
                'status': 'skipped',
                'reason': f'PySR fit failed: {exc}',
                'method': 'pysr',
            }

        # Extract best expression
        try:
            best_expr = str(model.sympy())
            prediction = np.asarray(model.predict(X)).ravel()
            ss_res = np.sum((target - prediction) ** 2)
            ss_tot = np.sum((target - np.mean(target)) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

            # Approximate term count: split by + or -
            n_terms = max(1, len([t for t in best_expr.replace('-', '+').split('+')
                                  if t.strip()]))

            logger.info(
                f"PySR: expression={best_expr}, R²={r2:.6f}, n_terms={n_terms}"
            )
            result = {
                'status': 'completed',
                'symbolic_expression': best_expr,
                'r2_score': float(r2),
                'n_terms': int(n_terms),
                'method': 'pysr',
            }
        except Exception as exc:
            logger.warning(f"PySR result extraction failed: {exc}")
            result = {
                'status': 'skipped',
                'reason': f'Result extraction failed: {exc}',
                'method': 'pysr',
            }
    finally:
        if use_alarm:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    return result


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------

def run_all_baselines(V, S_grid, t_grid, true_sigma=None, true_r=None,
                      K=100, T=1.0, trim=5):
    """
    Run all four baseline methods on a price surface.

    Computes derivatives and builds the candidate library using the same
    functions as the SINDy pipeline, then passes them to each baseline.

    Parameters
    ----------
    V : ndarray, shape (n_S, n_t)
        Option price surface.
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    true_sigma : float or None
        True volatility (for error computation).
    true_r : float or None
        True risk-free rate (for error computation).
    K : float
        Strike price.
    T : float
        Option maturity.
    trim : int
        Boundary trim width for derivative computation.

    Returns
    -------
    dict
        Keys: 'dense', 'lasso', 'ridge_threshold', 'symbolic' (may be None),
              'library_info' with condition_number and n_points.
    """
    set_all_seeds(42)

    logger.info("=" * 60)
    logger.info("Running all baseline methods")
    logger.info("=" * 60)

    # Compute derivatives and build library
    derivs = compute_derivatives(V, S_grid, t_grid, trim=trim)
    library = build_candidate_library(
        derivs['V'], derivs['dVdS'], derivs['d2VdS2'], derivs['S_mesh']
    )
    target = derivs['dVdt'].ravel()
    cond_number = np.linalg.cond(library)
    n_points = len(target)

    logger.info(f"Library shape: {library.shape}, condition number: {cond_number:.2e}")

    # Run baselines
    dense_result = dense_regression(library, target)
    lasso_result = lasso_regression(library, target)
    ridge_result = ridge_threshold(library, target)
    symbolic_result = symbolic_regression(library, target, feature_names=TERM_NAMES)

    # Compute relative errors against true BS coefficients if known
    if true_sigma is not None and true_r is not None:
        true_coeffs = np.array([
            true_r,                       # V
            0.0,                          # dV/dS
            0.0,                          # d2V/dS2
            -true_r,                      # S*dV/dS
            -0.5 * true_sigma ** 2,       # S^2*d2V/dS2
        ])

        for label, result in [('dense', dense_result),
                               ('lasso', lasso_result),
                               ('ridge_threshold', ridge_result)]:
            if result is not None:
                rel_err = safe_relative_error(result['coefficients'], true_coeffs)
                result['relative_errors'] = rel_err
                result['true_coefficients'] = true_coeffs
                logger.info(
                    f"{label} relative errors: "
                    f"{np.array2string(rel_err, precision=4)}"
                )

    results = {
        'dense': dense_result,
        'lasso': lasso_result,
        'ridge_threshold': ridge_result,
        'symbolic': symbolic_result,
        'library_info': {
            'condition_number': cond_number,
            'n_points': n_points,
        },
    }

    logger.info("All baselines complete")
    return results


def run_baselines_noisy(V, S_grid, t_grid, noise_pct=0.05,
                        true_sigma=None, true_r=None, K=100, T=1.0):
    """
    Run all baselines on a noise-corrupted price surface.

    Adds Gaussian noise to the price surface via
    ``data_generation.add_noise`` before computing derivatives and
    running each baseline method.

    Parameters
    ----------
    V : ndarray, shape (n_S, n_t)
        Clean option price surface.
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    noise_pct : float
        Noise level as fraction of V's standard deviation.
    true_sigma : float or None
    true_r : float or None
    K : float
    T : float

    Returns
    -------
    dict
        Same structure as ``run_all_baselines``.
    """
    logger.info(f"Adding {noise_pct:.1%} noise to price surface")
    V_noisy = add_noise(V, noise_pct, seed=42)

    return run_all_baselines(
        V_noisy, S_grid, t_grid,
        true_sigma=true_sigma, true_r=true_r,
        K=K, T=T, trim=5,
    )
