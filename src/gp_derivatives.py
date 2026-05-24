"""
Gaussian Process-based derivative estimation for noise-robust SINDy PDE discovery.

Fits a GP (RBF + WhiteKernel) to a noisy price surface using a random subsample
of the (S, t) grid. Derivatives dV/dt, dV/dS, d2V/dS2 are computed analytically
from the GP posterior using closed-form RBF kernel derivative formulas.

This avoids the noise-amplification problems of finite differences and the
training instability / approximation bias of small neural networks.
"""

import time
import warnings

import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern

from src.utils import set_all_seeds, setup_logging, safe_relative_error
from src.data_generation import generate_price_surface, add_noise
from src.sindy_discovery import (
    build_candidate_library,
    stlsq_sweep,
    format_pde_string,
    compute_r2_clean,
    compute_coefficient_metrics,
    TERM_NAMES,
)

logger = setup_logging(__name__)


def fit_gp_surface(V_noisy, S_grid, t_grid, n_subsample=500, seed=42,
                   kernel='rbf', return_info=False):
    """
    Fit a GaussianProcessRegressor (RBF + WhiteKernel) to a noisy price surface.

    Random-subsamples ``n_subsample`` points from the full (S, t) grid to
    keep the O(N^3) GP fit tractable.  When the full grid is small enough that
    the requested ``n_subsample`` would cover most of it, the subsample is
    automatically reduced to ``min(n_subsample, int(total_points * 0.7))``.

    Parameters
    ----------
    V_noisy : ndarray, shape (n_S, n_t)
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    n_subsample : int
        Requested number of grid points to subsample for the GP fit.  May be
        auto-reduced on sparse surfaces.
    seed : int
    kernel : {'rbf', 'matern'}, default 'rbf'
        Choice of covariance.  ``'matern'`` uses Matern(nu=2.5), which is
        twice-differentiable (still admits the analytical derivatives we
        compute) but produces less smooth realisations than RBF and is
        therefore preferred for noisy real-data surfaces.
    return_info : bool, default False
        Backwards-compatible flag.  When False, the function returns the
        legacy ``(gp, subsample_idx)`` tuple.  When True, returns a dict with
        ``gp``, ``subsample_idx``, ``length_scales``, ``noise_level``,
        ``constant_value``, ``kernel_used``.

    Returns
    -------
    gp, subsample_idx : when ``return_info=False`` (default)
    info : dict, when ``return_info=True``
    """
    set_all_seeds(seed)

    n_S, n_t = V_noisy.shape
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')

    X_full = np.column_stack([S_mesh.ravel(), t_mesh.ravel()])
    y_full = V_noisy.ravel()

    n_total = X_full.shape[0]

    # Auto-reduce subsample when the grid is too small to leave headroom.
    total_points = V_noisy.size
    auto_subsample = min(n_subsample, int(total_points * 0.7))
    if auto_subsample < n_subsample:
        logger.info(
            f"Auto-reduced subsample from {n_subsample} to {auto_subsample} "
            f"(total grid points = {total_points})"
        )
    n_subsample = max(auto_subsample, 1)

    rng = np.random.RandomState(seed)
    n_use = min(n_subsample, n_total)
    subsample_idx = rng.choice(n_total, size=n_use, replace=False)

    X_train = X_full[subsample_idx]
    y_train = y_full[subsample_idx]

    # Reasonable length-scale initial guess: a small multiple of grid extent
    S_extent = float(S_grid[-1] - S_grid[0])
    t_extent = float(t_grid[-1] - t_grid[0])
    length_scale_init = [0.2 * S_extent, 0.2 * t_extent]

    # Estimate noise level from y variability (lower bound for the white kernel)
    y_var = float(np.var(y_train))
    noise_init = max(1e-4, 0.01 * y_var)

    kernel_used = str(kernel).lower()
    if kernel_used == 'matern':
        # Matern with nu=2.5 is twice-differentiable analytically -- still
        # admits dV/dS and d2V/dS2 in closed form but with shorter effective
        # correlation length than RBF, which is what we want on rough real
        # market surfaces.
        k_obj = (
            ConstantKernel(constant_value=max(y_var, 1e-3),
                           constant_value_bounds=(1e-5, 1e8))
            * Matern(length_scale=length_scale_init,
                     nu=2.5,
                     length_scale_bounds=(1e-2, 1e4))
            + WhiteKernel(noise_level=noise_init,
                          noise_level_bounds=(1e-10, 1e2))
        )
    else:
        kernel_used = 'rbf'
        k_obj = (
            ConstantKernel(constant_value=max(y_var, 1e-3),
                           constant_value_bounds=(1e-5, 1e8))
            * RBF(length_scale=length_scale_init,
                  length_scale_bounds=(1e-2, 1e4))
            + WhiteKernel(noise_level=noise_init,
                          noise_level_bounds=(1e-10, 1e2))
        )

    gp = GaussianProcessRegressor(
        kernel=k_obj,
        n_restarts_optimizer=2,
        normalize_y=True,
        random_state=seed,
        alpha=0.0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_train, y_train)

    logger.info(
        f"GP fit complete ({kernel_used}): n_train={n_use}, "
        f"learned kernel={gp.kernel_}, "
        f"log-marginal-likelihood={gp.log_marginal_likelihood_value_:.2f}"
    )

    # Pull learned hyperparameters for return-info / kernel comparison.
    try:
        product = gp.kernel_.k1
        white = gp.kernel_.k2
        constant = product.k1
        inner = product.k2
        ls = np.atleast_1d(np.asarray(inner.length_scale, dtype=float))
        if ls.size == 1:
            ls = np.repeat(ls, 2)
        learned = {
            'length_scales': ls,
            'noise_level': float(white.noise_level),
            'constant_value': float(constant.constant_value),
            'kernel_used': kernel_used,
        }
    except Exception:
        learned = {
            'length_scales': np.array([np.nan, np.nan]),
            'noise_level': float('nan'),
            'constant_value': float('nan'),
            'kernel_used': kernel_used,
        }

    if return_info:
        return {
            'gp': gp,
            'subsample_idx': subsample_idx,
            **learned,
        }
    return gp, subsample_idx


def _unpack_rbf_kernel(gp):
    """
    Extract (constant^2, length_scales, noise_level) from a fitted GP.

    Accepts the kernel structure built in ``fit_gp_surface``:
        ConstantKernel * RBF + WhiteKernel
    Returns
    -------
    sigma_f2 : float (constant value, i.e. signal variance prefactor)
    length_scales : ndarray, shape (2,)
    noise_level : float
    y_train_mean : float
    y_train_std : float
    """
    kernel = gp.kernel_

    # The fitted kernel is a Sum(Product(ConstantKernel, RBF), WhiteKernel)
    sum_part = kernel
    product = sum_part.k1
    white = sum_part.k2

    constant = product.k1
    rbf = product.k2

    sigma_f2 = float(constant.constant_value)
    length_scales = np.atleast_1d(np.asarray(rbf.length_scale, dtype=float))
    if length_scales.size == 1:
        length_scales = np.repeat(length_scales, 2)

    noise_level = float(white.noise_level)

    # normalize_y stores mean/std
    y_train_mean = float(getattr(gp, '_y_train_mean', 0.0))
    y_train_std = float(getattr(gp, '_y_train_std', 1.0))

    return sigma_f2, length_scales, noise_level, y_train_mean, y_train_std


def _is_rbf_inner(gp):
    """True if the fitted GP's inner kernel is an RBF (vs Matern, etc.)."""
    try:
        inner = gp.kernel_.k1.k2
        return isinstance(inner, RBF)
    except Exception:
        return False


def _compute_gp_derivatives_numerical(gp, S_grid, t_grid):
    """Fallback: predict GP on full grid, take centered finite differences.

    Used for non-RBF kernels (e.g. Matern) where the closed-form RBF
    derivative formulas in :func:`compute_gp_derivatives` do not apply.
    """
    n_S = len(S_grid)
    n_t = len(t_grid)
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')
    X_pred = np.column_stack([S_mesh.ravel(), t_mesh.ravel()])
    V_smooth = gp.predict(X_pred).reshape(n_S, n_t)

    # Centered finite differences (interior); one-sided at edges.
    dS = float(S_grid[1] - S_grid[0])
    dt = float(t_grid[1] - t_grid[0])

    dV_dS = np.gradient(V_smooth, dS, axis=0)
    dV_dt = np.gradient(V_smooth, dt, axis=1)

    d2V_dS2 = np.zeros_like(V_smooth)
    d2V_dS2[1:-1, :] = (V_smooth[2:, :] - 2.0 * V_smooth[1:-1, :]
                        + V_smooth[:-2, :]) / (dS ** 2)
    d2V_dS2[0, :] = d2V_dS2[1, :]
    d2V_dS2[-1, :] = d2V_dS2[-2, :]

    return {
        'V_smooth': V_smooth,
        'dV_dt': dV_dt,
        'dV_dS': dV_dS,
        'd2V_dS2': d2V_dS2,
    }


def compute_gp_derivatives(gp, S_grid, t_grid):
    """
    Compute analytical derivatives of the GP posterior mean on the full grid.

    For RBF kernel k(x, x') = sigma_f^2 * exp(-0.5 * sum_d (x_d - x'_d)^2 / L_d^2):
        dk/dx_d = -k * (x_d - x'_d) / L_d^2
        d2k/dx_d^2 = k * ((x_d - x'_d)^2 / L_d^4 - 1 / L_d^2)

    The posterior mean is mu(x*) = k(x*, X_train) @ alpha. Its derivatives
    are obtained by differentiating k under the sum (alpha is independent of x*).

    For non-RBF kernels (e.g. Matern), falls back to numerical derivatives
    of ``gp.predict`` on the dense grid.

    Vectorised: no Python loop over prediction points.

    Returns
    -------
    dict with keys:
        'V_smooth', 'dV_dt', 'dV_dS', 'd2V_dS2' all shape (n_S, n_t)
    """
    if not _is_rbf_inner(gp):
        return _compute_gp_derivatives_numerical(gp, S_grid, t_grid)

    sigma_f2, length_scales, _, y_mean, y_std = _unpack_rbf_kernel(gp)
    L_S, L_t = float(length_scales[0]), float(length_scales[1])

    alpha = np.asarray(gp.alpha_, dtype=float).ravel()  # shape (n_train,)
    X_train = np.asarray(gp.X_train_, dtype=float)       # shape (n_train, 2)

    n_S = len(S_grid)
    n_t = len(t_grid)
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')
    X_pred = np.column_stack([S_mesh.ravel(), t_mesh.ravel()])  # (n_pred, 2)

    # Pairwise differences (n_pred, n_train) in each dim
    dS = X_pred[:, 0:1] - X_train[:, 0:1].T  # (n_pred, n_train)
    dt = X_pred[:, 1:2] - X_train[:, 1:2].T

    # RBF kernel matrix between pred and train
    sq = (dS / L_S) ** 2 + (dt / L_t) ** 2
    K = sigma_f2 * np.exp(-0.5 * sq)

    # Derivative kernels
    dK_dS = -K * (dS / (L_S ** 2))
    dK_dt = -K * (dt / (L_t ** 2))
    d2K_dS2 = K * ((dS ** 2) / (L_S ** 4) - 1.0 / (L_S ** 2))

    # Apply alpha: predictions in normalized space, then rescale by y_std.
    # The y-mean is constant so its derivative is zero.
    V_norm = K @ alpha
    dV_dS = (dK_dS @ alpha) * y_std
    dV_dt = (dK_dt @ alpha) * y_std
    d2V_dS2 = (d2K_dS2 @ alpha) * y_std

    V_smooth = y_mean + y_std * V_norm

    return {
        'V_smooth': V_smooth.reshape(n_S, n_t),
        'dV_dt': dV_dt.reshape(n_S, n_t),
        'dV_dS': dV_dS.reshape(n_S, n_t),
        'd2V_dS2': d2V_dS2.reshape(n_S, n_t),
    }


def sindy_with_gp_derivatives(V_noisy, S_grid, t_grid, threshold=0.1,
                                n_subsample=500, seed=42, trim=5,
                                K=100, r=0.05, sigma=0.2, T=1.0,
                                option_type='call', true_sigma=None,
                                true_r=None):
    """
    Run SINDy PDE discovery using GP-derived derivatives.

    Returns the same dict structure as ``discover_pde`` plus ``r2_clean``.
    """
    set_all_seeds(seed)

    # Sub-sample timeout guard: if first attempt is slow, retry with 300
    t_start = time.perf_counter()
    gp, _ = fit_gp_surface(V_noisy, S_grid, t_grid,
                           n_subsample=n_subsample, seed=seed)
    elapsed = time.perf_counter() - t_start
    if elapsed > 60.0 and n_subsample > 300:
        logger.warning(
            f"GP fit took {elapsed:.1f}s (>60s). Refitting with n_subsample=300."
        )
        gp, _ = fit_gp_surface(V_noisy, S_grid, t_grid,
                               n_subsample=300, seed=seed)

    derivs = compute_gp_derivatives(gp, S_grid, t_grid)

    # Trim to remove any edge weirdness (consistent with FD/neural pipelines)
    s = slice(trim, -trim) if trim > 0 else slice(None)
    V_tr = derivs['V_smooth'][s, s]
    dVdt_tr = derivs['dV_dt'][s, s]
    dVdS_tr = derivs['dV_dS'][s, s]
    d2VdS2_tr = derivs['d2V_dS2'][s, s]

    S_tr = S_grid[s]
    t_tr = t_grid[s]
    S_mesh_tr, _ = np.meshgrid(S_tr, t_tr, indexing='ij')

    library = build_candidate_library(V_tr, dVdS_tr, d2VdS2_tr, S_mesh_tr)
    target = dVdt_tr.ravel()
    cond_number = float(np.linalg.cond(library))

    # Use the requested threshold as the minimum of a small sweep
    thresholds = np.sort(np.unique(np.concatenate([
        np.logspace(-3, np.log10(2.0), 30),
        np.linspace(0.001, 0.1, 20),
        np.array([threshold]),
    ])))
    best, sweep_results = stlsq_sweep(library, target, thresholds=thresholds)
    discovered = best['coefficients']

    # Resolve true coefficients (default to BS pipeline params)
    if true_sigma is None:
        true_sigma = sigma
    if true_r is None:
        true_r = r

    true_coeffs = np.array([
        true_r, 0.0, 0.0, -true_r, -0.5 * true_sigma ** 2,
    ])
    rel_errors = safe_relative_error(discovered, true_coeffs)

    # R²(clean): how well the discovered PDE predicts analytical dV/dt
    try:
        r2_clean = compute_r2_clean(
            discovered, S_grid, t_grid,
            K=K, r=true_r, sigma=true_sigma, T=T,
            option_type=option_type, trim=trim,
        )
    except Exception as e:
        logger.warning(f"compute_r2_clean failed: {e}")
        r2_clean = float('nan')

    active_terms = [TERM_NAMES[i] for i in range(5) if best['active_mask'][i]]
    pde_str = format_pde_string(discovered, TERM_NAMES)

    logger.info(
        f"GP-SINDy: R²(noisy)={best['r2']:.6f}, R²(clean)={r2_clean:.6f}, "
        f"active={best['n_active']}, PDE: {pde_str}"
    )

    return {
        'discovered_coefficients': discovered,
        'true_coefficients': true_coeffs,
        'active_terms': active_terms,
        'term_names': TERM_NAMES,
        'relative_errors': rel_errors,
        'best_threshold': best['threshold'],
        'r2_score': best['r2'],
        'r2_clean': r2_clean,
        'bic': best['bic'],
        'condition_number': cond_number,
        'sweep_results': sweep_results,
        'human_readable_pde': pde_str,
        'active_mask': best['active_mask'],
        'n_active': best['n_active'],
        'gp_derivatives': derivs,
    }


def run_gp_noise_robustness(noise_levels=None, n_S=50, n_t=50, K=100,
                              r=0.05, sigma=0.2, T=1.0, seed=42,
                              n_subsample=500):
    """
    Run GP-based SINDy across a list of noise levels and return a DataFrame.

    Columns: noise_pct, r2_clean, r2_noisy, sigma_recovered,
             max_coeff_rel_err, runtime_s.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    V_clean, S_grid, t_grid = generate_price_surface(
        n_S=n_S, n_t=n_t, K=K, r=r, sigma=sigma, T=T,
    )

    rows = []
    for noise_pct in noise_levels:
        V_noisy = add_noise(V_clean, noise_pct, seed=seed) if noise_pct > 0 else V_clean
        t_start = time.perf_counter()
        try:
            result = sindy_with_gp_derivatives(
                V_noisy, S_grid, t_grid,
                n_subsample=n_subsample, seed=seed,
                K=K, r=r, sigma=sigma, T=T,
                true_r=r, true_sigma=sigma,
            )
            runtime_s = time.perf_counter() - t_start
            cm = compute_coefficient_metrics(
                result['discovered_coefficients'],
                true_r=r, true_sigma=sigma,
            )
            # Recovered sigma from -0.5*sigma^2 coefficient on S^2*d2V/dS2
            c4 = float(result['discovered_coefficients'][4])
            sigma_rec = float(np.sqrt(-2.0 * c4)) if c4 < 0 else float('nan')

            rows.append({
                'noise_pct': float(noise_pct),
                'r2_clean': float(result['r2_clean']),
                'r2_noisy': float(result['r2_score']),
                'sigma_recovered': sigma_rec,
                'max_coeff_rel_err': float(cm['max_coeff_rel_error']),
                'runtime_s': float(runtime_s),
            })
        except Exception as e:
            runtime_s = time.perf_counter() - t_start
            logger.warning(
                f"GP-SINDy failed at noise={noise_pct}: {e}. Recording NaN."
            )
            rows.append({
                'noise_pct': float(noise_pct),
                'r2_clean': float('nan'),
                'r2_noisy': float('nan'),
                'sigma_recovered': float('nan'),
                'max_coeff_rel_err': float('nan'),
                'runtime_s': float(runtime_s),
            })

    df = pd.DataFrame(rows)
    logger.info(f"GP noise robustness sweep:\n{df.to_string(index=False)}")
    return df


# ---------------------------------------------------------------------------
# Fix #3 -- Constrained-length-scale GP for derivative-preserving fits.
# ---------------------------------------------------------------------------

def fit_gp_surface_constrained(V, S_grid, t_grid, n_subsample=500, seed=42,
                                ls_bounds_S=None, ls_bounds_t=None,
                                ls_init_S=None, ls_init_t=None,
                                stratified=False, kernel='rbf',
                                n_restarts_optimizer=2):
    """
    Fit a GP with constrained per-axis length scales.

    Designed for derivative-preservation use cases such as GP-Dupire where
    over-large length scales on the K (strike) axis collapse the second
    derivative.  Lets callers pin or bound the length scale on either axis
    and optionally use stratified sampling so every level of the S/K axis
    is represented in the training set.

    Parameters
    ----------
    V : ndarray, shape (n_S, n_t)
    S_grid, t_grid : 1-D ndarrays
    n_subsample : int
    seed : int
    ls_bounds_S, ls_bounds_t : tuple (low, high) or None
        Bounds on length scale along each axis.  If None, falls back to the
        default ``(1e-2, 1e4)``.
    ls_init_S, ls_init_t : float or None
        Initial length-scale guesses.  If None, defaults to 20% of the
        respective grid extent.
    stratified : bool, default False
        If True, perform stratified sampling: take roughly
        ``n_subsample / n_S`` points from each S level (with at least one
        sample per S level so curvature along S is well constrained).
    kernel : {'rbf', 'matern'}, default 'rbf'
    n_restarts_optimizer : int, default 2

    Returns
    -------
    gp, subsample_idx
    """
    set_all_seeds(seed)

    n_S, n_t = V.shape
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')

    X_full = np.column_stack([S_mesh.ravel(), t_mesh.ravel()])
    y_full = V.ravel()
    n_total = X_full.shape[0]

    total_points = V.size
    auto_subsample = min(n_subsample, int(total_points * 0.7))
    n_subsample = max(auto_subsample, 1)

    rng = np.random.RandomState(seed)

    if stratified:
        per_level = max(int(np.ceil(n_subsample / n_S)), 1)
        idx_list = []
        for i in range(n_S):
            base = i * n_t
            n_take = min(per_level, n_t)
            cols = rng.choice(n_t, size=n_take, replace=False)
            idx_list.append(base + cols)
        subsample_idx = np.concatenate(idx_list)
        if subsample_idx.size > n_subsample:
            keep = rng.choice(subsample_idx.size, size=n_subsample,
                              replace=False)
            subsample_idx = subsample_idx[keep]
    else:
        n_use = min(n_subsample, n_total)
        subsample_idx = rng.choice(n_total, size=n_use, replace=False)

    X_train = X_full[subsample_idx]
    y_train = y_full[subsample_idx]

    S_extent = float(S_grid[-1] - S_grid[0])
    t_extent = float(t_grid[-1] - t_grid[0])

    ls_S = float(ls_init_S) if ls_init_S is not None else 0.2 * S_extent
    ls_t = float(ls_init_t) if ls_init_t is not None else 0.2 * t_extent
    bounds_S = tuple(ls_bounds_S) if ls_bounds_S is not None else (1e-2, 1e4)
    bounds_t = tuple(ls_bounds_t) if ls_bounds_t is not None else (1e-2, 1e4)

    # Clip the init into its own bounds so sklearn doesn't reject it.
    ls_S = float(np.clip(ls_S, bounds_S[0] * 1.001, bounds_S[1] * 0.999))
    ls_t = float(np.clip(ls_t, bounds_t[0] * 1.001, bounds_t[1] * 0.999))

    y_var = float(np.var(y_train))
    noise_init = max(1e-4, 0.01 * y_var)

    if str(kernel).lower() == 'matern':
        inner = Matern(length_scale=[ls_S, ls_t], nu=2.5,
                       length_scale_bounds=[bounds_S, bounds_t])
    else:
        inner = RBF(length_scale=[ls_S, ls_t],
                    length_scale_bounds=[bounds_S, bounds_t])

    k_obj = (
        ConstantKernel(constant_value=max(y_var, 1e-3),
                       constant_value_bounds=(1e-5, 1e8))
        * inner
        + WhiteKernel(noise_level=noise_init,
                      noise_level_bounds=(1e-10, 1e2))
    )

    gp = GaussianProcessRegressor(
        kernel=k_obj,
        n_restarts_optimizer=int(n_restarts_optimizer),
        normalize_y=True,
        random_state=seed,
        alpha=0.0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_train, y_train)

    logger.info(
        f"Constrained GP fit: n_train={len(subsample_idx)}, "
        f"bounds_S={bounds_S}, bounds_t={bounds_t}, stratified={stratified}, "
        f"learned kernel={gp.kernel_}"
    )

    return gp, subsample_idx
