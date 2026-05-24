"""
Publication-readiness improvements: GP-based derivative estimation applied
to real option-market surfaces.

Three additions on top of the existing pipeline:

    Improvement #2 -- GP-based SINDy on real option surfaces.
    Improvement #3 -- GP-analytical derivatives feeding the Dupire library.
    Improvement #5 -- Sliding-window local volatility extraction via
                       per-window GP + Dupire SINDy.

Every per-ticker analysis is wrapped in try/except so a single bad surface
does not abort the rest of the run.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.utils import set_all_seeds, setup_logging
from src.sindy_discovery import (
    discover_pde,
    build_candidate_library,
    stlsq_sweep,
    stlsq,
    format_pde_string,
    TERM_NAMES,
)
from src.gp_derivatives import (
    fit_gp_surface,
    compute_gp_derivatives,
    fit_gp_surface_constrained,
)
from src.dupire_discovery import (
    build_dupire_library,
    discover_dupire,
    DUPIRE_TERM_NAMES,
    _extract_call_surface,
)

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bs_to_st_axes(C, K_grid, tau_grid):
    """Convert a (K, tau) call surface to (S, t) axes expected by discover_pde.

    The BS-side SINDy pipeline differentiates w.r.t. ascending calendar time
    ``t`` with the spatial coordinate being the strike-equivalent ``S``.
    ``t = T_max - tau`` and the tau-axis is reversed accordingly.

    Returns
    -------
    V_st, S_grid, t_grid, T_max
    """
    T_max = float(tau_grid.max())
    t_grid = T_max - tau_grid[::-1]
    V_st = C[:, ::-1]
    return V_st, np.asarray(K_grid, dtype=float), t_grid, T_max


def _sigma_from_bs_coeff(coeffs):
    """Recover sigma from the discovered S^2*d2V/dS2 BS coefficient."""
    c4 = float(coeffs[TERM_NAMES.index('S2*d2V/dS2')])
    if c4 < 0:
        return float(np.sqrt(-2.0 * c4))
    return float('nan')


def _sigma_from_dupire_coeff(coeffs):
    """Recover sigma from the Dupire K^2 d2C/dK2 coefficient."""
    c4 = float(coeffs[DUPIRE_TERM_NAMES.index('K2*d2C/dK2')])
    if c4 > 0:
        return float(np.sqrt(2.0 * c4))
    return float('nan')


def _adaptive_trim(shape):
    min_dim = min(shape)
    if min_dim > 25:
        return 10
    if min_dim > 15:
        return 5
    return max(min_dim // 4, 1)


# ---------------------------------------------------------------------------
# Improvement #2 -- GP-based SINDy on real option surfaces
# ---------------------------------------------------------------------------

def _gp_sindy_on_surface(V_st, S_grid, t_grid, K, T_max, r, sigma_hint,
                         seed=42, standardize=True, n_subsample=500):
    """Fit a GP, build the BS library from GP derivatives, run STLSQ.

    Returns a dict with keys mirroring ``discover_pde``.
    """
    set_all_seeds(seed)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gp, _ = fit_gp_surface(
            V_st, S_grid, t_grid, n_subsample=n_subsample, seed=seed,
        )

    derivs = compute_gp_derivatives(gp, S_grid, t_grid)

    trim = _adaptive_trim(V_st.shape)
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

    if standardize:
        col_std = library.std(axis=0)
        col_std_safe = np.where(col_std < 1e-15, 1.0, col_std)
        library_for_fit = library / col_std_safe
        best, sweep_results = stlsq_sweep(library_for_fit, target)
        discovered = best['coefficients'] / col_std_safe
        best['coefficients'] = discovered
    else:
        best, sweep_results = stlsq_sweep(library, target)
        discovered = best['coefficients']

    active_terms = [TERM_NAMES[i] for i in range(len(TERM_NAMES))
                    if best['active_mask'][i]]
    pde_str = format_pde_string(discovered, TERM_NAMES)

    return {
        'discovered_coefficients': np.asarray(discovered, dtype=float),
        'active_terms': active_terms,
        'active_mask': np.asarray(best['active_mask'], dtype=bool),
        'n_active': int(best['n_active']),
        'r2_score': float(best['r2']),
        'best_threshold': float(best['threshold']),
        'bic': float(best['bic']),
        'condition_number': cond_number,
        'human_readable_pde': pde_str,
        'term_names': list(TERM_NAMES),
        'sweep_results': sweep_results,
    }


def run_gp_sindy_on_real_data(per_ticker_results, standardize=True):
    """Apply GP-based SINDy to each ticker's cached real option surface.

    Parameters
    ----------
    per_ticker_results : dict
        Output of :func:`src.real_data.run_real_data_experiment`.
    standardize : bool, default True
        Standardize the library before STLSQ (recommended for the real-data
        regime where strike levels are ~$500).

    Returns
    -------
    dict
        ``{ticker: {gp_r2, gp_coefficients, gp_active_terms,
        sigma_discovered, ...}}``.  Tickers that fail map to
        ``{'error': str, 'message': str}``.
    """
    out = {}
    for ticker, entry in per_ticker_results.items():
        try:
            surface = entry.get('surface_data')
            option_data = entry.get('option_data', {})
            if surface is None:
                raise KeyError("missing surface_data")

            C = np.asarray(surface['V_surface'], dtype=float)
            K_grid = np.asarray(surface['K_grid'], dtype=float)
            tau_grid = np.asarray(surface['tau_grid'], dtype=float)
            r = float(surface.get('r', option_data.get('r', 0.045)))

            V_st, S_grid, t_grid, T_max = _bs_to_st_axes(C, K_grid, tau_grid)
            sigma_hint = float(entry.get('avg_implied_vol', 0.2))

            res = _gp_sindy_on_surface(
                V_st, S_grid, t_grid,
                K=float(np.median(K_grid)),
                T_max=T_max,
                r=r,
                sigma_hint=sigma_hint,
                standardize=standardize,
            )

            out[ticker] = {
                'ticker': ticker,
                'gp_r2': res['r2_score'],
                'gp_coefficients': res['discovered_coefficients'],
                'gp_active_terms': res['active_terms'],
                'gp_active_mask': res['active_mask'],
                'gp_n_active': res['n_active'],
                'gp_pde': res['human_readable_pde'],
                'gp_condition_number': res['condition_number'],
                'sigma_discovered': _sigma_from_bs_coeff(
                    res['discovered_coefficients']
                ),
                'avg_implied_vol': sigma_hint,
                'term_names': res['term_names'],
            }
            logger.info(
                "%s GP-SINDy: R2=%.4f sigma=%.4f active=%s",
                ticker, res['r2_score'],
                out[ticker]['sigma_discovered'],
                res['active_terms'],
            )
        except Exception as exc:
            logger.error("GP-SINDy failed for %s: %s", ticker, exc,
                         exc_info=False)
            out[ticker] = {
                'ticker': ticker,
                'error': type(exc).__name__,
                'message': str(exc),
            }
    return out


def _savgol_smooth_surface(V, window=11, poly=3):
    """Apply 2-D Savitzky-Golay smoothing, clamping windows that are too long."""
    V_out = V.astype(np.float64, copy=True)
    for axis in (0, 1):
        n = V_out.shape[axis]
        win = window
        if win > n:
            win = n if n % 2 == 1 else n - 1
        if win < poly + 2:
            continue
        try:
            V_out = savgol_filter(V_out, window_length=win,
                                  polyorder=poly, axis=axis)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("SavGol smoothing failed on axis %d: %s", axis, exc)
    return V_out


def compare_derivative_methods_on_real_data(per_ticker_results,
                                            standardize=True):
    """Compare FD, SavGol, and GP-SINDy on each ticker's surface.

    Returns
    -------
    pandas.DataFrame
        One row per ticker with columns ``r2_fd``, ``r2_savgol``, ``r2_gp``,
        ``sigma_fd``, ``sigma_savgol``, ``sigma_gp`` and the discovered
        coefficient vectors as object columns.
    """
    rows = []
    for ticker, entry in per_ticker_results.items():
        row = {'ticker': ticker}

        try:
            surface = entry['surface_data']
            option_data = entry.get('option_data', {})
            C = np.asarray(surface['V_surface'], dtype=float)
            K_grid = np.asarray(surface['K_grid'], dtype=float)
            tau_grid = np.asarray(surface['tau_grid'], dtype=float)
            r = float(surface.get('r', option_data.get('r', 0.045)))
            sigma_hint = float(entry.get('avg_implied_vol', 0.2))

            V_st, S_grid, t_grid, T_max = _bs_to_st_axes(C, K_grid, tau_grid)
            trim = _adaptive_trim(V_st.shape)
            K_med = float(np.median(K_grid))

            common_kw = dict(
                true_sigma=sigma_hint,
                true_r=r,
                K=K_med,
                T=T_max,
                option_type='call',
                trim=trim,
            )

            # FD (no smoothing)
            try:
                fd_res = discover_pde(
                    V_st, S_grid, t_grid,
                    smooth=False, standardize=standardize, **common_kw,
                )
                row['r2_fd'] = float(fd_res['r2_score'])
                row['coeffs_fd'] = np.asarray(
                    fd_res['discovered_coefficients'], dtype=float
                )
                row['sigma_fd'] = _sigma_from_bs_coeff(row['coeffs_fd'])
            except Exception as exc:
                logger.warning("FD failed for %s: %s", ticker, exc)
                row['r2_fd'] = float('nan')
                row['coeffs_fd'] = None
                row['sigma_fd'] = float('nan')

            # SavGol-smoothed surface, then FD pipeline
            try:
                V_smooth = _savgol_smooth_surface(V_st, window=11, poly=3)
                sg_res = discover_pde(
                    V_smooth, S_grid, t_grid,
                    smooth=False, standardize=standardize, **common_kw,
                )
                row['r2_savgol'] = float(sg_res['r2_score'])
                row['coeffs_savgol'] = np.asarray(
                    sg_res['discovered_coefficients'], dtype=float
                )
                row['sigma_savgol'] = _sigma_from_bs_coeff(row['coeffs_savgol'])
            except Exception as exc:
                logger.warning("SavGol failed for %s: %s", ticker, exc)
                row['r2_savgol'] = float('nan')
                row['coeffs_savgol'] = None
                row['sigma_savgol'] = float('nan')

            # GP
            try:
                gp_res = _gp_sindy_on_surface(
                    V_st, S_grid, t_grid,
                    K=K_med, T_max=T_max, r=r,
                    sigma_hint=sigma_hint,
                    standardize=standardize,
                )
                row['r2_gp'] = float(gp_res['r2_score'])
                row['coeffs_gp'] = np.asarray(
                    gp_res['discovered_coefficients'], dtype=float
                )
                row['sigma_gp'] = _sigma_from_bs_coeff(row['coeffs_gp'])
            except Exception as exc:
                logger.warning("GP failed for %s: %s", ticker, exc)
                row['r2_gp'] = float('nan')
                row['coeffs_gp'] = None
                row['sigma_gp'] = float('nan')

        except Exception as exc:
            logger.error("compare_derivative_methods failed for %s: %s",
                         ticker, exc)
            for key in ('r2_fd', 'r2_savgol', 'r2_gp',
                        'sigma_fd', 'sigma_savgol', 'sigma_gp'):
                row.setdefault(key, float('nan'))

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Improvement #3 -- GP-based Dupire on real data
# ---------------------------------------------------------------------------

def _gp_dupire_on_surface(C, K_grid, tau_grid, seed=42, standardize=True,
                          n_subsample=500):
    """Fit GP on C(K, tau), build Dupire library from GP derivatives, run STLSQ."""
    set_all_seeds(seed)

    # GP infrastructure expects ``S_grid, t_grid`` but it's just two
    # spatial coordinates -- we feed K as ``S_grid`` and tau as ``t_grid``.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gp, _ = fit_gp_surface(
            C, K_grid, tau_grid, n_subsample=n_subsample, seed=seed,
        )

    derivs = compute_gp_derivatives(gp, K_grid, tau_grid)
    # Map GP derivative naming -> Dupire naming
    # dV_dt (axis 1) -> dC/dtau ; dV_dS (axis 0) -> dC/dK ; d2V_dS2 -> d2C/dK2

    trim = _adaptive_trim(C.shape)
    s = slice(trim, -trim) if trim > 0 else slice(None)
    C_tr = derivs['V_smooth'][s, s]
    dCdtau_tr = derivs['dV_dt'][s, s]
    dCdK_tr = derivs['dV_dS'][s, s]
    d2CdK2_tr = derivs['d2V_dS2'][s, s]

    K_tr = K_grid[s]
    K_mesh = np.broadcast_to(K_tr[:, None], C_tr.shape)

    library = np.column_stack([
        C_tr.ravel(),
        dCdK_tr.ravel(),
        d2CdK2_tr.ravel(),
        (K_mesh * dCdK_tr).ravel(),
        (K_mesh ** 2 * d2CdK2_tr).ravel(),
    ]).astype(np.float64)
    target = dCdtau_tr.ravel()
    cond_number = float(np.linalg.cond(library))

    if standardize:
        col_std = library.std(axis=0)
        col_std_safe = np.where(col_std < 1e-15, 1.0, col_std)
        library_for_fit = library / col_std_safe
        best, sweep_results = stlsq_sweep(library_for_fit, target)
        discovered = best['coefficients'] / col_std_safe
        best['coefficients'] = discovered
    else:
        best, sweep_results = stlsq_sweep(library, target)
        discovered = best['coefficients']

    coeff_K2 = float(discovered[DUPIRE_TERM_NAMES.index('K2*d2C/dK2')])
    coeff_K = float(discovered[DUPIRE_TERM_NAMES.index('K*dC/dK')])
    sigma_discovered = float(np.sqrt(2.0 * coeff_K2)) if coeff_K2 > 0 else float('nan')
    drift_discovered = float(-coeff_K)

    active_terms = [DUPIRE_TERM_NAMES[i]
                    for i in range(len(DUPIRE_TERM_NAMES))
                    if best['active_mask'][i]]
    pde_str = format_pde_string(
        discovered, term_names=DUPIRE_TERM_NAMES,
    ).replace('dV/dt', 'dC/dT')

    return {
        'discovered_coefficients': np.asarray(discovered, dtype=float),
        'active_terms': active_terms,
        'active_mask': np.asarray(best['active_mask'], dtype=bool),
        'n_active': int(best['n_active']),
        'r2_score': float(best['r2']),
        'sigma_discovered': sigma_discovered,
        'drift_discovered': drift_discovered,
        'best_threshold': float(best['threshold']),
        'bic': float(best['bic']),
        'condition_number': cond_number,
        'human_readable_pde': pde_str,
        'term_names': list(DUPIRE_TERM_NAMES),
        'sweep_results': sweep_results,
    }


def run_gp_dupire_on_real_data(per_ticker_results, standardize=True):
    """Run GP-derived Dupire SINDy on each ticker's C(K, tau) surface.

    Mirrors :func:`src.dupire_discovery.run_dupire_on_real_data` but uses
    GP-analytical derivatives in place of finite differences.
    """
    out = {}
    for ticker, entry in per_ticker_results.items():
        try:
            C, K_grid, tau_grid = _extract_call_surface(entry)
            # Resample tau onto a uniform grid if needed -- GP itself does
            # not require uniformity, but downstream consumers do.
            tau_uniform = np.linspace(
                float(tau_grid[0]), float(tau_grid[-1]), len(tau_grid),
            )
            if not np.allclose(tau_grid, tau_uniform, rtol=1e-6, atol=1e-9):
                C_uniform = np.empty_like(C)
                for i in range(C.shape[0]):
                    C_uniform[i, :] = np.interp(tau_uniform, tau_grid, C[i, :])
                C = C_uniform
                tau_grid = tau_uniform

            res = _gp_dupire_on_surface(
                C, K_grid, tau_grid, standardize=standardize,
            )
            res['ticker'] = ticker
            out[ticker] = res
            logger.info(
                "%s GP-Dupire: R2=%.4f sigma=%.4f r-q=%.4f active=%s",
                ticker, res['r2_score'], res['sigma_discovered'],
                res['drift_discovered'], res['active_terms'],
            )
        except Exception as exc:
            logger.error("GP-Dupire failed for %s: %s", ticker, exc,
                         exc_info=False)
            out[ticker] = {
                'ticker': ticker,
                'error': type(exc).__name__,
                'message': str(exc),
            }
    return out


def compare_dupire_methods(per_ticker_results):
    """Side-by-side comparison of FD-Dupire vs GP-Dupire per ticker.

    Returns
    -------
    pandas.DataFrame
        Columns: ``ticker, r2_fd, r2_gp, sigma_fd, sigma_gp,
        drift_fd, drift_gp, avg_market_iv``.
    """
    from src.dupire_discovery import run_dupire_on_real_data

    fd_results = run_dupire_on_real_data(per_ticker_results)
    gp_results = run_gp_dupire_on_real_data(per_ticker_results)

    rows = []
    for ticker in per_ticker_results.keys():
        fd = fd_results.get(ticker, {})
        gp = gp_results.get(ticker, {})
        entry = per_ticker_results.get(ticker, {})
        avg_iv = float(entry.get('avg_implied_vol', float('nan')))

        def _g(d, k):
            v = d.get(k)
            return float(v) if v is not None and not isinstance(v, str) else float('nan')

        rows.append({
            'ticker': ticker,
            'r2_fd': _g(fd, 'r2_score'),
            'r2_gp': _g(gp, 'r2_score'),
            'sigma_fd': _g(fd, 'sigma_discovered'),
            'sigma_gp': _g(gp, 'sigma_discovered'),
            'drift_fd': _g(fd, 'drift_discovered'),
            'drift_gp': _g(gp, 'drift_discovered'),
            'avg_market_iv': avg_iv,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Improvement #5 -- Windowed local-vol extraction
# ---------------------------------------------------------------------------

def _local_dupire_sigma(C_win, K_win, tau_win, seed=42,
                       threshold=0.05, n_subsample=200):
    """Fit a local GP and run a single-threshold Dupire STLSQ.

    Returns
    -------
    sigma_local, r2 : floats (``nan`` if window cannot be fit).
    """
    if C_win.shape[0] < 5 or C_win.shape[1] < 5:
        return float('nan'), float('nan')

    set_all_seeds(seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            gp, _ = fit_gp_surface(
                C_win, K_win, tau_win,
                n_subsample=min(n_subsample, C_win.size),
                seed=seed,
            )
        except Exception as exc:
            logger.debug("Local GP fit failed: %s", exc)
            return float('nan'), float('nan')

    derivs = compute_gp_derivatives(gp, K_win, tau_win)

    # Modest trim
    trim = 2 if min(C_win.shape) >= 11 else 1
    s = slice(trim, -trim) if trim > 0 else slice(None)
    C_tr = derivs['V_smooth'][s, s]
    dCdtau = derivs['dV_dt'][s, s]
    dCdK = derivs['dV_dS'][s, s]
    d2CdK2 = derivs['d2V_dS2'][s, s]

    K_tr = K_win[s]
    K_mesh = np.broadcast_to(K_tr[:, None], C_tr.shape)

    library = np.column_stack([
        C_tr.ravel(),
        dCdK.ravel(),
        d2CdK2.ravel(),
        (K_mesh * dCdK).ravel(),
        (K_mesh ** 2 * d2CdK2).ravel(),
    ]).astype(np.float64)
    target = dCdtau.ravel()

    # Standardize for numerical stability at small strike scales
    col_std = library.std(axis=0)
    col_std_safe = np.where(col_std < 1e-15, 1.0, col_std)
    lib_fit = library / col_std_safe

    try:
        coeffs_std, active = stlsq(lib_fit, target, threshold)
    except Exception as exc:
        logger.debug("Local STLSQ failed: %s", exc)
        return float('nan'), float('nan')

    coeffs = coeffs_std / col_std_safe
    pred = library @ coeffs
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0

    idx = DUPIRE_TERM_NAMES.index('K2*d2C/dK2')
    coeff_K2 = float(coeffs[idx])
    if coeff_K2 > 0:
        sigma_local = float(np.sqrt(2.0 * coeff_K2))
    else:
        sigma_local = float('nan')

    return sigma_local, float(r2)


def windowed_local_vol_extraction(option_data, ticker, window_size=15,
                                  stride=3, min_r2=0.5):
    """Slide a window across the (K, tau) surface and extract local sigma.

    Parameters
    ----------
    option_data : dict
        Either a per-ticker result dict (with ``surface_data``) or a
        ``surface_data``-style dict carrying ``V_surface``, ``K_grid``,
        ``tau_grid``.
    ticker : str
        Ticker label (used in logs only).
    window_size : int, default 15
        Edge length of the square window in grid units.  225 points at the
        default size is well above the 30-point minimum requested.
    stride : int, default 3
        Step (in grid units) between successive window centers.
    min_r2 : float, default 0.5
        Discard windows where the local Dupire fit drops below this R^2.

    Returns
    -------
    dict
        Keys ``K_centers``, ``tau_centers``, ``sigma_local_grid`` (2-D),
        ``r2_grid``, ``n_valid_windows``, ``n_total_windows``, ``ticker``.
    """
    # Accept either a per-ticker entry or a raw surface_data dict.
    if 'surface_data' in option_data:
        surface = option_data['surface_data']
    elif 'V_surface' in option_data:
        surface = option_data
    else:
        raise KeyError(
            "option_data needs either 'surface_data' or 'V_surface'"
        )

    C = np.asarray(surface['V_surface'], dtype=float)
    K_grid = np.asarray(surface['K_grid'], dtype=float)
    tau_grid = np.asarray(surface['tau_grid'], dtype=float)

    n_K, n_tau = C.shape
    if window_size > min(n_K, n_tau):
        window_size = max(min(n_K, n_tau) - 1, 5)
        logger.warning(
            "%s window_size clamped to %d", ticker, window_size,
        )

    K_starts = list(range(0, n_K - window_size + 1, stride))
    tau_starts = list(range(0, n_tau - window_size + 1, stride))
    n_total = len(K_starts) * len(tau_starts)

    sigma_grid = np.full((len(K_starts), len(tau_starts)), np.nan)
    r2_grid = np.full((len(K_starts), len(tau_starts)), np.nan)
    K_centers = np.zeros(len(K_starts))
    tau_centers = np.zeros(len(tau_starts))

    for i, ks in enumerate(K_starts):
        K_centers[i] = float(K_grid[ks:ks + window_size].mean())
    for j, ts in enumerate(tau_starts):
        tau_centers[j] = float(tau_grid[ts:ts + window_size].mean())

    n_valid = 0
    for i, ks in enumerate(K_starts):
        for j, ts in enumerate(tau_starts):
            C_win = C[ks:ks + window_size, ts:ts + window_size]
            K_win = K_grid[ks:ks + window_size]
            tau_win = tau_grid[ts:ts + window_size]

            try:
                sigma_local, r2 = _local_dupire_sigma(
                    C_win, K_win, tau_win, seed=42,
                )
            except Exception as exc:
                logger.debug("Window (%d,%d) failed: %s", i, j, exc)
                sigma_local, r2 = float('nan'), float('nan')

            r2_grid[i, j] = r2
            if (np.isfinite(sigma_local)
                    and np.isfinite(r2)
                    and r2 >= min_r2
                    and 0.0 < sigma_local < 2.0):
                sigma_grid[i, j] = sigma_local
                n_valid += 1

    logger.info(
        "%s windowed-local-vol: %d / %d valid windows",
        ticker, n_valid, n_total,
    )

    return {
        'ticker': ticker,
        'K_centers': K_centers,
        'tau_centers': tau_centers,
        'sigma_local_grid': sigma_grid,
        'r2_grid': r2_grid,
        'n_valid_windows': int(n_valid),
        'n_total_windows': int(n_total),
        'window_size': int(window_size),
        'stride': int(stride),
        'min_r2': float(min_r2),
    }


# ---------------------------------------------------------------------------
# Fix #2 -- Compare RBF vs Matern kernels for GP-SINDy on real data.
# ---------------------------------------------------------------------------

def _gp_sindy_on_surface_kernel(V_st, S_grid, t_grid, kernel='rbf',
                                seed=42, standardize=True, n_subsample=500):
    """Same as :func:`_gp_sindy_on_surface` but explicit kernel selection."""
    set_all_seeds(seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        info = fit_gp_surface(
            V_st, S_grid, t_grid, n_subsample=n_subsample, seed=seed,
            kernel=kernel, return_info=True,
        )

    gp = info['gp']
    derivs = compute_gp_derivatives(gp, S_grid, t_grid)

    trim = _adaptive_trim(V_st.shape)
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

    if standardize:
        col_std = library.std(axis=0)
        col_std_safe = np.where(col_std < 1e-15, 1.0, col_std)
        best, _ = stlsq_sweep(library / col_std_safe, target)
        discovered = best['coefficients'] / col_std_safe
    else:
        best, _ = stlsq_sweep(library, target)
        discovered = best['coefficients']

    return {
        'r2_score': float(best['r2']),
        'discovered': np.asarray(discovered, dtype=float),
        'length_scales': info['length_scales'],
        'noise_level': info['noise_level'],
        'kernel_used': info['kernel_used'],
    }


def compare_gp_kernels_on_real_data(per_ticker_results, n_subsample=500,
                                     standardize=True, seed=42):
    """For each ticker, fit GP-SINDy with both RBF and Matern kernels.

    Returns
    -------
    pandas.DataFrame
        Columns: ``ticker, r2_rbf, r2_matern, kernel_winner,
        length_scales_rbf, length_scales_matern``.
    """
    rows = []
    for ticker, entry in per_ticker_results.items():
        row = {'ticker': ticker}
        try:
            surface = entry.get('surface_data')
            if surface is None:
                raise KeyError("missing surface_data")
            C = np.asarray(surface['V_surface'], dtype=float)
            K_grid = np.asarray(surface['K_grid'], dtype=float)
            tau_grid = np.asarray(surface['tau_grid'], dtype=float)
            V_st, S_grid, t_grid, _ = _bs_to_st_axes(C, K_grid, tau_grid)

            r2_rbf, r2_matern = float('nan'), float('nan')
            ls_rbf, ls_matern = None, None

            try:
                res = _gp_sindy_on_surface_kernel(
                    V_st, S_grid, t_grid, kernel='rbf',
                    seed=seed, standardize=standardize,
                    n_subsample=n_subsample,
                )
                r2_rbf = res['r2_score']
                ls_rbf = res['length_scales']
            except Exception as exc:
                logger.warning("RBF GP failed for %s: %s", ticker, exc)

            try:
                res = _gp_sindy_on_surface_kernel(
                    V_st, S_grid, t_grid, kernel='matern',
                    seed=seed, standardize=standardize,
                    n_subsample=n_subsample,
                )
                r2_matern = res['r2_score']
                ls_matern = res['length_scales']
            except Exception as exc:
                logger.warning("Matern GP failed for %s: %s", ticker, exc)

            if np.isfinite(r2_rbf) and np.isfinite(r2_matern):
                winner = 'matern' if r2_matern > r2_rbf else 'rbf'
            elif np.isfinite(r2_matern):
                winner = 'matern'
            elif np.isfinite(r2_rbf):
                winner = 'rbf'
            else:
                winner = 'none'

            row.update({
                'r2_rbf': r2_rbf,
                'r2_matern': r2_matern,
                'kernel_winner': winner,
                'length_scales_rbf': ls_rbf,
                'length_scales_matern': ls_matern,
            })
        except Exception as exc:
            logger.error("Kernel comparison failed for %s: %s", ticker, exc)
            row.update({
                'r2_rbf': float('nan'),
                'r2_matern': float('nan'),
                'kernel_winner': 'error',
                'length_scales_rbf': None,
                'length_scales_matern': None,
            })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fix #3 -- Try multiple GP-derivative approaches for Dupire.
# ---------------------------------------------------------------------------

def _dupire_sindy_from_derivs(C_tr, dCdtau, dCdK, d2CdK2, K_tr,
                              standardize=True):
    """Common tail: build Dupire library + STLSQ from a set of derivatives."""
    K_mesh = np.broadcast_to(K_tr[:, None], C_tr.shape)
    library = np.column_stack([
        C_tr.ravel(),
        dCdK.ravel(),
        d2CdK2.ravel(),
        (K_mesh * dCdK).ravel(),
        (K_mesh ** 2 * d2CdK2).ravel(),
    ]).astype(np.float64)
    target = dCdtau.ravel()

    if standardize:
        col_std = library.std(axis=0)
        col_std_safe = np.where(col_std < 1e-15, 1.0, col_std)
        best, _ = stlsq_sweep(library / col_std_safe, target)
        discovered = best['coefficients'] / col_std_safe
    else:
        best, _ = stlsq_sweep(library, target)
        discovered = best['coefficients']

    coeff_K2 = float(discovered[DUPIRE_TERM_NAMES.index('K2*d2C/dK2')])
    coeff_K = float(discovered[DUPIRE_TERM_NAMES.index('K*dC/dK')])
    sigma_disc = float(np.sqrt(2.0 * coeff_K2)) if coeff_K2 > 0 else float('nan')
    drift_disc = float(-coeff_K)
    return {
        'r2_score': float(best['r2']),
        'sigma_discovered': sigma_disc,
        'drift_discovered': drift_disc,
        'coefficients': np.asarray(discovered, dtype=float),
        'active_mask': np.asarray(best['active_mask'], dtype=bool),
    }


def dupire_with_gp_approach(C_surface, K_grid, tau_grid,
                            approach='constrained_ls',
                            seed=42, standardize=True, n_subsample=500,
                            **kwargs):
    """
    Try one of four GP-derivative approaches for Dupire discovery.

    Parameters
    ----------
    C_surface : ndarray, shape (n_K, n_tau)
    K_grid, tau_grid : 1-D ndarrays
    approach : {'constrained_ls', 'short_ls_init', 'stratified',
                'gp_smooth_fd_deriv'}
    seed : int
    standardize : bool
    n_subsample : int

    Returns
    -------
    dict
        Keys ``r2_score``, ``sigma_discovered``, ``drift_discovered``,
        ``coefficients``, ``approach_used``.  On failure, returns NaN R^2
        and ``approach_used`` reflecting the attempted approach.
    """
    set_all_seeds(seed)
    n_K = len(K_grid)
    K_extent = float(K_grid[-1] - K_grid[0])
    dK = float(K_grid[1] - K_grid[0]) if n_K > 1 else 1.0
    trim = _adaptive_trim(C_surface.shape)
    s = slice(trim, -trim) if trim > 0 else slice(None)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            if approach == 'constrained_ls':
                # Bound K-axis length scale to (1, K_extent/5).  Lets the GP
                # learn a smooth-ish surface but never wash out curvature.
                ls_hi = max(K_extent / 5.0, 2.0 * dK)
                gp, _ = fit_gp_surface_constrained(
                    C_surface, K_grid, tau_grid,
                    n_subsample=n_subsample, seed=seed,
                    ls_bounds_S=(1.0, ls_hi),
                    ls_init_S=min(2.0 * dK, ls_hi * 0.5),
                )
                derivs = compute_gp_derivatives(gp, K_grid, tau_grid)

            elif approach == 'short_ls_init':
                # Initialise K length scale to ~5 strikes, allow many
                # restarts so optimiser can still leave the basin.
                ls_init = max(5.0 * dK, 0.01 * K_extent)
                gp, _ = fit_gp_surface_constrained(
                    C_surface, K_grid, tau_grid,
                    n_subsample=n_subsample, seed=seed,
                    ls_init_S=ls_init,
                    n_restarts_optimizer=10,
                )
                derivs = compute_gp_derivatives(gp, K_grid, tau_grid)

            elif approach == 'stratified':
                gp, _ = fit_gp_surface_constrained(
                    C_surface, K_grid, tau_grid,
                    n_subsample=n_subsample, seed=seed,
                    stratified=True,
                )
                derivs = compute_gp_derivatives(gp, K_grid, tau_grid)

            elif approach == 'gp_smooth_fd_deriv':
                # Fit GP, evaluate on full grid, use FD derivatives.
                info = fit_gp_surface(
                    C_surface, K_grid, tau_grid,
                    n_subsample=n_subsample, seed=seed,
                    return_info=True,
                )
                gp = info['gp']
                K_mesh, tau_mesh = np.meshgrid(K_grid, tau_grid, indexing='ij')
                X_pred = np.column_stack([K_mesh.ravel(), tau_mesh.ravel()])
                C_smooth = gp.predict(X_pred).reshape(C_surface.shape)
                # Use NumericalDifferentiator with no further smoothing.
                from src.utils import NumericalDifferentiator
                diff = NumericalDifferentiator(order=2, smooth=False)
                dtau = float(tau_grid[1] - tau_grid[0])
                derivs = {
                    'V_smooth': C_smooth,
                    'dV_dt': diff.first_derivative(C_smooth, dtau, axis=1),
                    'dV_dS': diff.first_derivative(C_smooth, dK, axis=0),
                    'd2V_dS2': diff.second_derivative(C_smooth, dK, axis=0),
                }
            else:
                raise ValueError(f"Unknown approach: {approach}")

            C_tr = derivs['V_smooth'][s, s]
            dCdtau = derivs['dV_dt'][s, s]
            dCdK = derivs['dV_dS'][s, s]
            d2CdK2 = derivs['d2V_dS2'][s, s]
            K_tr = K_grid[s]

            out = _dupire_sindy_from_derivs(
                C_tr, dCdtau, dCdK, d2CdK2, K_tr, standardize=standardize,
            )
            out['approach_used'] = approach
            return out
    except Exception as exc:
        logger.warning("dupire_with_gp_approach[%s] failed: %s",
                       approach, exc)
        return {
            'r2_score': float('nan'),
            'sigma_discovered': float('nan'),
            'drift_discovered': float('nan'),
            'coefficients': np.full(len(DUPIRE_TERM_NAMES), np.nan),
            'approach_used': approach,
        }


def compare_dupire_approaches_synthetic(K_min=70.0, K_max=130.0, n_K=50,
                                        tau_min=0.05, tau_max=1.5, n_tau=50,
                                        S0=100.0, r=0.05, sigma=0.20,
                                        seed=42, n_subsample=500):
    """Run the 4 GP-Dupire approaches + baseline FD-Dupire + baseline GP-Dupire
    on a synthetic Black-Scholes Dupire surface, and pick a winning approach.

    Returns
    -------
    pandas.DataFrame
        Columns: ``approach, r2_score, sigma_recovered, sigma_rel_error``.
        Includes a ``best_approach`` attribute on the DataFrame (string).
    """
    from src.data_generation import bs_call_price
    from src.dupire_discovery import discover_dupire

    set_all_seeds(seed)
    K_grid = np.linspace(float(K_min), float(K_max), int(n_K), dtype=np.float64)
    tau_grid = np.linspace(float(tau_min), float(tau_max), int(n_tau),
                           dtype=np.float64)
    C = np.zeros((len(K_grid), len(tau_grid)), dtype=np.float64)
    for j, tau in enumerate(tau_grid):
        C[:, j] = bs_call_price(float(S0), K_grid, float(r), float(sigma),
                                float(tau))

    rows = []

    # Baseline FD-Dupire
    try:
        fd_res = discover_dupire(C, K_grid, tau_grid, smooth=True)
        rows.append({
            'approach': 'fd_dupire',
            'r2_score': float(fd_res['r2_score']),
            'sigma_recovered': float(fd_res['sigma_discovered']),
            'sigma_rel_error': abs(float(fd_res['sigma_discovered']) - sigma) / sigma
            if np.isfinite(fd_res['sigma_discovered']) else float('nan'),
        })
    except Exception as exc:
        logger.warning("FD-Dupire baseline failed: %s", exc)
        rows.append({'approach': 'fd_dupire', 'r2_score': float('nan'),
                     'sigma_recovered': float('nan'),
                     'sigma_rel_error': float('nan')})

    # Baseline GP-Dupire (existing pipeline)
    try:
        baseline = _gp_dupire_on_surface(C, K_grid, tau_grid,
                                         n_subsample=n_subsample)
        rows.append({
            'approach': 'gp_dupire_baseline',
            'r2_score': float(baseline['r2_score']),
            'sigma_recovered': float(baseline['sigma_discovered']),
            'sigma_rel_error': abs(float(baseline['sigma_discovered']) - sigma) / sigma
            if np.isfinite(baseline['sigma_discovered']) else float('nan'),
        })
    except Exception as exc:
        logger.warning("GP-Dupire baseline failed: %s", exc)
        rows.append({'approach': 'gp_dupire_baseline',
                     'r2_score': float('nan'),
                     'sigma_recovered': float('nan'),
                     'sigma_rel_error': float('nan')})

    for approach in ['constrained_ls', 'short_ls_init', 'stratified',
                     'gp_smooth_fd_deriv']:
        res = dupire_with_gp_approach(
            C, K_grid, tau_grid, approach=approach,
            seed=seed, n_subsample=n_subsample,
        )
        sig = res['sigma_discovered']
        rel = abs(sig - sigma) / sigma if np.isfinite(sig) else float('nan')
        rows.append({
            'approach': approach,
            'r2_score': float(res['r2_score']),
            'sigma_recovered': float(sig),
            'sigma_rel_error': float(rel),
        })

    df = pd.DataFrame(rows)

    # Pick winning approach: highest R^2 with sigma within 10% of truth.
    valid = df[(df['r2_score'].notna())
               & (df['sigma_rel_error'] < 0.10)]
    if len(valid) > 0:
        best_idx = valid['r2_score'].idxmax()
        best_approach = str(df.loc[best_idx, 'approach'])
    elif df['r2_score'].notna().any():
        best_approach = str(df.loc[df['r2_score'].idxmax(), 'approach'])
    else:
        best_approach = 'none'

    df.attrs['best_approach'] = best_approach
    logger.info("Dupire-approach synthetic comparison:\n%s\nWinner: %s",
                df.to_string(index=False), best_approach)
    return df


def compare_dupire_approaches_real(per_ticker_results, best_approach,
                                    seed=42, n_subsample=500,
                                    standardize=True):
    """Run the winning Dupire approach on each ticker's call surface.

    Returns
    -------
    pandas.DataFrame
        Columns: ``ticker, r2_score, sigma_recovered, drift_recovered,
        avg_market_iv, sigma_rel_err, approach_used``.
    """
    rows = []
    for ticker, entry in per_ticker_results.items():
        row = {'ticker': ticker, 'approach_used': best_approach}
        try:
            C, K_grid, tau_grid = _extract_call_surface(entry)
            tau_uniform = np.linspace(
                float(tau_grid[0]), float(tau_grid[-1]), len(tau_grid),
            )
            if not np.allclose(tau_grid, tau_uniform, rtol=1e-6, atol=1e-9):
                C_uni = np.empty_like(C)
                for i in range(C.shape[0]):
                    C_uni[i, :] = np.interp(tau_uniform, tau_grid, C[i, :])
                C = C_uni
                tau_grid = tau_uniform

            avg_iv = float(entry.get('avg_implied_vol', float('nan')))

            if best_approach in ('constrained_ls', 'short_ls_init',
                                 'stratified', 'gp_smooth_fd_deriv'):
                res = dupire_with_gp_approach(
                    C, K_grid, tau_grid, approach=best_approach,
                    seed=seed, standardize=standardize,
                    n_subsample=n_subsample,
                )
                r2 = res['r2_score']
                sig = res['sigma_discovered']
                drift = res['drift_discovered']
            elif best_approach == 'fd_dupire':
                from src.dupire_discovery import discover_dupire
                res = discover_dupire(C, K_grid, tau_grid, smooth=True)
                r2 = float(res['r2_score'])
                sig = float(res['sigma_discovered'])
                drift = float(res['drift_discovered'])
            elif best_approach == 'gp_dupire_baseline':
                res = _gp_dupire_on_surface(
                    C, K_grid, tau_grid, standardize=standardize,
                    n_subsample=n_subsample,
                )
                r2 = float(res['r2_score'])
                sig = float(res['sigma_discovered'])
                drift = float(res['drift_discovered'])
            else:
                raise ValueError(f"Unknown best_approach: {best_approach}")

            rel = (abs(sig - avg_iv) / avg_iv
                   if np.isfinite(sig) and np.isfinite(avg_iv) and avg_iv > 0
                   else float('nan'))
            row.update({
                'r2_score': r2,
                'sigma_recovered': sig,
                'drift_recovered': drift,
                'avg_market_iv': avg_iv,
                'sigma_rel_err': rel,
            })
        except Exception as exc:
            logger.error("compare_dupire_approaches_real failed for %s: %s",
                         ticker, exc)
            row.update({
                'r2_score': float('nan'),
                'sigma_recovered': float('nan'),
                'drift_recovered': float('nan'),
                'avg_market_iv': float('nan'),
                'sigma_rel_err': float('nan'),
            })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PRD Part A — Leave-one-expiration-out CV for Dupire approach selection
# ---------------------------------------------------------------------------

#: The 4 approaches tested by the CV selector.
_DUPIRE_CV_APPROACHES = (
    'fd_baseline',
    'savgol_fd',
    'gp_smooth_fd_deriv',
    'stratified_gp',
)


def _build_surface_from_df(df_sub, S0, r, n_K=40):
    """Build a (K, tau) call surface from a subset of ``option_df`` rows.

    Mirrors :func:`src.real_data.construct_smooth_surface` but works on a
    pre-filtered dataframe (i.e. an arbitrary subset of the chain).  Returns
    ``None`` if there is not enough data to build a surface.
    """
    from scipy.interpolate import griddata
    from src.data_generation import bs_call_price

    iv = df_sub['implied_vol'].values.astype(float)
    valid = np.isfinite(iv) & (iv > 0) & (iv <= 2.0)
    df_sub = df_sub.loc[valid].copy()
    if len(df_sub) < 10:
        return None

    strikes = df_sub['strike'].values.astype(float)
    taus = df_sub['tau'].values.astype(float)
    ivs = df_sub['implied_vol'].values.astype(float)

    K_min, K_max = float(strikes.min()), float(strikes.max())
    if K_max - K_min < 1e-6:
        return None
    K_grid = np.linspace(K_min, K_max, n_K)

    unique_taus = np.sort(np.unique(np.round(taus, 6)))
    if len(unique_taus) < 3:
        return None

    tau_grid = unique_taus.astype(float)
    KK, TT = np.meshgrid(K_grid, tau_grid, indexing='ij')

    points = np.column_stack([strikes, taus])
    try:
        iv_surface = griddata(points, ivs, (KK, TT), method='linear')
    except Exception:
        iv_surface = griddata(points, ivs, (KK, TT), method='nearest')

    nan_mask = np.isnan(iv_surface)
    if np.any(nan_mask):
        iv_nearest = griddata(points, ivs, (KK, TT), method='nearest')
        iv_surface[nan_mask] = iv_nearest[nan_mask]
    iv_surface = np.clip(iv_surface, 0.01, 2.0)

    V_surface = np.zeros_like(KK)
    for i in range(KK.shape[0]):
        for j in range(KK.shape[1]):
            K_val, tau_val, sigma_val = KK[i, j], TT[i, j], iv_surface[i, j]
            if tau_val <= 0 or sigma_val <= 0:
                V_surface[i, j] = max(S0 - K_val, 0.0)
            else:
                V_surface[i, j] = float(
                    bs_call_price(S0, K_val, r, sigma_val, tau_val)
                )

    # Resample onto a uniform tau grid for the FD/GP pipeline.
    tau_uniform = np.linspace(float(tau_grid[0]), float(tau_grid[-1]),
                              len(tau_grid))
    if not np.allclose(tau_grid, tau_uniform, rtol=1e-6, atol=1e-9):
        V_uni = np.empty_like(V_surface)
        for i in range(V_surface.shape[0]):
            V_uni[i, :] = np.interp(tau_uniform, tau_grid, V_surface[i, :])
        V_surface = V_uni
        tau_grid = tau_uniform

    return V_surface, K_grid, tau_grid


def _fit_dupire_approach(C, K_grid, tau_grid, approach, seed=42,
                         n_subsample=500):
    """Fit a single Dupire approach and return discovered coefficients.

    Returns
    -------
    coeffs : ndarray, shape (5,)
        Coefficients in DUPIRE_TERM_NAMES order, or NaNs on failure.
    """
    from src.dupire_discovery import discover_dupire
    nan_coeffs = np.full(len(DUPIRE_TERM_NAMES), np.nan)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if approach == 'fd_baseline':
                res = discover_dupire(C, K_grid, tau_grid, smooth=True,
                                      savgol_window=11, savgol_poly=3)
                return np.asarray(res['discovered_coefficients'], dtype=float)
            if approach == 'savgol_fd':
                # Heavier smoothing window than the FD baseline.
                res = discover_dupire(C, K_grid, tau_grid, smooth=True,
                                      savgol_window=15, savgol_poly=3)
                return np.asarray(res['discovered_coefficients'], dtype=float)
            if approach == 'gp_smooth_fd_deriv':
                res = dupire_with_gp_approach(
                    C, K_grid, tau_grid, approach='gp_smooth_fd_deriv',
                    seed=seed, n_subsample=n_subsample,
                )
                return np.asarray(res['coefficients'], dtype=float)
            if approach == 'stratified_gp':
                res = dupire_with_gp_approach(
                    C, K_grid, tau_grid, approach='stratified',
                    seed=seed, n_subsample=n_subsample,
                )
                return np.asarray(res['coefficients'], dtype=float)
            raise ValueError(f"Unknown approach: {approach}")
    except Exception as exc:
        logger.warning("Dupire approach %s failed: %s", approach, exc)
        return nan_coeffs


def _predict_dcdtau(C_slice, K_grid, coeffs):
    """Evaluate the Dupire RHS at a single tau slice given discovered coeffs.

    Parameters
    ----------
    C_slice : ndarray, shape (n_K,)
        Call price as a function of strike at one expiration.
    K_grid : ndarray, shape (n_K,)
    coeffs : ndarray, shape (5,)
        [C, dC/dK, d2C/dK2, K*dC/dK, K^2*d2C/dK2]

    Returns
    -------
    dCdtau_pred : ndarray, shape (n_K,)
    """
    from src.utils import NumericalDifferentiator
    dK = float(K_grid[1] - K_grid[0])
    diff = NumericalDifferentiator(order=2, smooth=False)
    C_arr = np.asarray(C_slice, dtype=float)[:, None]  # (n_K, 1)
    dCdK = diff.first_derivative(C_arr, dK, axis=0).ravel()
    d2CdK2 = diff.second_derivative(C_arr, dK, axis=0).ravel()
    C_flat = C_arr.ravel()
    K = np.asarray(K_grid, dtype=float)
    c1, c2, c3, c4, c5 = (float(x) for x in coeffs)
    return (c1 * C_flat
            + c2 * dCdK
            + c3 * d2CdK2
            + c4 * K * dCdK
            + c5 * (K ** 2) * d2CdK2)


def _actual_dcdtau_at_holdout(option_data, holdout_tau, S0, r,
                              K_grid, n_K=40):
    """Approximate actual dC/dtau at the held-out expiration via centred FD
    using the two neighbouring expirations (or one-sided FD at the edges).
    """
    df = option_data['option_df']
    unique_taus = np.sort(np.unique(np.round(df['tau'].values.astype(float),
                                             6)))
    idx = int(np.argmin(np.abs(unique_taus - holdout_tau)))

    def _slice_at(tau_target):
        from scipy.interpolate import griddata
        from src.data_generation import bs_call_price
        rows = df[np.isclose(df['tau'].values.astype(float), tau_target,
                             atol=1e-6)]
        if len(rows) < 3:
            return None
        ivs = rows['implied_vol'].values.astype(float)
        strikes = rows['strike'].values.astype(float)
        ok = np.isfinite(ivs) & (ivs > 0) & (ivs <= 2.0)
        if ok.sum() < 3:
            return None
        # Interpolate IV at K_grid (linear, with nearest fallback at edges).
        try:
            iv_interp = griddata(strikes[ok], ivs[ok], K_grid,
                                 method='linear')
        except Exception:
            iv_interp = griddata(strikes[ok], ivs[ok], K_grid,
                                 method='nearest')
        nan_mask = np.isnan(iv_interp)
        if nan_mask.any():
            iv_near = griddata(strikes[ok], ivs[ok], K_grid,
                               method='nearest')
            iv_interp[nan_mask] = iv_near[nan_mask]
        iv_interp = np.clip(iv_interp, 0.01, 2.0)
        slc = np.array([
            bs_call_price(S0, float(K), r, float(iv), float(tau_target))
            for K, iv in zip(K_grid, iv_interp)
        ])
        return slc

    C_hold = _slice_at(holdout_tau)
    if C_hold is None:
        return None, None

    # Try centred FD using neighbours; fall back to one-sided.
    if 0 < idx < len(unique_taus) - 1:
        tau_lo, tau_hi = unique_taus[idx - 1], unique_taus[idx + 1]
        C_lo, C_hi = _slice_at(tau_lo), _slice_at(tau_hi)
        if C_lo is None or C_hi is None:
            return C_hold, None
        dCdtau = (C_hi - C_lo) / (tau_hi - tau_lo)
        return C_hold, dCdtau
    if idx == 0 and len(unique_taus) >= 2:
        tau_hi = unique_taus[idx + 1]
        C_hi = _slice_at(tau_hi)
        if C_hi is None:
            return C_hold, None
        return C_hold, (C_hi - C_hold) / (tau_hi - holdout_tau)
    if idx == len(unique_taus) - 1 and len(unique_taus) >= 2:
        tau_lo = unique_taus[idx - 1]
        C_lo = _slice_at(tau_lo)
        if C_lo is None:
            return C_hold, None
        return C_hold, (C_hold - C_lo) / (holdout_tau - tau_lo)
    return C_hold, None


def dupire_cv_select_approach(option_data, ticker, approaches=None,
                              normalize_moneyness=False, seed=42):
    """Leave-one-expiration-out CV to pick the best Dupire derivative approach.

    For each unique expiration in ``option_data['option_df']``:
      1. Hold out that expiration.
      2. Construct C(K, tau) surface from the rest.
      3. Run Dupire SINDy with each approach on the reduced surface.
      4. Use the discovered PDE to predict dC/dtau at the held-out expiration.
      5. Record prediction error (RMSE) vs an FD estimate of the actual
         dC/dtau across the two neighbouring expirations.

    Parameters
    ----------
    option_data : dict
        ``option_data`` portion of a per-ticker entry; must carry
        ``option_df``, ``S0`` and ``r``.
    ticker : str
    approaches : iterable of str or None
        Defaults to the 4 PRD approaches.
    normalize_moneyness : bool, default False
        Replace ``K`` with ``K / S0`` before building surfaces / running SINDy.
    seed : int, default 42

    Returns
    -------
    dict
        Keys: ``ticker``, ``approaches_tested``, ``mean_errors``,
        ``best_approach``, ``per_fold_errors_df``, ``normalize_moneyness``.
    """
    set_all_seeds(seed)
    if approaches is None:
        approaches = list(_DUPIRE_CV_APPROACHES)
    approaches = list(approaches)

    df = option_data.get('option_df')
    if df is None or len(df) == 0:
        return {
            'ticker': ticker,
            'approaches_tested': approaches,
            'mean_errors': {a: float('nan') for a in approaches},
            'best_approach': None,
            'per_fold_errors_df': pd.DataFrame(),
            'normalize_moneyness': bool(normalize_moneyness),
        }

    df = df.copy()
    S0 = float(option_data['S0'])
    r = float(option_data['r'])

    if normalize_moneyness:
        df['strike'] = df['strike'].astype(float) / S0
        S0_eff = 1.0
    else:
        S0_eff = S0

    unique_taus = np.sort(np.unique(np.round(df['tau'].values.astype(float),
                                             6)))
    fold_rows = []

    for holdout_tau in unique_taus:
        remaining_taus = [t for t in unique_taus
                          if not np.isclose(t, holdout_tau, atol=1e-6)]
        if len(remaining_taus) < 3:
            logger.info("CV(%s): skipping fold tau=%.4f (only %d "
                        "expirations left after holdout)",
                        ticker, holdout_tau, len(remaining_taus))
            continue

        train_mask = ~np.isclose(df['tau'].values.astype(float), holdout_tau,
                                 atol=1e-6)
        df_train = df.loc[train_mask].copy()

        surf = _build_surface_from_df(df_train, S0_eff, r, n_K=40)
        if surf is None:
            logger.info("CV(%s): fold tau=%.4f surface build failed",
                        ticker, holdout_tau)
            continue
        C_train, K_grid, tau_grid = surf

        # Actual held-out target.
        option_data_local = dict(option_data)
        option_data_local['option_df'] = df  # already moneyness-adjusted
        C_hold, dCdtau_actual = _actual_dcdtau_at_holdout(
            option_data_local, float(holdout_tau), S0_eff, r, K_grid,
        )
        if C_hold is None or dCdtau_actual is None:
            logger.info("CV(%s): fold tau=%.4f cannot estimate actual "
                        "dC/dtau", ticker, holdout_tau)
            continue

        for approach in approaches:
            coeffs = _fit_dupire_approach(C_train, K_grid, tau_grid,
                                          approach, seed=seed)
            if not np.all(np.isfinite(coeffs)):
                rmse = float('nan')
            else:
                try:
                    dCdtau_pred = _predict_dcdtau(C_hold, K_grid, coeffs)
                    # Trim 2 grid points on each side to avoid FD edge noise.
                    if len(dCdtau_pred) > 6:
                        sl = slice(2, -2)
                    else:
                        sl = slice(None)
                    err = dCdtau_pred[sl] - dCdtau_actual[sl]
                    rmse = float(np.sqrt(np.mean(err ** 2)))
                except Exception as exc:
                    logger.warning("CV(%s) approach=%s prediction failed: %s",
                                   ticker, approach, exc)
                    rmse = float('nan')
            fold_rows.append({
                'ticker': ticker,
                'holdout_tau': float(holdout_tau),
                'approach': approach,
                'rmse': rmse,
            })

    per_fold_df = pd.DataFrame(fold_rows)
    mean_errors = {a: float('nan') for a in approaches}
    if not per_fold_df.empty:
        for approach in approaches:
            sub = per_fold_df[per_fold_df['approach'] == approach]
            vals = sub['rmse'].values.astype(float)
            finite = vals[np.isfinite(vals)]
            mean_errors[approach] = (float(np.mean(finite))
                                     if finite.size else float('nan'))
        # Per the PRD: if a single approach errors on a fold, use the mean of
        # other folds for that approach (this is what np.mean over the finite
        # subset already gives us).

    finite_means = {a: e for a, e in mean_errors.items() if np.isfinite(e)}
    if finite_means:
        best_approach = min(finite_means, key=finite_means.get)
    else:
        best_approach = None

    logger.info("Dupire CV(%s, moneyness=%s): mean_errors=%s -> best=%s",
                ticker, normalize_moneyness, mean_errors, best_approach)

    return {
        'ticker': ticker,
        'approaches_tested': approaches,
        'mean_errors': mean_errors,
        'best_approach': best_approach,
        'per_fold_errors_df': per_fold_df,
        'normalize_moneyness': bool(normalize_moneyness),
    }


def _final_dupire_with_approach(C, K_grid, tau_grid, approach, seed=42):
    """Run a single Dupire approach end-to-end (R², sigma, drift)."""
    from src.dupire_discovery import discover_dupire
    if approach == 'fd_baseline':
        res = discover_dupire(C, K_grid, tau_grid, smooth=True,
                              savgol_window=11, savgol_poly=3)
        return {
            'r2_score': float(res['r2_score']),
            'sigma_recovered': float(res['sigma_discovered']),
            'drift_recovered': float(res['drift_discovered']),
        }
    if approach == 'savgol_fd':
        res = discover_dupire(C, K_grid, tau_grid, smooth=True,
                              savgol_window=15, savgol_poly=3)
        return {
            'r2_score': float(res['r2_score']),
            'sigma_recovered': float(res['sigma_discovered']),
            'drift_recovered': float(res['drift_discovered']),
        }
    gp_alias = {'gp_smooth_fd_deriv': 'gp_smooth_fd_deriv',
                'stratified_gp': 'stratified'}
    res = dupire_with_gp_approach(
        C, K_grid, tau_grid, approach=gp_alias[approach], seed=seed,
    )
    return {
        'r2_score': float(res['r2_score']),
        'sigma_recovered': float(res['sigma_discovered']),
        'drift_recovered': float(res['drift_discovered']),
    }


def run_dupire_cv_on_real_data(per_ticker_results, tickers=None,
                                normalize_moneyness=True, seed=42):
    """For SPY and QQQ run CV; for AAPL/MSFT default to SPY's winner.

    Returns
    -------
    dict
        ``{ticker: {best_approach, mean_errors, per_fold_errors_df, final}}``
        where ``final`` carries ``r2_score``, ``sigma_recovered``,
        ``drift_recovered``, ``avg_market_iv`` evaluated with the chosen
        approach on that ticker's full surface.  Also includes
        ``'_meta': {'spy_winner': ..., 'normalize_moneyness': ...}``.
    """
    if tickers is None:
        tickers = list(per_ticker_results.keys())

    out = {}
    cv_tickers = [t for t in tickers if t in ('SPY', 'QQQ')]
    fallback_tickers = [t for t in tickers if t not in ('SPY', 'QQQ')]

    spy_winner = None
    for ticker in cv_tickers:
        entry = per_ticker_results.get(ticker)
        if entry is None:
            continue
        option_data = entry.get('option_data')
        if option_data is None or option_data.get('option_df') is None:
            logger.warning("CV: ticker %s has no option_df; skipping", ticker)
            continue
        try:
            cv_res = dupire_cv_select_approach(
                option_data, ticker,
                normalize_moneyness=normalize_moneyness, seed=seed,
            )
        except Exception as exc:
            logger.error("Dupire CV failed for %s: %s", ticker, exc,
                         exc_info=False)
            out[ticker] = {'error': str(exc)}
            continue
        best = cv_res['best_approach']
        if ticker == 'SPY':
            spy_winner = best

        # Evaluate the chosen approach on the full (non-CV) surface for the
        # final R² / sigma / drift.
        try:
            C, K_grid, tau_grid = _extract_call_surface(entry)
            tau_uniform = np.linspace(float(tau_grid[0]),
                                      float(tau_grid[-1]), len(tau_grid))
            if not np.allclose(tau_grid, tau_uniform, rtol=1e-6, atol=1e-9):
                C_uni = np.empty_like(C)
                for i in range(C.shape[0]):
                    C_uni[i, :] = np.interp(tau_uniform, tau_grid, C[i, :])
                C = C_uni
                tau_grid = tau_uniform
            if best is None:
                final = {'r2_score': float('nan'),
                         'sigma_recovered': float('nan'),
                         'drift_recovered': float('nan')}
            else:
                final = _final_dupire_with_approach(
                    C, K_grid, tau_grid, best, seed=seed,
                )
        except Exception as exc:
            logger.error("Final Dupire fit failed for %s: %s", ticker, exc,
                         exc_info=False)
            final = {'r2_score': float('nan'),
                     'sigma_recovered': float('nan'),
                     'drift_recovered': float('nan')}

        final['avg_market_iv'] = float(entry.get('avg_implied_vol',
                                                  float('nan')))
        out[ticker] = {
            'best_approach': best,
            'mean_errors': cv_res['mean_errors'],
            'per_fold_errors_df': cv_res['per_fold_errors_df'],
            'normalize_moneyness': bool(normalize_moneyness),
            'final': final,
            'applied_spy_winner': False,
        }

    # Apply SPY winner (or fallback) to AAPL/MSFT.
    fallback_approach = spy_winner or 'fd_baseline'
    for ticker in fallback_tickers:
        entry = per_ticker_results.get(ticker)
        if entry is None:
            continue
        try:
            C, K_grid, tau_grid = _extract_call_surface(entry)
            tau_uniform = np.linspace(float(tau_grid[0]),
                                      float(tau_grid[-1]), len(tau_grid))
            if not np.allclose(tau_grid, tau_uniform, rtol=1e-6, atol=1e-9):
                C_uni = np.empty_like(C)
                for i in range(C.shape[0]):
                    C_uni[i, :] = np.interp(tau_uniform, tau_grid, C[i, :])
                C = C_uni
                tau_grid = tau_uniform
            final = _final_dupire_with_approach(
                C, K_grid, tau_grid, fallback_approach, seed=seed,
            )
        except Exception as exc:
            logger.error("Fallback Dupire fit failed for %s: %s", ticker, exc,
                         exc_info=False)
            final = {'r2_score': float('nan'),
                     'sigma_recovered': float('nan'),
                     'drift_recovered': float('nan')}
        final['avg_market_iv'] = float(entry.get('avg_implied_vol',
                                                  float('nan')))
        out[ticker] = {
            'best_approach': fallback_approach,
            'mean_errors': {},
            'per_fold_errors_df': pd.DataFrame(),
            'normalize_moneyness': bool(normalize_moneyness),
            'final': final,
            'applied_spy_winner': True,
        }

    out['_meta'] = {
        'spy_winner': spy_winner,
        'normalize_moneyness': bool(normalize_moneyness),
        'seed': int(seed),
    }
    return out
