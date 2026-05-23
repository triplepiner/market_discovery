"""
Dupire PDE discovery via SINDy.

Real option chains are observed as C(K, tau) -- call price as a function of
strike and time to maturity -- which evolve according to the Dupire equation:

    dC/dT = 0.5 * sigma_local^2 * K^2 * d2C/dK2
            - (r - q) * K * dC/dK
            - q * C

For the call-only, no-dividend case (q = 0):

    dC/dT = 0.5 * sigma^2 * K^2 * d2C/dK2 - r * K * dC/dK

This module builds a 5-term library and runs STLSQ sparse regression to
recover those coefficients from a (K, tau) price surface.
"""

import numpy as np
import pandas as pd

from src.utils import (
    set_all_seeds,
    setup_logging,
    NumericalDifferentiator,
)
from src.data_generation import bs_call_price
from src.sindy_discovery import stlsq_sweep, format_pde_string

logger = setup_logging(__name__)

DUPIRE_TERM_NAMES = [
    'C',
    'dC/dK',
    'd2C/dK2',
    'K*dC/dK',
    'K2*d2C/dK2',
]


def _smooth_surface(C, savgol_window, savgol_poly):
    """Apply Savitzky-Golay smoothing along both axes of a 2-D surface."""
    from scipy.signal import savgol_filter

    C_s = C.astype(np.float64, copy=True)
    for axis in (0, 1):
        n = C_s.shape[axis]
        win = savgol_window
        if win > n:
            win = n if n % 2 == 1 else n - 1
        if win < savgol_poly + 2:
            continue
        try:
            C_s = savgol_filter(C_s, window_length=win,
                                polyorder=savgol_poly, axis=axis)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "savgol smoothing failed on axis %d: %s", axis, exc
            )
    return C_s


def build_dupire_library(C, K_grid, tau_grid, smooth=True,
                         savgol_window=11, savgol_poly=3):
    """Build the 5-term Dupire SINDy library and the dC/dT target.

    Parameters
    ----------
    C : ndarray, shape (n_K, n_tau)
        Call price surface, with strike along axis 0 and time-to-maturity
        along axis 1.
    K_grid : ndarray, shape (n_K,)
        Uniformly spaced strike grid.
    tau_grid : ndarray, shape (n_tau,)
        Uniformly spaced maturity grid.
    smooth : bool, default True
        Apply Savitzky-Golay smoothing to ``C`` before differentiating.
    savgol_window : int, default 11
        Window length for the Savitzky-Golay filter (must be odd).
    savgol_poly : int, default 3
        Polynomial order for the Savitzky-Golay filter.

    Returns
    -------
    target : ndarray, shape (n_points,)
        Flattened dC/dT values.
    library : ndarray, shape (n_points, 5)
        Columns: [C, dC/dK, d2C/dK2, K*dC/dK, K^2*d2C/dK2].
    term_names : list of str
        Human-readable column names.
    """
    C = np.asarray(C, dtype=np.float64)
    K_grid = np.asarray(K_grid, dtype=np.float64)
    tau_grid = np.asarray(tau_grid, dtype=np.float64)

    if C.ndim != 2:
        raise ValueError(f"C must be 2-D (n_K, n_tau); got shape {C.shape}")
    if C.shape != (len(K_grid), len(tau_grid)):
        raise ValueError(
            f"C shape {C.shape} does not match "
            f"(len(K_grid)={len(K_grid)}, len(tau_grid)={len(tau_grid)})"
        )
    if len(K_grid) < 3 or len(tau_grid) < 3:
        raise ValueError(
            "Need at least 3 points along each axis to differentiate."
        )

    dK = float(K_grid[1] - K_grid[0])
    dtau = float(tau_grid[1] - tau_grid[0])
    if dK <= 0 or dtau <= 0:
        raise ValueError("K_grid and tau_grid must be strictly increasing.")

    # Optional smoothing
    if smooth:
        try:
            C_work = _smooth_surface(C, savgol_window, savgol_poly)
        except Exception as exc:
            logger.warning(
                "Smoothing failed (%s); falling back to raw surface.", exc
            )
            C_work = C
    else:
        C_work = C

    # Derivatives via a NumericalDifferentiator without further smoothing
    diff = NumericalDifferentiator(order=2, smooth=False)
    dCdtau = diff.first_derivative(C_work, dtau, axis=1)  # dC/dT
    dCdK = diff.first_derivative(C_work, dK, axis=0)
    d2CdK2 = diff.second_derivative(C_work, dK, axis=0)

    # Broadcast K across the tau dimension
    K_mesh = np.broadcast_to(K_grid[:, None], C_work.shape)

    library = np.column_stack([
        C_work.ravel(),
        dCdK.ravel(),
        d2CdK2.ravel(),
        (K_mesh * dCdK).ravel(),
        (K_mesh ** 2 * d2CdK2).ravel(),
    ]).astype(np.float64)

    target = dCdtau.ravel().astype(np.float64)

    try:
        cond = float(np.linalg.cond(library))
        logger.info("Dupire library condition number: %.3e", cond)
        if cond > 1e10:
            logger.warning(
                "Dupire library is ill-conditioned (cond=%.2e); "
                "results may be unreliable.", cond
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Condition number check failed: %s", exc)

    return target, library, list(DUPIRE_TERM_NAMES)


def _format_dupire_pde(coefficients, threshold=1e-10):
    """Format Dupire-side PDE as ``dC/dT = ...``."""
    return format_pde_string(
        coefficients, term_names=DUPIRE_TERM_NAMES, threshold=threshold
    ).replace('dV/dt', 'dC/dT')


def discover_dupire(C, K_grid, tau_grid, smooth=True,
                    savgol_window=11, savgol_poly=3):
    """Run STLSQ sparse regression to recover the Dupire PDE.

    Parameters
    ----------
    C : ndarray, shape (n_K, n_tau)
    K_grid : ndarray, shape (n_K,)
    tau_grid : ndarray, shape (n_tau,)
    smooth : bool, default True
    savgol_window : int, default 11
    savgol_poly : int, default 3

    Returns
    -------
    dict
        Keys: ``discovered_coefficients``, ``active_terms``, ``r2_score``,
        ``term_names``, ``condition_number``, ``human_readable_pde``,
        ``sigma_discovered``, ``drift_discovered``, ``best_threshold``,
        ``active_mask``, ``n_active``, ``sweep_results``.
    """
    set_all_seeds(42)

    target, library, term_names = build_dupire_library(
        C, K_grid, tau_grid,
        smooth=smooth, savgol_window=savgol_window, savgol_poly=savgol_poly,
    )

    cond_number = float(np.linalg.cond(library))

    try:
        best, sweep_results = stlsq_sweep(library, target)
    except Exception as exc:
        logger.warning("STLSQ sweep failed: %s; returning zero result.", exc)
        zero = np.zeros(library.shape[1], dtype=np.float64)
        best = {
            'coefficients': zero,
            'active_mask': np.zeros(library.shape[1], dtype=bool),
            'threshold': 0.0,
            'r2': 0.0,
            'bic': float('inf'),
            'n_active': 0,
        }
        sweep_results = []

    discovered = np.asarray(best['coefficients'], dtype=np.float64)
    active_mask = np.asarray(best['active_mask'], dtype=bool)
    active_terms = [term_names[i] for i in range(len(term_names))
                    if active_mask[i]]

    # Derive financial parameters from the structural coefficients.
    coeff_K2 = float(discovered[4])
    coeff_K = float(discovered[3])

    if coeff_K2 > 0:
        sigma_discovered = float(np.sqrt(2.0 * coeff_K2))
    else:
        if coeff_K2 < 0:
            logger.warning(
                "Discovered K^2 d2C/dK2 coefficient is negative (%.4e); "
                "sigma cannot be recovered.", coeff_K2
            )
        sigma_discovered = float('nan')

    # Convention: target = dC/dT, true coefficient on K*dC/dK is -(r-q),
    # so r - q = -coeff_K.
    drift_discovered = float(-coeff_K)

    pde_str = _format_dupire_pde(discovered)

    logger.info(
        "Dupire discovery: R^2=%.6f, n_active=%d, sigma=%.4f, r-q=%.4f",
        best['r2'], best['n_active'], sigma_discovered, drift_discovered,
    )

    return {
        'discovered_coefficients': discovered,
        'active_terms': active_terms,
        'r2_score': float(best['r2']),
        'term_names': list(term_names),
        'condition_number': cond_number,
        'human_readable_pde': pde_str,
        'sigma_discovered': sigma_discovered,
        'drift_discovered': drift_discovered,
        'best_threshold': float(best['threshold']),
        'active_mask': active_mask,
        'n_active': int(best['n_active']),
        'bic': float(best['bic']),
        'sweep_results': sweep_results,
    }


def dupire_sanity_check(K_min=70, K_max=130, n_K=80,
                        tau_min=0.05, tau_max=1.5, n_tau=80,
                        S0=100, r=0.05, sigma=0.20, seed=42):
    """Validate the Dupire pipeline on an analytical BS call surface.

    Generates a clean call surface from ``bs_call_price``, runs the Dupire
    SINDy pipeline, and returns the discovery result.  The pipeline should
    recover ``R^2 > 0.99`` and a sigma within 5 percent of the input.

    Returns
    -------
    dict
        Same shape as :func:`discover_dupire`, plus the ``K_grid``,
        ``tau_grid`` and ``C`` used.
    """
    set_all_seeds(seed)

    K_grid = np.linspace(float(K_min), float(K_max), int(n_K), dtype=np.float64)
    tau_grid = np.linspace(
        float(tau_min), float(tau_max), int(n_tau), dtype=np.float64,
    )

    # Build the surface C(K, tau)
    C = np.zeros((len(K_grid), len(tau_grid)), dtype=np.float64)
    for j, tau in enumerate(tau_grid):
        C[:, j] = bs_call_price(float(S0), K_grid, float(r), float(sigma),
                                float(tau))

    result = discover_dupire(C, K_grid, tau_grid, smooth=True)
    result['K_grid'] = K_grid
    result['tau_grid'] = tau_grid
    result['C'] = C

    r2 = result['r2_score']
    sigma_err = abs(result['sigma_discovered'] - sigma) / sigma
    if r2 < 0.99 or sigma_err > 0.05:
        logger.warning(
            "Dupire sanity check did not meet target: "
            "R^2=%.6f, sigma_discovered=%.6f (err=%.4f)",
            r2, result['sigma_discovered'], sigma_err,
        )
    else:
        logger.info(
            "Dupire sanity check PASSED: R^2=%.6f, sigma=%.6f (err=%.4f)",
            r2, result['sigma_discovered'], sigma_err,
        )

    return result


def _extract_call_surface(ticker_result):
    """Pull (C, K_grid, tau_grid) from a per-ticker result dict.

    The current pipeline stores the call price surface under
    ``surface_data['V_surface']`` (calls are the default).  We also look
    for an explicit ``V_surface_call`` key, in case a future change adds a
    call/put split.
    """
    surface = ticker_result.get('surface_data')
    if surface is None:
        raise KeyError("ticker_result missing 'surface_data'")

    if 'V_surface_call' in surface:
        C = surface['V_surface_call']
    elif 'V_surface' in surface:
        C = surface['V_surface']
    else:
        raise KeyError(
            "surface_data has neither 'V_surface_call' nor 'V_surface'"
        )

    K_grid = surface['K_grid']
    tau_grid = surface['tau_grid']
    return (
        np.asarray(C, dtype=np.float64),
        np.asarray(K_grid, dtype=np.float64),
        np.asarray(tau_grid, dtype=np.float64),
    )


def run_dupire_on_real_data(ticker_results, tickers=None):
    """Run Dupire SINDy on each ticker's call surface.

    Parameters
    ----------
    ticker_results : dict
        The ``per_ticker_results`` mapping from
        :func:`src.real_data.run_real_data_experiment`.
    tickers : list of str or None
        Subset of tickers to process.  Defaults to all keys.

    Returns
    -------
    dict
        Mapping ``ticker -> dupire_result``.  Tickers that fail are mapped
        to ``{'error': str}`` so callers can keep going.
    """
    if tickers is None:
        tickers = list(ticker_results.keys())

    out = {}
    for ticker in tickers:
        if ticker not in ticker_results:
            logger.warning("Ticker %s not present in results; skipping.", ticker)
            continue
        try:
            C, K_grid, tau_grid = _extract_call_surface(ticker_results[ticker])
            # tau must be uniformly spaced for the FD derivatives; if not,
            # resample onto a uniform grid.
            tau_uniform = np.linspace(
                float(tau_grid[0]), float(tau_grid[-1]), len(tau_grid),
            )
            if not np.allclose(tau_grid, tau_uniform, rtol=1e-6, atol=1e-9):
                logger.info(
                    "Resampling %s tau grid onto a uniform spacing.", ticker,
                )
                C_uniform = np.empty_like(C)
                for i in range(C.shape[0]):
                    C_uniform[i, :] = np.interp(tau_uniform, tau_grid, C[i, :])
                C = C_uniform
                tau_grid = tau_uniform

            result = discover_dupire(C, K_grid, tau_grid, smooth=True)
            result['ticker'] = ticker
            out[ticker] = result
        except Exception as exc:
            logger.error(
                "Dupire discovery failed for %s: %s", ticker, exc,
                exc_info=True,
            )
            out[ticker] = {'error': str(ticker), 'message': str(exc)}
    return out


def compare_bs_vs_dupire_on_real_data(bs_real_results, dupire_real_results):
    """Compare BS-in-(S,t) vs Dupire-in-(K,tau) discovery across tickers.

    Parameters
    ----------
    bs_real_results : dict
        ``per_ticker_results`` from the existing BS pipeline.  Each value
        carries a ``sindy_result`` dict with ``r2_score``, plus an
        ``avg_implied_vol`` field.
    dupire_real_results : dict
        Output of :func:`run_dupire_on_real_data`.

    Returns
    -------
    pandas.DataFrame
        One row per ticker present in both dicts.
    """
    rows = []
    tickers = sorted(set(bs_real_results.keys()) & set(dupire_real_results.keys()))
    for ticker in tickers:
        bs_entry = bs_real_results.get(ticker, {})
        dup_entry = dupire_real_results.get(ticker, {})

        bs_r2 = float('nan')
        sindy = bs_entry.get('sindy_result') if isinstance(bs_entry, dict) else None
        if isinstance(sindy, dict):
            bs_r2 = float(sindy.get('r2_score', float('nan')))

        avg_iv = float('nan')
        if isinstance(bs_entry, dict):
            avg_iv = float(bs_entry.get('avg_implied_vol', float('nan')))

        if 'error' in dup_entry:
            dup_r2 = float('nan')
            sigma_disc = float('nan')
            drift_disc = float('nan')
        else:
            dup_r2 = float(dup_entry.get('r2_score', float('nan')))
            sigma_disc = float(dup_entry.get('sigma_discovered', float('nan')))
            drift_disc = float(dup_entry.get('drift_discovered', float('nan')))

        sigma_err = float('nan')
        if np.isfinite(sigma_disc) and np.isfinite(avg_iv) and avg_iv > 0:
            sigma_err = abs(sigma_disc - avg_iv) / avg_iv

        rows.append({
            'ticker': ticker,
            'r2_bs': bs_r2,
            'r2_dupire': dup_r2,
            'sigma_discovered_dupire': sigma_disc,
            'avg_market_iv': avg_iv,
            'sigma_rel_err': sigma_err,
            'drift_discovered_dupire': drift_disc,
        })

    return pd.DataFrame(rows)
