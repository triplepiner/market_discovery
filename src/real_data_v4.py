"""
Real-data pipeline v4: analytical theta target + direct Dupire formula.

The v2/v3 stack regressed dC/dtau (computed via finite differences across ~8
discrete expirations) onto a Dupire library. The FD theta target was the
dominant source of error: very few tau levels and non-uniform spacing produce
catastrophic noise, while the k-derivatives (computed on the SVI-smoothed
surface) are clean.

This module replaces FD theta with the analytical Black-Scholes theta computed
pointwise from each grid point's own implied volatility. SVI gives us a smooth
sigma_imp(k, tau) surface, so we can compute theta(k, tau) exactly via the
closed-form BS formula, sidestepping the FD differencing across expirations.

Fix 1 -- analytical BS theta target (bs_theta_analytical,
         dupire_2term_analytical_theta).
Fix 3 -- direct Dupire formula (direct_dupire_local_vol).
Fix 4 -- convenience wrapper that re-runs all v3 experiments with the
         analytical-theta target (run_v4_experiments_on_ticker).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.real_data_v2 import (
    _svi_total_variance,
    build_logm_surface_svi,
    compute_forward_prices,
    compute_liquidity_weights,
    compute_log_moneyness,
)
from src.real_data_v3 import (
    bootstrap_sigma_v2,
    compare_sigma_methods,
    per_expiration_sigma,
    windowed_2term_dupire_v2,
)
from src.utils import set_all_seeds, setup_logging

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r2_from(target: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    if ss_tot < 1e-30:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _central_dk(C: np.ndarray, k_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dk = float(k_grid[1] - k_grid[0])
    dCdk = np.gradient(C, dk, axis=0, edge_order=2)
    d2Cdk2 = np.gradient(dCdk, dk, axis=0, edge_order=2)
    return dCdk, d2Cdk2


def reconstruct_sigma_imp_grid(svi_params: list[dict[str, Any]],
                                k_grid: np.ndarray,
                                tau_grid: np.ndarray) -> np.ndarray:
    """Evaluate the per-tau SVI/quadratic fits on the regular (k, tau) grid.

    Parameters
    ----------
    svi_params : list of dicts
        Per-expiration fit dicts as stored by ``build_logm_surface_svi``.
        Each must contain ``a, b, rho, m, s`` (SVI raw form), ``tau``, and
        ``method`` (``'svi'`` or ``'quadratic'``). For ``'quadratic'``
        entries we fall back to a flat sigma = sqrt(a) when ``a`` is
        finite, otherwise a constant 0.20.
    k_grid : ndarray, shape (n_k,)
    tau_grid : ndarray, shape (n_tau,)
        Regular tau grid; the SVI fits are interpolated (nearest in tau) to
        cover the entire grid.

    Returns
    -------
    ndarray
        sigma_imp on the (n_k, n_tau) grid. Clipped to [0.01, 3.0].
    """
    k_grid = np.asarray(k_grid, dtype=np.float64)
    tau_grid = np.asarray(tau_grid, dtype=np.float64)
    n_k = len(k_grid)
    n_tau = len(tau_grid)

    if not svi_params:
        return np.full((n_k, n_tau), 0.20)

    fit_taus = np.array([float(p.get('tau', np.nan)) for p in svi_params])
    sigma_grid = np.zeros((n_k, n_tau), dtype=np.float64)

    for j, tau in enumerate(tau_grid):
        # Find nearest fitted slice in tau.
        diffs = np.abs(fit_taus - float(tau))
        if not np.all(np.isfinite(diffs)):
            diffs = np.where(np.isfinite(diffs), diffs, np.inf)
        idx = int(np.argmin(diffs))
        fit = svi_params[idx]
        tau_fit = float(fit.get('tau', tau))
        method = fit.get('method', 'svi')
        if method == 'svi' and np.isfinite(fit.get('a', np.nan)):
            params = np.array([fit['a'], fit['b'], fit['rho'], fit['m'],
                               fit['s']])
            try:
                w_dense = _svi_total_variance(params, k_grid)
                w_dense = np.clip(w_dense, 1e-8, None)
                sigma_slice = np.sqrt(w_dense / max(tau_fit, 1e-8))
            except Exception:
                sigma_slice = np.full(n_k, 0.20)
        else:
            # Quadratic fallback or anything else: use a flat estimate.
            a_val = fit.get('a', np.nan)
            if np.isfinite(a_val) and a_val > 0:
                sigma_slice = np.full(n_k, float(np.sqrt(max(a_val / max(tau_fit, 1e-8), 1e-4))))
            else:
                sigma_slice = np.full(n_k, 0.20)
        sigma_grid[:, j] = sigma_slice

    sigma_grid = np.clip(sigma_grid, 0.01, 3.0)
    return sigma_grid


# ---------------------------------------------------------------------------
# Fix 1 -- analytical BS theta target
# ---------------------------------------------------------------------------

def bs_theta_analytical(S0: float, K_grid: np.ndarray, tau_grid: np.ndarray,
                         sigma_imp_grid: np.ndarray, r: float,
                         q: float = 0.0) -> np.ndarray:
    """Analytical BS theta dC/dtau on a (k, tau) grid using sigma_imp.

    Formula (with continuous dividend yield q):

        dC/dtau = S0*exp(-q*tau) * phi(d1) * sigma_imp / (2*sqrt(tau))
                  + r * K * exp(-r*tau) * N(d2)
                  - q * S0 * exp(-q*tau) * N(d1)

    where d1 = [log(S0/K) + (r - q + 0.5*sigma^2)*tau] / (sigma*sqrt(tau))
          d2 = d1 - sigma*sqrt(tau).

    With q=0 this reduces to the formula in the PRD.

    Parameters
    ----------
    S0 : float
    K_grid : ndarray, shape (n_k,) OR (n_k, n_tau)
        Strikes at each grid point. If 1D it's broadcast along tau; if 2D it
        must be (n_k, n_tau) (e.g. when K(tau) = F(tau)*exp(k) varies with tau).
    tau_grid : ndarray, shape (n_tau,)
    sigma_imp_grid : ndarray, shape (n_k, n_tau)
    r : float
    q : float, default 0

    Returns
    -------
    ndarray
        theta of shape (n_k, n_tau).
    """
    K_grid = np.asarray(K_grid, dtype=np.float64)
    tau_grid = np.asarray(tau_grid, dtype=np.float64)
    sigma = np.asarray(sigma_imp_grid, dtype=np.float64)

    n_k, n_tau = sigma.shape
    tau_2d = np.tile(tau_grid.reshape(1, -1), (n_k, 1))
    if K_grid.ndim == 1:
        K_2d = np.tile(K_grid.reshape(-1, 1), (1, n_tau))
    else:
        K_2d = K_grid
        if K_2d.shape != (n_k, n_tau):
            raise ValueError(
                f"K_grid shape {K_2d.shape} != sigma shape {sigma.shape}"
            )

    safe_tau = np.where(tau_2d > 1e-8, tau_2d, 1e-8)
    safe_sigma = np.where(sigma > 1e-6, sigma, 1e-6)
    sqrt_tau = np.sqrt(safe_tau)

    d1 = (np.log(float(S0) / K_2d) + (float(r) - float(q)
                                       + 0.5 * safe_sigma ** 2) * safe_tau) \
         / (safe_sigma * sqrt_tau)
    d2 = d1 - safe_sigma * sqrt_tau

    phi_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)

    spot_term = float(S0) * np.exp(-float(q) * safe_tau)
    theta_vega = spot_term * phi_d1 * safe_sigma / (2.0 * sqrt_tau)
    theta_rate = float(r) * K_2d * np.exp(-float(r) * safe_tau) * N_d2
    theta_div = float(q) * spot_term * N_d1

    theta = theta_vega + theta_rate - theta_div
    # Mask invalid taus.
    theta = np.where(tau_2d > 1e-8, theta, 0.0)
    return theta


def dupire_2term_analytical_theta(C_surface: np.ndarray,
                                   sigma_imp_surface: np.ndarray,
                                   k_grid: np.ndarray, tau_grid: np.ndarray,
                                   S0: float, r: float, q: float = 0.0,
                                   weights: Optional[np.ndarray] = None,
                                   ) -> dict[str, Any]:
    """2-term Dupire OLS using analytical theta as the target.

    PDE in log-moneyness:

        theta = 0.5 * sigma_loc^2 * d2C/dk2 - (r - q - 0.5*sigma_loc^2) * dC/dk
              = c1 * dC/dk + c2 * d2C/dk2

    Procedure:
      1. Build K(tau) = F(tau) * exp(k) per column.
      2. theta = bs_theta_analytical(S0, K, tau, sigma_imp, r, q).
      3. dC/dk and d2C/dk2 via centered FD on C_surface.
      4. OLS solve for (c1, c2).
      5. sigma_loc = sqrt(2*c2) if c2 > 0.

    Returns
    -------
    dict
        ``r2_score``, ``sigma_loc_discovered``, ``coef_dCdk``,
        ``coef_d2Cdk2``, ``drift_discovered``, ``rq_implied``,
        ``condition_number``, ``theta_stats``.
    """
    C = np.asarray(C_surface, dtype=np.float64)
    sigma_imp = np.asarray(sigma_imp_surface, dtype=np.float64)
    k_grid = np.asarray(k_grid, dtype=np.float64)
    tau_grid = np.asarray(tau_grid, dtype=np.float64)
    n_k, n_tau = C.shape
    if sigma_imp.shape != C.shape:
        raise ValueError(
            f"sigma_imp shape {sigma_imp.shape} != C shape {C.shape}"
        )

    # K(tau) = F(tau) * exp(k).
    F_grid = compute_forward_prices(S0, r, q, tau_grid)
    K_grid_2d = np.outer(np.exp(k_grid), F_grid)  # (n_k, n_tau)

    theta = bs_theta_analytical(S0, K_grid_2d, tau_grid, sigma_imp, r, q)

    dCdk, d2Cdk2 = _central_dk(C, k_grid)

    library = np.column_stack([dCdk.ravel(), d2Cdk2.ravel()])
    target = theta.ravel()

    mask = (np.all(np.isfinite(library), axis=1) & np.isfinite(target))
    library_m = library[mask]
    target_m = target[mask]

    if weights is not None:
        w = np.clip(np.asarray(weights, dtype=np.float64).ravel()[mask],
                    0.0, None)
        sw = np.sqrt(w)
        A = library_m * sw[:, None]
        b = target_m * sw
    else:
        A = library_m
        b = target_m

    cond = float(np.linalg.cond(library_m)) if library_m.size > 0 else float('inf')
    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception as exc:
        logger.warning("dupire_2term_analytical_theta lstsq failed: %s", exc)
        coef = np.array([0.0, 0.0])

    c1 = float(coef[0])
    c2 = float(coef[1])
    pred = library_m @ coef
    r2 = _r2_from(target_m, pred)

    sigma_loc = float(np.sqrt(max(0.0, 2.0 * c2)))
    drift = c1
    rq_implied = -drift - 0.5 * sigma_loc * sigma_loc

    theta_finite = theta[np.isfinite(theta)]
    if theta_finite.size > 0:
        theta_stats = {
            'min': float(theta_finite.min()),
            'max': float(theta_finite.max()),
            'mean': float(theta_finite.mean()),
            'std': float(theta_finite.std()),
        }
    else:
        theta_stats = {'min': float('nan'), 'max': float('nan'),
                       'mean': float('nan'), 'std': float('nan')}

    return {
        'coef_dCdk': c1,
        'coef_d2Cdk2': c2,
        'sigma_loc_discovered': sigma_loc,
        'drift_discovered': drift,
        'rq_implied': float(rq_implied),
        'r2_score': float(r2),
        'condition_number': cond,
        'theta_stats': theta_stats,
    }


def quadratic_dupire_analytical_theta(C_surface: np.ndarray,
                                       sigma_imp_surface: np.ndarray,
                                       k_grid: np.ndarray, tau_grid: np.ndarray,
                                       S0: float, r: float, q: float = 0.0,
                                       weights: Optional[np.ndarray] = None,
                                       k_eval: Optional[list[float]] = None,
                                       ) -> dict[str, Any]:
    """Quadratic sigma^2(k) Dupire using analytical theta target.

    sigma^2(k) = alpha + beta*k + gamma*k^2.
    Library: [dC/dk, d2C/dk2, k*d2C/dk2, k^2*d2C/dk2]; target = theta.
    """
    C = np.asarray(C_surface, dtype=np.float64)
    sigma_imp = np.asarray(sigma_imp_surface, dtype=np.float64)
    k_grid = np.asarray(k_grid, dtype=np.float64)
    tau_grid = np.asarray(tau_grid, dtype=np.float64)
    n_k, n_tau = C.shape

    F_grid = compute_forward_prices(S0, r, q, tau_grid)
    K_grid_2d = np.outer(np.exp(k_grid), F_grid)
    theta = bs_theta_analytical(S0, K_grid_2d, tau_grid, sigma_imp, r, q)

    dCdk, d2Cdk2 = _central_dk(C, k_grid)
    KK = np.tile(k_grid.reshape(-1, 1), (1, n_tau))

    lib = np.column_stack([
        dCdk.ravel(),
        d2Cdk2.ravel(),
        (KK * d2Cdk2).ravel(),
        (KK * KK * d2Cdk2).ravel(),
    ])
    tgt = theta.ravel()
    mask = np.all(np.isfinite(lib), axis=1) & np.isfinite(tgt)
    lib_m = lib[mask]
    tgt_m = tgt[mask]

    if weights is not None:
        w = np.clip(np.asarray(weights, dtype=np.float64).ravel()[mask],
                    0.0, None)
        sw = np.sqrt(w)
        A = lib_m * sw[:, None]
        b = tgt_m * sw
    else:
        A = lib_m
        b = tgt_m

    cond = float(np.linalg.cond(lib_m)) if lib_m.size > 0 else float('inf')
    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        coef = np.zeros(4)
    drift = float(coef[0])
    alpha = 2.0 * float(coef[1])
    beta = 2.0 * float(coef[2])
    gamma = 2.0 * float(coef[3])

    pred = lib_m @ coef
    r2 = _r2_from(tgt_m, pred)

    if k_eval is None:
        k_eval = [-0.2, -0.1, 0.0, 0.1, 0.2]
    sigma_at_k: dict[float, float] = {}
    for kv in k_eval:
        sig2 = alpha + beta * kv + gamma * kv * kv
        sigma_at_k[float(kv)] = float(np.sqrt(sig2)) if sig2 > 0 else float('nan')

    return {
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'drift': drift,
        'r2_score': float(r2),
        'sigma_at_k_dict': sigma_at_k,
        'condition_number': cond,
    }


# ---------------------------------------------------------------------------
# Fix 3 -- Direct Dupire formula
# ---------------------------------------------------------------------------

def direct_dupire_local_vol(C_surface: np.ndarray,
                             sigma_imp_surface: np.ndarray,
                             k_grid: np.ndarray, tau_grid: np.ndarray,
                             S0: float, r: float,
                             q: float = 0.0) -> dict[str, Any]:
    """Direct Dupire formula for local volatility on the grid.

    sigma^2_loc(K, tau) = 2 * (dC/dtau + (r-q)*K*dC/dK + q*C)
                          / (K^2 * d2C/dK^2)

    Uses analytical theta for dC/dtau. The strike derivatives are computed
    from log-moneyness FD via the chain rule:

        dC/dK = (1/K) * dC/dk
        d2C/dK2 = (1/K^2) * (d2C/dk2 - dC/dk)

    Returns
    -------
    dict
        ``sigma_loc_grid`` 2D array (NaN where invalid), ``n_valid``,
        ``n_total``, ``n_valid_pct``, ``sigma_loc_median``,
        ``sigma_loc_mean``, ``k_grid``, ``tau_grid``.
    """
    C = np.asarray(C_surface, dtype=np.float64)
    sigma_imp = np.asarray(sigma_imp_surface, dtype=np.float64)
    k_grid = np.asarray(k_grid, dtype=np.float64)
    tau_grid = np.asarray(tau_grid, dtype=np.float64)
    n_k, n_tau = C.shape

    F_grid = compute_forward_prices(S0, r, q, tau_grid)
    K_2d = np.outer(np.exp(k_grid), F_grid)  # (n_k, n_tau)

    theta = bs_theta_analytical(S0, K_2d, tau_grid, sigma_imp, r, q)
    dCdk, d2Cdk2 = _central_dk(C, k_grid)

    # Chain rule -> strike derivatives.
    with np.errstate(divide='ignore', invalid='ignore'):
        dCdK = dCdk / K_2d
        d2CdK2 = (d2Cdk2 - dCdk) / (K_2d * K_2d)

    numerator = theta + (float(r) - float(q)) * K_2d * dCdK + float(q) * C
    denominator = K_2d * K_2d * d2CdK2

    with np.errstate(divide='ignore', invalid='ignore'):
        sigma2 = 2.0 * numerator / denominator

    sigma2 = np.where(np.isfinite(sigma2), sigma2, np.nan)
    sigma2 = np.where(sigma2 > 0, sigma2, np.nan)
    sigma_loc = np.sqrt(sigma2)

    valid = np.isfinite(sigma_loc)
    n_valid = int(valid.sum())
    n_total = int(sigma_loc.size)
    if n_valid > 0:
        sigma_median = float(np.nanmedian(sigma_loc))
        sigma_mean = float(np.nanmean(sigma_loc))
    else:
        sigma_median = float('nan')
        sigma_mean = float('nan')

    return {
        'sigma_loc_grid': sigma_loc,
        'n_valid': n_valid,
        'n_total': n_total,
        'n_valid_pct': float(n_valid) / float(n_total) if n_total > 0 else 0.0,
        'sigma_loc_median': sigma_median,
        'sigma_loc_mean': sigma_mean,
        'k_grid': k_grid,
        'tau_grid': tau_grid,
    }


# ---------------------------------------------------------------------------
# Fix 4 -- bootstrap with analytical theta
# ---------------------------------------------------------------------------

def bootstrap_sigma_analytical(option_data: dict[str, Any], ticker: str,
                                n_bootstrap: int = 50, seed: int = 42,
                                q: Optional[float] = None,
                                n_k: int = 40,
                                k_range: tuple[float, float] = (-0.25, 0.25),
                                ) -> dict[str, Any]:
    """Resample the chain, rebuild SVI, refit Dupire with *analytical theta*."""
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)
    df = option_data['option_df'].copy().reset_index(drop=True)
    S0 = float(option_data['S0'])
    r = float(option_data['r'])
    if q is None:
        try:
            from src.real_data_v2 import get_dividend_yield
            q = get_dividend_yield(ticker)
        except Exception:
            q = 0.0

    n_rows = len(df)
    sigmas: list[float] = []
    for b in range(int(n_bootstrap)):
        try:
            idx = rng.integers(0, n_rows, size=n_rows)
            df_b = df.iloc[idx].reset_index(drop=True)
            surface = build_logm_surface_svi(df_b, S0, r, q, n_k=n_k,
                                             k_range=k_range)
            sigma_imp = reconstruct_sigma_imp_grid(
                surface['svi_params'], surface['k_grid'], surface['tau_grid'],
            )
            res = dupire_2term_analytical_theta(
                surface['C_surface'], sigma_imp,
                surface['k_grid'], surface['tau_grid'], S0, r, q,
            )
            sigma_b = float(res.get('sigma_loc_discovered', float('nan')))
            if np.isfinite(sigma_b) and sigma_b > 0:
                sigmas.append(sigma_b)
        except Exception as exc:
            logger.debug("bootstrap analytical iter %d failed: %s", b, exc)
            continue

    n_success = len(sigmas)
    if n_success == 0:
        return {
            'sigma_mean': float('nan'), 'sigma_std': float('nan'),
            'ci_low': float('nan'), 'ci_high': float('nan'),
            'n_success': 0, 'n_total': int(n_bootstrap), 'sigmas': [],
        }
    arr = np.asarray(sigmas, dtype=np.float64)
    return {
        'sigma_mean': float(np.mean(arr)),
        'sigma_std': float(np.std(arr, ddof=1)) if n_success > 1 else 0.0,
        'ci_low': float(np.percentile(arr, 2.5)),
        'ci_high': float(np.percentile(arr, 97.5)),
        'n_success': n_success,
        'n_total': int(n_bootstrap),
        'sigmas': sigmas,
    }


def run_v4_experiments_on_ticker(option_data: dict[str, Any], ticker: str,
                                  n_bootstrap: int = 50, seed: int = 42,
                                  n_k: int = 40,
                                  k_range: tuple[float, float] = (-0.25, 0.25),
                                  ) -> dict[str, Any]:
    """Full re-run of v3 experiments using the analytical-theta target.

    Returns
    -------
    dict
        Keys: ``ticker``, ``q``, ``svi_stats``, ``global_2term``,
        ``quadratic``, ``windowed``, ``bootstrap``, ``sigma_comparison``,
        ``direct_dupire``, ``errors``.
    """
    set_all_seeds(seed)
    out: dict[str, Any] = {'ticker': ticker, 'errors': {}}

    try:
        S0 = float(option_data['S0'])
        r = float(option_data['r'])
        df = option_data['option_df'].copy()
    except Exception as exc:
        out['errors']['top'] = str(exc)
        return out

    try:
        from src.real_data_v2 import get_dividend_yield
        q = get_dividend_yield(ticker)
    except Exception as exc:
        out['errors']['q'] = str(exc)
        q = 0.0
    out['q'] = float(q)

    # SVI surface + sigma_imp grid.
    try:
        surface = build_logm_surface_svi(df, S0, r, q, n_k=n_k, k_range=k_range)
        C_surface = surface['C_surface']
        k_grid = surface['k_grid']
        tau_grid = surface['tau_grid']
        sigma_imp = reconstruct_sigma_imp_grid(
            surface['svi_params'], k_grid, tau_grid,
        )
        out['svi_stats'] = {
            'n_success': int(surface.get('n_svi_success', 0)),
            'n_total': int(surface.get('n_svi_success', 0)
                           + surface.get('n_svi_fallback', 0)),
        }
    except Exception as exc:
        logger.warning("%s: surface build failed: %s", ticker, exc)
        out['errors']['surface'] = str(exc)
        return out

    # Liquidity weights.
    weights = None
    try:
        weights = compute_liquidity_weights(df, k_grid, tau_grid, q=q, S0=S0)
    except Exception as exc:
        out['errors']['weights'] = str(exc)
        weights = None

    # Market reference IV (ATM mean).
    try:
        F_obs = float(S0) * np.exp((r - q) * df['tau'].values)
        k_obs = compute_log_moneyness(df['strike'].values, F_obs)
        atm_mask = np.abs(k_obs) < 0.05
        ivs = df['implied_vol'].values
        m = atm_mask & np.isfinite(ivs) & (ivs > 0)
        atm_iv = float(np.mean(ivs[m])) if m.sum() > 0 else float('nan')
    except Exception:
        atm_iv = float('nan')
    out['atm_iv_market'] = atm_iv

    # Global 2-term Dupire with analytical theta.
    try:
        global_2term = dupire_2term_analytical_theta(
            C_surface, sigma_imp, k_grid, tau_grid, S0, r, q, weights=weights,
        )
        if np.isfinite(atm_iv) and atm_iv > 0:
            global_2term['sigma_rel_err'] = float(
                abs(global_2term['sigma_loc_discovered'] - atm_iv) / atm_iv
            )
        else:
            global_2term['sigma_rel_err'] = float('nan')
        out['global_2term'] = global_2term
    except Exception as exc:
        logger.warning("%s: global 2-term failed: %s", ticker, exc)
        out['errors']['global_2term'] = str(exc)
        out['global_2term'] = None

    # Quadratic Dupire with analytical theta.
    try:
        quad = quadratic_dupire_analytical_theta(
            C_surface, sigma_imp, k_grid, tau_grid, S0, r, q, weights=weights,
        )
        quad['atm_iv_market'] = atm_iv
        sig0 = quad['sigma_at_k_dict'].get(0.0, float('nan'))
        if np.isfinite(sig0) and np.isfinite(atm_iv) and atm_iv > 0:
            quad['sigma_at_zero_rel_err'] = float(abs(sig0 - atm_iv) / atm_iv)
        else:
            quad['sigma_at_zero_rel_err'] = float('nan')
        out['quadratic'] = quad
    except Exception as exc:
        logger.warning("%s: quadratic failed: %s", ticker, exc)
        out['errors']['quadratic'] = str(exc)
        out['quadratic'] = None

    # Windowed Dupire reuses the existing FD-theta windowed code (still useful
    # for term-structure summaries; the v3 implementation does its own OLS).
    try:
        window_size = min(8, len(k_grid), len(tau_grid))
        if window_size >= 4:
            win = windowed_2term_dupire_v2(
                C_surface, k_grid, tau_grid,
                window_size=window_size, stride=max(1, window_size // 3),
                min_r2=0.5, sigma_bounds=(0.01, 2.0), weights=weights,
            )
            out['windowed'] = {
                'n_valid': int(win['n_valid']),
                'n_total': int(win['n_total']),
                'sigma_median': float(win['sigma_median']),
                'sigma_mean': float(win['sigma_mean']),
            }
        else:
            out['windowed'] = None
    except Exception as exc:
        logger.warning("%s: windowed failed: %s", ticker, exc)
        out['errors']['windowed'] = str(exc)
        out['windowed'] = None

    # Bootstrap CI (analytical theta).
    try:
        bs = bootstrap_sigma_analytical(
            option_data, ticker, n_bootstrap=n_bootstrap, seed=seed,
            q=q, n_k=n_k, k_range=k_range,
        )
        # Check whether the point estimate sits inside the CI.
        point_est = float('nan')
        if out.get('global_2term') is not None:
            point_est = float(out['global_2term'].get('sigma_loc_discovered',
                                                       float('nan')))
        if np.isfinite(point_est) and np.isfinite(bs['ci_low']) \
                and np.isfinite(bs['ci_high']):
            in_ci = bool(bs['ci_low'] <= point_est <= bs['ci_high'])
        else:
            in_ci = False
        bs['ci_width'] = float(bs['ci_high'] - bs['ci_low']) \
            if np.isfinite(bs['ci_high']) and np.isfinite(bs['ci_low']) \
            else float('nan')
        bs['point_in_ci'] = in_ci
        bs['point_estimate'] = point_est
        out['bootstrap'] = bs
    except Exception as exc:
        logger.warning("%s: bootstrap failed: %s", ticker, exc)
        out['errors']['bootstrap'] = str(exc)
        out['bootstrap'] = None

    # Sigma method comparison (sindy = analytical-theta global 2-term).
    try:
        sigma_sindy = float('nan')
        if out.get('global_2term') is not None:
            sigma_sindy = float(out['global_2term'].get(
                'sigma_loc_discovered', float('nan')))
        cmp_df = compare_sigma_methods(option_data, ticker,
                                        sigma_sindy=sigma_sindy, q=q)
        out['sigma_comparison'] = cmp_df
    except Exception as exc:
        logger.warning("%s: sigma comparison failed: %s", ticker, exc)
        out['errors']['sigma_comparison'] = str(exc)
        out['sigma_comparison'] = None

    # Direct Dupire local-vol grid.
    try:
        direct = direct_dupire_local_vol(
            C_surface, sigma_imp, k_grid, tau_grid, S0, r, q,
        )
        out['direct_dupire'] = {
            'n_valid': int(direct['n_valid']),
            'n_total': int(direct['n_total']),
            'n_valid_pct': float(direct['n_valid_pct']),
            'sigma_loc_median': float(direct['sigma_loc_median']),
            'sigma_loc_mean': float(direct['sigma_loc_mean']),
            'sigma_loc_grid': direct['sigma_loc_grid'],
        }
    except Exception as exc:
        logger.warning("%s: direct Dupire failed: %s", ticker, exc)
        out['errors']['direct_dupire'] = str(exc)
        out['direct_dupire'] = None

    return out


def run_v4_all_tickers(per_ticker_results: dict[str, Any],
                        **kwargs: Any) -> dict[str, Any]:
    """Run :func:`run_v4_experiments_on_ticker` over a dict of ticker results."""
    per_ticker: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for ticker, entry in per_ticker_results.items():
        if isinstance(entry, dict) and 'option_data' in entry:
            option_data = entry['option_data']
        else:
            option_data = entry
        try:
            res = run_v4_experiments_on_ticker(option_data, ticker, **kwargs)
            per_ticker[ticker] = res
            g2 = res.get('global_2term') or {}
            quad = res.get('quadratic') or {}
            win = res.get('windowed') or {}
            bs = res.get('bootstrap') or {}
            direct = res.get('direct_dupire') or {}
            rows.append({
                'ticker': ticker,
                'q': res.get('q', float('nan')),
                'g2_r2': g2.get('r2_score', float('nan')),
                'g2_sigma': g2.get('sigma_loc_discovered', float('nan')),
                'g2_rq': g2.get('rq_implied', float('nan')),
                'quad_r2': quad.get('r2_score', float('nan')),
                'quad_alpha': quad.get('alpha', float('nan')),
                'quad_beta': quad.get('beta', float('nan')),
                'quad_gamma': quad.get('gamma', float('nan')),
                'win_valid': win.get('n_valid', 0),
                'win_total': win.get('n_total', 0),
                'bs_ci_low': bs.get('ci_low', float('nan')),
                'bs_ci_high': bs.get('ci_high', float('nan')),
                'bs_ci_width': bs.get('ci_width', float('nan')),
                'bs_point_in_ci': bs.get('point_in_ci', False),
                'direct_valid_pct': direct.get('n_valid_pct', float('nan')),
                'direct_sigma_median': direct.get('sigma_loc_median',
                                                   float('nan')),
            })
        except Exception as exc:
            logger.error("Ticker %s v4 failed: %s", ticker, exc, exc_info=True)
            per_ticker[ticker] = {'error': str(exc)}
            rows.append({'ticker': ticker, 'error': str(exc)})

    summary_df = pd.DataFrame(rows)
    return {'per_ticker': per_ticker, 'summary_df': summary_df}
