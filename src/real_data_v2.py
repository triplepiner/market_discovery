"""
Real-data pipeline v2: log-moneyness Dupire, SVI smoothing, liquidity weights.

This module re-frames PDE discovery on real option data in the coordinate system
where the Dupire equation reduces to two terms:

    k = log(K / F),  F = S0 * exp((r - q) * tau)

    dC/dtau = 0.5 * sigma_loc^2 * d2C/dk2
              - (r - q - 0.5 * sigma_loc^2) * dC/dk

This is a "2-term library" that should be far better conditioned than the
5-term (K, tau) library and recover sigma_loc directly from a clean OLS.

Pipeline pieces:
    Fix 1: log-moneyness coordinates (compute_log_moneyness, build_logm_surface)
    Fix 2: liquidity-weighted STLSQ (compute_liquidity_weights, weighted_stlsq)
    Fix 3: SVI smoothing of IV slices (fit_svi_slice, build_logm_surface_svi)
    Fix 4: 2-term Dupire regression (dupire_logm_2term, _windowed variant)
    Fix 5: forward prices (compute_forward_prices, get_dividend_yield)
    Fix 6: ATM filter (filter_atm)
    Fix 7: combined pipeline (run_improved_real_pipeline, _all_tickers)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import minimize, brentq
from scipy.stats import norm

from src.data_generation import bs_call_price
from src.sindy_discovery import stlsq
from src.utils import set_all_seeds, setup_logging

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Fix 5: Forward prices and dividend yields
# ---------------------------------------------------------------------------

KNOWN_DIVIDEND_YIELDS: dict[str, float] = {
    'SPY': 0.013,
    'QQQ': 0.005,
    'AAPL': 0.005,
    'MSFT': 0.007,
}


def compute_forward_prices(S0: float, r: float, q: float,
                           tau_grid: np.ndarray) -> np.ndarray:
    """Compute forward prices ``F(tau) = S0 * exp((r - q) * tau)``.

    Parameters
    ----------
    S0 : float
        Current spot price.
    r : float
        Risk-free rate (continuously compounded).
    q : float
        Continuous dividend yield.
    tau_grid : ndarray
        Times-to-maturity (years).

    Returns
    -------
    ndarray
        Forward prices, same shape as ``tau_grid``.
    """
    tau_grid = np.asarray(tau_grid, dtype=np.float64)
    return float(S0) * np.exp((float(r) - float(q)) * tau_grid)


def get_dividend_yield(ticker: str) -> float:
    """Return continuous dividend yield for a ticker.

    Tries ``yfinance.Ticker.info['dividendYield']`` first, then falls back
    to ``KNOWN_DIVIDEND_YIELDS``, then to 0.0.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol.

    Returns
    -------
    float
        Annualized dividend yield in decimal form.
    """
    try:
        import yfinance as yf  # type: ignore
        try:
            info = yf.Ticker(ticker).info
            q = info.get('dividendYield', None)
            if q is not None and np.isfinite(q) and q > 0:
                # yfinance returns either decimal or percent; normalize.
                if q > 1.0:  # likely percent
                    q = q / 100.0
                if 0 < q < 0.20:
                    return float(q)
        except Exception as exc:
            logger.debug("yfinance dividendYield fetch failed for %s: %s",
                         ticker, exc)
    except ImportError:
        pass
    return float(KNOWN_DIVIDEND_YIELDS.get(ticker, 0.0))


# ---------------------------------------------------------------------------
# Fix 1: Log-moneyness coordinates
# ---------------------------------------------------------------------------

def compute_log_moneyness(K: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Compute log-moneyness ``k = log(K / F)``.

    Parameters
    ----------
    K : array-like
        Strike prices.
    F : array-like
        Forward prices (must broadcast with ``K``).

    Returns
    -------
    ndarray
        Log-moneyness.
    """
    K = np.asarray(K, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    return np.log(K / F)


def build_logm_surface(option_df: pd.DataFrame, S0: float, r: float, q: float,
                       n_k: int = 40, k_range: tuple[float, float] = (-0.25, 0.25),
                       n_tau: Optional[int] = None) -> dict[str, Any]:
    """Build a call-price surface ``C(k, tau)`` on a regular log-moneyness grid.

    Interpolates raw mid prices directly (no IV inversion). Useful when SVI
    smoothing is undesired or fails.

    Parameters
    ----------
    option_df : DataFrame
        Must contain columns ``strike, tau, mid_price``.
    S0, r, q : float
        Spot, risk-free rate, dividend yield.
    n_k : int, default 40
        Number of log-moneyness grid points.
    k_range : (float, float)
        (k_min, k_max) for grid.
    n_tau : int or None
        Number of tau grid points; defaults to unique expirations in data
        (must be >= 3).

    Returns
    -------
    dict
        Keys: ``C_surface`` (n_k, n_tau), ``k_grid``, ``tau_grid``,
        ``F_grid``, ``k_obs``, ``q``, ``method`` (``'direct_C_interp'``).
    """
    df = option_df.copy()
    df = df.dropna(subset=['strike', 'tau', 'mid_price'])
    df = df[df['mid_price'] > 0]
    if len(df) < 10:
        raise ValueError(f"Too few options ({len(df)}) for surface build.")

    K_obs = df['strike'].values.astype(np.float64)
    tau_obs = df['tau'].values.astype(np.float64)
    C_obs = df['mid_price'].values.astype(np.float64)

    F_obs = float(S0) * np.exp((float(r) - float(q)) * tau_obs)
    k_obs = compute_log_moneyness(K_obs, F_obs)

    unique_taus = np.sort(np.unique(np.round(tau_obs, 6)))
    if len(unique_taus) < 3:
        raise ValueError(
            f"Need at least 3 distinct expirations, got {len(unique_taus)}."
        )
    if n_tau is not None and n_tau >= 3:
        tau_grid = np.linspace(unique_taus.min(), unique_taus.max(),
                               int(n_tau))
    else:
        tau_grid = unique_taus

    k_grid = np.linspace(float(k_range[0]), float(k_range[1]), int(n_k))
    F_grid = compute_forward_prices(S0, r, q, tau_grid)

    # Interpolate raw C on (k, tau) directly.
    KK, TT = np.meshgrid(k_grid, tau_grid, indexing='ij')
    points = np.column_stack([k_obs, tau_obs])
    try:
        C_surface = griddata(points, C_obs, (KK, TT), method='linear')
    except Exception:
        C_surface = griddata(points, C_obs, (KK, TT), method='nearest')

    nan_mask = np.isnan(C_surface)
    if np.any(nan_mask):
        C_nn = griddata(points, C_obs, (KK, TT), method='nearest')
        C_surface[nan_mask] = C_nn[nan_mask]

    C_surface = np.clip(C_surface, 1e-6, None)

    return {
        'C_surface': C_surface,
        'k_grid': k_grid,
        'tau_grid': tau_grid,
        'F_grid': F_grid,
        'k_obs': k_obs,
        'q': float(q),
        'method': 'direct_C_interp',
    }


# ---------------------------------------------------------------------------
# Fix 3: SVI smoothing
# ---------------------------------------------------------------------------

def _svi_total_variance(params: np.ndarray, k: np.ndarray) -> np.ndarray:
    """SVI raw parameterization: ``w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + s^2))``."""
    a, b, rho, m, s = params
    km = k - m
    return a + b * (rho * km + np.sqrt(km * km + s * s))


def fit_svi_slice(k_obs: np.ndarray, w_obs: np.ndarray,
                  init: Optional[np.ndarray] = None) -> Optional[dict[str, Any]]:
    """Fit SVI parameterization to one expiration's total implied variance.

    SVI raw form:

        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + s^2))

    Parameters
    ----------
    k_obs : ndarray
        Log-moneyness observations.
    w_obs : ndarray
        Total implied variance observations (sigma^2 * tau).
    init : ndarray or None
        Initial guess [a, b, rho, m, s].

    Returns
    -------
    dict or None
        Keys: ``a, b, rho, m, s, converged, rmse``; ``None`` on failure.
    """
    k_obs = np.asarray(k_obs, dtype=np.float64)
    w_obs = np.asarray(w_obs, dtype=np.float64)

    mask = np.isfinite(k_obs) & np.isfinite(w_obs) & (w_obs > 0)
    k_obs = k_obs[mask]
    w_obs = w_obs[mask]
    if len(k_obs) < 5:
        return None

    w_mean = float(np.mean(w_obs))
    if init is None:
        init = np.array([w_mean * 0.5, 0.1, -0.3, 0.0, 0.1],
                        dtype=np.float64)

    bounds = [
        (-1.0, 5.0),     # a
        (1e-6, 5.0),     # b
        (-0.999, 0.999),  # rho
        (-1.0, 1.0),     # m
        (1e-3, 5.0),     # s
    ]

    def loss(params: np.ndarray) -> float:
        try:
            w_pred = _svi_total_variance(params, k_obs)
        except Exception:
            return 1e10
        if np.any(~np.isfinite(w_pred)) or np.any(w_pred < 0):
            return 1e10
        return float(np.mean((w_pred - w_obs) ** 2))

    try:
        result = minimize(loss, init, method='L-BFGS-B', bounds=bounds)
    except Exception as exc:
        logger.debug("SVI minimize failed: %s", exc)
        return None

    if not result.success:
        # Try a second start.
        init2 = np.array([w_mean * 0.3, 0.2, 0.0, 0.0, 0.2])
        try:
            result2 = minimize(loss, init2, method='L-BFGS-B', bounds=bounds)
            if result2.fun < result.fun:
                result = result2
        except Exception:
            pass

    a, b, rho, m, s = result.x
    w_pred = _svi_total_variance(result.x, k_obs)
    rmse = float(np.sqrt(np.mean((w_pred - w_obs) ** 2)))

    return {
        'a': float(a),
        'b': float(b),
        'rho': float(rho),
        'm': float(m),
        's': float(s),
        'converged': bool(result.success),
        'rmse': rmse,
    }


def _bs_call_iv(C: float, S0: float, K: float, r: float, q: float,
                tau: float) -> float:
    """Brent inversion of BS call price -> implied vol (with dividends)."""
    if tau <= 0 or C <= 0:
        return float('nan')
    # Discounted intrinsic bounds.
    forward = S0 * np.exp((r - q) * tau)
    disc = np.exp(-r * tau)
    intrinsic = max(disc * (forward - K), 0.0)
    upper = S0 * np.exp(-q * tau)
    if C <= intrinsic + 1e-10 or C >= upper - 1e-10:
        return float('nan')

    def f(sigma: float) -> float:
        # BS call with continuous dividend: discount spot by exp(-q*tau).
        S_eff = S0 * np.exp(-q * tau)
        return float(bs_call_price(S_eff, K, r, sigma, tau)) - C

    try:
        return float(brentq(f, 1e-4, 5.0, maxiter=100, xtol=1e-6))
    except Exception:
        return float('nan')


def _bs_call_with_div(S0: float, K: float, r: float, q: float, sigma: float,
                      tau: float) -> float:
    """BS call price with continuous dividend yield."""
    if tau <= 0:
        return float(max(S0 - K, 0.0))
    S_eff = S0 * np.exp(-q * tau)
    return float(bs_call_price(S_eff, K, r, sigma, tau))


def build_logm_surface_svi(option_df: pd.DataFrame, S0: float, r: float, q: float,
                           n_k: int = 40, k_range: tuple[float, float] = (-0.25, 0.25),
                           n_tau: Optional[int] = None) -> dict[str, Any]:
    """Build SVI-smoothed call-price surface on a regular log-moneyness grid.

    For each expiration:
      1. compute observed log-moneyness and total variance w = iv^2 * tau,
      2. fit SVI to (k, w),
      3. evaluate smooth w(k) on the dense grid,
      4. convert back to sigma = sqrt(w / tau),
      5. price calls via BS at each (k, tau).

    Falls back to a quadratic fit on iv(k) when SVI fails for an expiration.

    Returns
    -------
    dict
        Same keys as ``build_logm_surface`` plus ``svi_params`` (list of
        per-tau dicts including ``method`` field), ``n_svi_success``,
        ``n_svi_fallback``, ``method='svi_smoothed'``.
    """
    df = option_df.copy()
    df = df.dropna(subset=['strike', 'tau', 'mid_price', 'implied_vol'])
    df = df[(df['mid_price'] > 0) & (df['implied_vol'] > 0)]
    if len(df) < 10:
        raise ValueError(f"Too few options ({len(df)}) for SVI surface build.")

    K_obs = df['strike'].values.astype(np.float64)
    tau_obs = df['tau'].values.astype(np.float64)
    iv_obs = df['implied_vol'].values.astype(np.float64)

    F_obs = float(S0) * np.exp((float(r) - float(q)) * tau_obs)
    k_obs_all = compute_log_moneyness(K_obs, F_obs)

    unique_taus = np.sort(np.unique(np.round(tau_obs, 6)))
    if len(unique_taus) < 3:
        raise ValueError(
            f"Need at least 3 distinct expirations, got {len(unique_taus)}."
        )
    if n_tau is not None and n_tau >= 3:
        tau_grid = np.linspace(unique_taus.min(), unique_taus.max(),
                               int(n_tau))
    else:
        tau_grid = unique_taus

    k_grid = np.linspace(float(k_range[0]), float(k_range[1]), int(n_k))
    F_grid = compute_forward_prices(S0, r, q, tau_grid)

    # Smooth IV per *raw* expiration, then interpolate sigma(k, tau) onto grid.
    svi_params: list[dict[str, Any]] = []
    n_success = 0
    n_fallback = 0

    smooth_k_pts: list[float] = []
    smooth_tau_pts: list[float] = []
    smooth_sigma_pts: list[float] = []

    for tau_val in unique_taus:
        sel = np.isclose(tau_obs, tau_val, atol=1e-6)
        if sel.sum() < 5:
            continue
        k_slice = k_obs_all[sel]
        iv_slice = iv_obs[sel]
        w_slice = iv_slice ** 2 * tau_val

        fit = fit_svi_slice(k_slice, w_slice)
        method_used = 'svi'

        # Dense k for evaluation at this expiration.
        eval_k = np.clip(k_grid, k_slice.min(), k_slice.max())

        if fit is not None and fit['converged']:
            params = np.array([fit['a'], fit['b'], fit['rho'], fit['m'],
                               fit['s']])
            w_dense = _svi_total_variance(params, eval_k)
            w_dense = np.clip(w_dense, 1e-8, None)
            sigma_dense = np.sqrt(w_dense / tau_val)
            n_success += 1
        else:
            method_used = 'quadratic'
            n_fallback += 1
            try:
                coef = np.polyfit(k_slice, iv_slice, 2)
                sigma_dense = np.polyval(coef, eval_k)
                sigma_dense = np.clip(sigma_dense, 0.01, 3.0)
                fit = {
                    'a': float('nan'), 'b': float('nan'),
                    'rho': float('nan'), 'm': float('nan'),
                    's': float('nan'),
                    'converged': False, 'rmse': float('nan'),
                }
            except Exception:
                sigma_dense = np.full_like(eval_k, float(np.median(iv_slice)))
                fit = {
                    'a': float('nan'), 'b': float('nan'),
                    'rho': float('nan'), 'm': float('nan'),
                    's': float('nan'),
                    'converged': False, 'rmse': float('nan'),
                }

        fit['tau'] = float(tau_val)
        fit['method'] = method_used
        svi_params.append(fit)

        for kk, ss in zip(k_grid, sigma_dense):
            smooth_k_pts.append(float(kk))
            smooth_tau_pts.append(float(tau_val))
            smooth_sigma_pts.append(float(ss))

    if not smooth_sigma_pts:
        raise ValueError("No SVI smoothing produced usable IV data.")

    smooth_k_arr = np.asarray(smooth_k_pts)
    smooth_tau_arr = np.asarray(smooth_tau_pts)
    smooth_sigma_arr = np.asarray(smooth_sigma_pts)

    KK, TT = np.meshgrid(k_grid, tau_grid, indexing='ij')
    sigma_surface = griddata(
        np.column_stack([smooth_k_arr, smooth_tau_arr]),
        smooth_sigma_arr,
        (KK, TT),
        method='linear',
    )
    nan_mask = np.isnan(sigma_surface)
    if np.any(nan_mask):
        sigma_nn = griddata(
            np.column_stack([smooth_k_arr, smooth_tau_arr]),
            smooth_sigma_arr,
            (KK, TT),
            method='nearest',
        )
        sigma_surface[nan_mask] = sigma_nn[nan_mask]
    sigma_surface = np.clip(sigma_surface, 0.01, 3.0)

    # Convert sigma to call price on (k, tau) using K = F(tau) * exp(k).
    C_surface = np.zeros_like(KK)
    for i in range(KK.shape[0]):
        for j in range(KK.shape[1]):
            K_val = F_grid[j] * np.exp(k_grid[i])
            tau_val = tau_grid[j]
            sigma_val = sigma_surface[i, j]
            C_surface[i, j] = _bs_call_with_div(
                float(S0), float(K_val), float(r), float(q),
                float(sigma_val), float(tau_val),
            )

    C_surface = np.clip(C_surface, 1e-6, None)

    return {
        'C_surface': C_surface,
        'k_grid': k_grid,
        'tau_grid': tau_grid,
        'F_grid': F_grid,
        'k_obs': k_obs_all,
        'sigma_surface': sigma_surface,
        'q': float(q),
        'svi_params': svi_params,
        'n_svi_success': int(n_success),
        'n_svi_fallback': int(n_fallback),
        'method': 'svi_smoothed',
    }


# ---------------------------------------------------------------------------
# Fix 2: Liquidity-weighted STLSQ
# ---------------------------------------------------------------------------

def compute_liquidity_weights(option_df: pd.DataFrame, k_grid: np.ndarray,
                              tau_grid: np.ndarray, q: Optional[float] = None,
                              S0: Optional[float] = None) -> np.ndarray:
    """Compute per-grid-point liquidity weights from nearest observation.

    For each (k, tau) grid point, find the nearest option (in (k, tau) space)
    and combine:
      w_liq = sqrt(volume * openInterest) (normalized so max = 1)
      w_spread = 1 / (1 + bid_ask_spread_pct)
      w_atm = exp(-2 * k^2)
      weight = w_liq * w_spread * w_atm

    Parameters
    ----------
    option_df : DataFrame
        Must contain at minimum ``strike, tau, mid_price``. ``volume``,
        ``openInterest``, ``bid``, ``ask`` are used when present.
    k_grid : ndarray
    tau_grid : ndarray
    q : float or None
        Dividend yield; if provided with ``S0``, k_obs is computed from
        the dataframe. If None, defaults to 0.
    S0 : float or None

    Returns
    -------
    ndarray
        2D weights array of shape (n_k, n_tau).
    """
    df = option_df.copy()
    df = df.dropna(subset=['strike', 'tau'])
    if len(df) == 0:
        return np.ones((len(k_grid), len(tau_grid)))

    # Compute log-moneyness for each observed option.
    if S0 is None:
        S0 = float(df['S0'].iloc[0]) if 'S0' in df.columns else 100.0
    if q is None:
        q = 0.0
    r_val = float(df['r'].iloc[0]) if 'r' in df.columns else 0.05

    K_obs = df['strike'].values.astype(np.float64)
    tau_obs = df['tau'].values.astype(np.float64)
    F_obs = float(S0) * np.exp((r_val - float(q)) * tau_obs)
    k_obs = compute_log_moneyness(K_obs, F_obs)

    # Per-option liquidity ingredients.
    volume = df['volume'].values.astype(np.float64) if 'volume' in df.columns \
        else np.ones(len(df))
    oi = df['openInterest'].values.astype(np.float64) if 'openInterest' in df.columns \
        else np.ones(len(df))
    volume = np.clip(volume, 0, None)
    oi = np.clip(oi, 0, None)
    liq_per_obs = np.sqrt(volume * oi)
    if liq_per_obs.max() > 0:
        liq_per_obs = liq_per_obs / liq_per_obs.max()
    else:
        liq_per_obs = np.ones_like(liq_per_obs)

    if 'bid' in df.columns and 'ask' in df.columns and 'mid_price' in df.columns:
        bid = df['bid'].values.astype(np.float64)
        ask = df['ask'].values.astype(np.float64)
        mid = np.clip(df['mid_price'].values.astype(np.float64), 1e-6, None)
        spread_pct = np.clip((ask - bid) / mid, 0.0, 10.0)
    else:
        spread_pct = np.zeros(len(df))
    spread_w_per_obs = 1.0 / (1.0 + spread_pct)

    # For each grid point, find nearest observation and use its w_liq * w_spread.
    n_k = len(k_grid)
    n_tau = len(tau_grid)
    weights = np.zeros((n_k, n_tau), dtype=np.float64)

    # Normalize axes for distance to avoid one dominating.
    k_scale = max(k_grid.max() - k_grid.min(), 1e-6)
    tau_scale = max(tau_grid.max() - tau_grid.min(), 1e-6)

    for i, k_val in enumerate(k_grid):
        for j, tau_val in enumerate(tau_grid):
            dk = (k_obs - k_val) / k_scale
            dt = (tau_obs - tau_val) / tau_scale
            d2 = dk * dk + dt * dt
            idx = int(np.argmin(d2))
            w_liq = liq_per_obs[idx]
            w_spread = spread_w_per_obs[idx]
            w_atm = float(np.exp(-2.0 * k_val * k_val))
            weights[i, j] = w_liq * w_spread * w_atm

    return weights


def weighted_stlsq(library: np.ndarray, target: np.ndarray,
                   weights: Optional[np.ndarray] = None,
                   threshold: float = 0.01,
                   max_iter: int = 10) -> tuple[np.ndarray, np.ndarray, float]:
    """Sequentially thresholded *weighted* least squares.

    If ``weights`` is None or all ones, this returns identical output to
    ``src.sindy_discovery.stlsq`` (and computes the unweighted R^2).
    Otherwise we left-multiply rows of ``library`` and ``target`` by
    ``sqrt(weight)`` before solving.

    Parameters
    ----------
    library : ndarray, shape (n, p)
    target : ndarray, shape (n,)
    weights : ndarray or None, shape (n,)
    threshold : float
    max_iter : int

    Returns
    -------
    (coefficients, active_mask, r2_score)
        R^2 is computed on the unweighted residuals.
    """
    library = np.asarray(library, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    is_uniform = (weights is None) or np.allclose(np.asarray(weights), 1.0)

    if is_uniform:
        # Delegate exactly to the existing stlsq for bit-equality.
        coeffs, active = stlsq(library, target, threshold, max_iter=max_iter)
    else:
        w = np.clip(np.asarray(weights, dtype=np.float64).ravel(), 0.0, None)
        sw = np.sqrt(w)
        lib_w = library * sw[:, None]
        tgt_w = target * sw
        coeffs, active = stlsq(lib_w, tgt_w, threshold, max_iter=max_iter)

    # Unweighted R^2 (so it's comparable across runs).
    pred = library @ coeffs
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    if ss_tot < 1e-30:
        r2 = 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    return coeffs, active, float(r2)


# ---------------------------------------------------------------------------
# Fix 4: 2-term Dupire in log-moneyness
# ---------------------------------------------------------------------------

def _compute_logm_derivatives(C: np.ndarray, k_grid: np.ndarray,
                              tau_grid: np.ndarray) -> dict[str, np.ndarray]:
    """Central FD derivatives of a (k, tau) surface."""
    dk = float(k_grid[1] - k_grid[0])
    dtau = float(tau_grid[1] - tau_grid[0])

    # dC/dk along axis 0
    dCdk = np.gradient(C, dk, axis=0, edge_order=2)
    # d2C/dk2
    d2Cdk2 = np.gradient(dCdk, dk, axis=0, edge_order=2)
    # dC/dtau along axis 1
    dCdtau = np.gradient(C, dtau, axis=1, edge_order=2)

    return {'dCdk': dCdk, 'd2Cdk2': d2Cdk2, 'dCdtau': dCdtau}


def dupire_logm_2term(C_surface: np.ndarray, k_grid: np.ndarray,
                      tau_grid: np.ndarray,
                      weights: Optional[np.ndarray] = None,
                      return_full: bool = False) -> dict[str, Any]:
    """2-term log-moneyness Dupire regression.

    Solves OLS for:

        dC/dtau = coef_d2 * d2C/dk2 + coef_d1 * dC/dk

    Then extracts:
        sigma_loc = sqrt(max(0, 2 * coef_d2))
        drift     = coef_d1
        (r - q)   = -drift - 0.5 * sigma_loc^2

    Parameters
    ----------
    C_surface : ndarray, shape (n_k, n_tau)
    k_grid : ndarray, shape (n_k,), uniformly spaced
    tau_grid : ndarray, shape (n_tau,), uniformly spaced
    weights : ndarray, shape (n_k, n_tau), optional
    return_full : bool
        If True, include the raw library and target.

    Returns
    -------
    dict
        Keys: ``coef_dCdk``, ``coef_d2Cdk2``, ``sigma_loc_discovered``,
        ``drift_discovered``, ``rq_implied``, ``r2_score``,
        ``condition_number``.
    """
    C = np.asarray(C_surface, dtype=np.float64)
    k_grid = np.asarray(k_grid, dtype=np.float64)
    tau_grid = np.asarray(tau_grid, dtype=np.float64)
    if C.shape != (len(k_grid), len(tau_grid)):
        raise ValueError(
            f"C_surface shape {C.shape} != (n_k={len(k_grid)}, "
            f"n_tau={len(tau_grid)})"
        )

    derivs = _compute_logm_derivatives(C, k_grid, tau_grid)
    library = np.column_stack([derivs['dCdk'].ravel(),
                               derivs['d2Cdk2'].ravel()])
    target = derivs['dCdtau'].ravel()

    if weights is not None:
        w = np.clip(np.asarray(weights, dtype=np.float64).ravel(), 0.0, None)
        sw = np.sqrt(w)
        A = library * sw[:, None]
        b = target * sw
    else:
        A = library
        b = target

    cond = float(np.linalg.cond(library))
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    coef_d1 = float(coef[0])
    coef_d2 = float(coef[1])

    pred = library @ coef
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0

    sigma_loc = float(np.sqrt(max(0.0, 2.0 * coef_d2)))
    drift = coef_d1
    rq_implied = -drift - 0.5 * sigma_loc * sigma_loc

    out = {
        'coef_dCdk': coef_d1,
        'coef_d2Cdk2': coef_d2,
        'sigma_loc_discovered': sigma_loc,
        'drift_discovered': drift,
        'rq_implied': float(rq_implied),
        'r2_score': float(r2),
        'condition_number': cond,
    }
    if return_full:
        out['library'] = library
        out['target'] = target
    return out


def dupire_logm_2term_windowed(C_surface: np.ndarray, k_grid: np.ndarray,
                               tau_grid: np.ndarray,
                               weights: Optional[np.ndarray] = None,
                               window_size: int = 10, stride: int = 2,
                               min_r2: float = 0.3) -> dict[str, Any]:
    """Sliding-window 2-term Dupire regression across (k, tau).

    For each (k_center, tau_center) we fit a local 2-term Dupire on the
    surrounding window. Windows must contain at least 100 grid points.

    Parameters
    ----------
    C_surface : ndarray, shape (n_k, n_tau)
    k_grid, tau_grid : ndarray
    weights : ndarray or None
    window_size : int, default 10
        Side length of the window (window_size x window_size).
    stride : int, default 2
    min_r2 : float, default 0.3

    Returns
    -------
    dict
        Keys: ``k_centers``, ``tau_centers``, ``sigma_loc_grid`` (2D),
        ``r2_grid``, ``n_valid_windows``, ``n_total_windows``.
    """
    C = np.asarray(C_surface, dtype=np.float64)
    n_k, n_tau = C.shape
    if window_size > n_k or window_size > n_tau:
        raise ValueError(
            f"window_size {window_size} too large for surface {C.shape}"
        )

    k_centers_list: list[float] = []
    tau_centers_list: list[float] = []

    i_starts = list(range(0, n_k - window_size + 1, stride))
    j_starts = list(range(0, n_tau - window_size + 1, stride))
    if not i_starts:
        i_starts = [0]
    if not j_starts:
        j_starts = [0]

    sigma_grid = np.full((len(i_starts), len(j_starts)), np.nan)
    r2_grid = np.full((len(i_starts), len(j_starts)), np.nan)

    n_valid = 0
    n_total = 0

    for ii, i in enumerate(i_starts):
        for jj, j in enumerate(j_starts):
            n_total += 1
            i_end = i + window_size
            j_end = j + window_size
            C_win = C[i:i_end, j:j_end]
            k_win = k_grid[i:i_end]
            tau_win = tau_grid[j:j_end]
            if C_win.size < 100:
                continue

            if weights is not None:
                w_win = weights[i:i_end, j:j_end]
            else:
                w_win = None

            try:
                res = dupire_logm_2term(C_win, k_win, tau_win, weights=w_win)
            except Exception:
                continue

            if not np.isfinite(res['sigma_loc_discovered']):
                continue
            if res['r2_score'] < min_r2:
                # Still record values, but mark as invalid for counting.
                sigma_grid[ii, jj] = res['sigma_loc_discovered']
                r2_grid[ii, jj] = res['r2_score']
                continue
            sigma_grid[ii, jj] = res['sigma_loc_discovered']
            r2_grid[ii, jj] = res['r2_score']
            n_valid += 1

            k_center = float(0.5 * (k_win[0] + k_win[-1]))
            tau_center = float(0.5 * (tau_win[0] + tau_win[-1]))
            if ii == 0:
                tau_centers_list.append(tau_center)
            if jj == 0:
                k_centers_list.append(k_center)

    # Build axis centers cleanly.
    k_centers = np.array([
        0.5 * (k_grid[i] + k_grid[i + window_size - 1]) for i in i_starts
    ])
    tau_centers = np.array([
        0.5 * (tau_grid[j] + tau_grid[j + window_size - 1]) for j in j_starts
    ])

    return {
        'k_centers': k_centers,
        'tau_centers': tau_centers,
        'sigma_loc_grid': sigma_grid,
        'r2_grid': r2_grid,
        'n_valid_windows': int(n_valid),
        'n_total_windows': int(n_total),
    }


# ---------------------------------------------------------------------------
# Fix 6: ATM filter
# ---------------------------------------------------------------------------

def filter_atm(option_df: pd.DataFrame, k_low: float = -0.08,
               k_high: float = 0.08, S0: Optional[float] = None,
               r: Optional[float] = None, q: Optional[float] = None,
               tau_col: str = 'tau') -> pd.DataFrame:
    """Filter option chain to near-ATM region in log-moneyness coordinates.

    Parameters
    ----------
    option_df : DataFrame
        Must contain ``strike`` and the tau column.
    k_low, k_high : float
        Inclusive log-moneyness band.
    S0 : float or None
        Spot price; if None, read from df.
    r : float or None
        Risk-free rate; if None, read from df.
    q : float or None
        Dividend yield; default 0.
    tau_col : str
        Column name for time-to-maturity.

    Returns
    -------
    DataFrame
        Subset of rows with k in [k_low, k_high].
    """
    df = option_df.copy()
    if S0 is None:
        S0 = float(df['S0'].iloc[0]) if 'S0' in df.columns else 100.0
    if r is None:
        r = float(df['r'].iloc[0]) if 'r' in df.columns else 0.05
    if q is None:
        q = 0.0

    K = df['strike'].values.astype(np.float64)
    tau = df[tau_col].values.astype(np.float64)
    F = float(S0) * np.exp((float(r) - float(q)) * tau)
    k = compute_log_moneyness(K, F)

    mask = (k >= float(k_low)) & (k <= float(k_high))
    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fix 7: Combined best-case pipeline
# ---------------------------------------------------------------------------

def run_improved_real_pipeline(option_data: dict[str, Any], ticker: str,
                               use_svi: bool = True, use_weights: bool = True,
                               run_atm: bool = True, run_windowed: bool = True,
                               seed: int = 42) -> dict[str, Any]:
    """Run the full improved real-data pipeline for one ticker.

    Each stage is wrapped in try/except so partial failures still return.

    Parameters
    ----------
    option_data : dict
        Must contain at least ``S0``, ``r``, ``option_df``, ``implied_vols``.
    ticker : str
    use_svi : bool, default True
    use_weights : bool, default True
    run_atm : bool, default True
    run_windowed : bool, default True
    seed : int, default 42

    Returns
    -------
    dict
        Comprehensive result with intermediate + final outputs.
    """
    set_all_seeds(seed)

    out: dict[str, Any] = {
        'ticker': ticker,
        'method_chain': [],
        'svi_stats': None,
        'surface_info': None,
        'full_range': None,
        'atm_only': None,
        'windowed': None,
        'avg_market_iv': float('nan'),
        'errors': {},
    }

    # 1. Dividend yield.
    try:
        q = get_dividend_yield(ticker)
    except Exception as exc:
        logger.warning("%s: get_dividend_yield failed (%s); using 0.0",
                       ticker, exc)
        q = 0.0
        out['errors']['q'] = str(exc)
    out['q'] = float(q)
    out['method_chain'].append(f"q={q:.4f}")

    S0 = float(option_data['S0'])
    r = float(option_data['r'])
    df = option_data['option_df'].copy()
    out['avg_market_iv'] = float(np.nanmean(option_data.get('implied_vols',
                                                            [float('nan')])))

    # 2. Forward prices on a hint tau grid.
    try:
        tau_unique = np.sort(np.unique(np.round(df['tau'].values, 6)))
        out['F_grid_hint'] = compute_forward_prices(S0, r, q, tau_unique)
    except Exception as exc:
        out['errors']['forward'] = str(exc)
        logger.warning("%s: forward price computation failed: %s", ticker, exc)

    # 3 + 4. Build log-moneyness surface (SVI or plain).
    surface = None
    try:
        if use_svi:
            surface = build_logm_surface_svi(df, S0, r, q)
            out['method_chain'].append('svi_smoothed')
            out['svi_stats'] = {
                'n_success': surface['n_svi_success'],
                'n_fallback': surface['n_svi_fallback'],
                'svi_params': surface['svi_params'],
            }
        else:
            surface = build_logm_surface(df, S0, r, q)
            out['method_chain'].append('direct_C_interp')
    except Exception as exc:
        logger.warning("%s: SVI surface failed (%s); falling back to direct interp",
                       ticker, exc)
        out['errors']['svi_surface'] = str(exc)
        try:
            surface = build_logm_surface(df, S0, r, q)
            out['method_chain'].append('direct_C_interp_fallback')
        except Exception as exc2:
            logger.error("%s: even direct surface failed: %s", ticker, exc2)
            out['errors']['direct_surface'] = str(exc2)
            return out

    out['surface_info'] = {
        'method': surface['method'],
        'n_k': len(surface['k_grid']),
        'n_tau': len(surface['tau_grid']),
        'k_range': (float(surface['k_grid'].min()),
                    float(surface['k_grid'].max())),
        'tau_range': (float(surface['tau_grid'].min()),
                      float(surface['tau_grid'].max())),
    }

    C_surface = surface['C_surface']
    k_grid = surface['k_grid']
    tau_grid = surface['tau_grid']

    # 5. Liquidity weights.
    weights = None
    if use_weights:
        try:
            weights = compute_liquidity_weights(df, k_grid, tau_grid, q=q,
                                                S0=S0)
            out['method_chain'].append('liquidity_weighted')
        except Exception as exc:
            logger.warning("%s: liquidity weights failed: %s", ticker, exc)
            out['errors']['weights'] = str(exc)
            weights = None

    # 6. Full-range 2-term Dupire.
    try:
        full = dupire_logm_2term(C_surface, k_grid, tau_grid, weights=weights)
        full['sigma_market'] = out['avg_market_iv']
        if np.isfinite(out['avg_market_iv']) and out['avg_market_iv'] > 0:
            full['sigma_rel_err'] = abs(
                full['sigma_loc_discovered'] - out['avg_market_iv']
            ) / out['avg_market_iv']
        else:
            full['sigma_rel_err'] = float('nan')
        out['full_range'] = full
        out['method_chain'].append('dupire_2term_full')
    except Exception as exc:
        logger.error("%s: full-range Dupire failed: %s", ticker, exc)
        out['errors']['full_range'] = str(exc)

    # 7. ATM-only run.
    if run_atm:
        try:
            atm_df = filter_atm(df, k_low=-0.08, k_high=0.08, S0=S0, r=r, q=q)
            if len(atm_df) >= 10:
                if use_svi:
                    try:
                        atm_surface = build_logm_surface_svi(
                            atm_df, S0, r, q, n_k=30, k_range=(-0.08, 0.08),
                        )
                    except Exception:
                        atm_surface = build_logm_surface(
                            atm_df, S0, r, q, n_k=30, k_range=(-0.08, 0.08),
                        )
                else:
                    atm_surface = build_logm_surface(
                        atm_df, S0, r, q, n_k=30, k_range=(-0.08, 0.08),
                    )
                atm_weights = None
                if use_weights:
                    try:
                        atm_weights = compute_liquidity_weights(
                            atm_df, atm_surface['k_grid'],
                            atm_surface['tau_grid'], q=q, S0=S0,
                        )
                    except Exception:
                        atm_weights = None
                atm_res = dupire_logm_2term(
                    atm_surface['C_surface'],
                    atm_surface['k_grid'],
                    atm_surface['tau_grid'],
                    weights=atm_weights,
                )
                atm_res['n_options'] = int(len(atm_df))
                if np.isfinite(out['avg_market_iv']) and out['avg_market_iv'] > 0:
                    atm_res['sigma_rel_err'] = abs(
                        atm_res['sigma_loc_discovered'] - out['avg_market_iv']
                    ) / out['avg_market_iv']
                else:
                    atm_res['sigma_rel_err'] = float('nan')
                out['atm_only'] = atm_res
                out['method_chain'].append('atm_2term')
            else:
                out['errors']['atm'] = f"only {len(atm_df)} ATM options"
        except Exception as exc:
            logger.warning("%s: ATM-only stage failed: %s", ticker, exc)
            out['errors']['atm'] = str(exc)

    # 8. Windowed 2-term Dupire.
    if run_windowed:
        try:
            window_size = min(10, len(k_grid), len(tau_grid))
            if window_size >= 4:
                win = dupire_logm_2term_windowed(
                    C_surface, k_grid, tau_grid, weights=weights,
                    window_size=window_size, stride=max(1, window_size // 5),
                )
                out['windowed'] = win
                out['method_chain'].append('windowed_2term')
        except Exception as exc:
            logger.warning("%s: windowed stage failed: %s", ticker, exc)
            out['errors']['windowed'] = str(exc)

    return out


def run_improved_pipeline_all_tickers(per_ticker_results: dict[str, Any],
                                      **kwargs: Any) -> dict[str, Any]:
    """Run :func:`run_improved_real_pipeline` over a dict of ticker results.

    Parameters
    ----------
    per_ticker_results : dict
        Mapping ticker -> dict that contains ``option_data`` (or itself is
        an option_data dict with ``S0, r, option_df``).
    **kwargs
        Passed through to :func:`run_improved_real_pipeline`.

    Returns
    -------
    dict
        Keys: ``per_ticker`` (mapping ticker -> result), ``summary_df``.
    """
    per_ticker: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for ticker, entry in per_ticker_results.items():
        # Allow entry to be the option_data itself or a wrapper.
        if isinstance(entry, dict) and 'option_data' in entry:
            option_data = entry['option_data']
        else:
            option_data = entry

        try:
            res = run_improved_real_pipeline(option_data, ticker, **kwargs)
            per_ticker[ticker] = res

            full = res.get('full_range') or {}
            atm = res.get('atm_only') or {}
            win = res.get('windowed') or {}
            rows.append({
                'ticker': ticker,
                'q': res.get('q', float('nan')),
                'avg_market_iv': res.get('avg_market_iv', float('nan')),
                'full_r2': full.get('r2_score', float('nan')),
                'full_sigma_loc': full.get('sigma_loc_discovered', float('nan')),
                'full_rq_implied': full.get('rq_implied', float('nan')),
                'full_sigma_rel_err': full.get('sigma_rel_err', float('nan')),
                'atm_r2': atm.get('r2_score', float('nan')),
                'atm_sigma_loc': atm.get('sigma_loc_discovered', float('nan')),
                'atm_sigma_rel_err': atm.get('sigma_rel_err', float('nan')),
                'win_n_valid': win.get('n_valid_windows', 0),
                'win_n_total': win.get('n_total_windows', 0),
                'method_chain': ' -> '.join(res.get('method_chain', [])),
            })
        except Exception as exc:
            logger.error("Ticker %s pipeline failed: %s", ticker, exc,
                         exc_info=True)
            per_ticker[ticker] = {'error': str(exc)}
            rows.append({'ticker': ticker, 'error': str(exc)})

    summary_df = pd.DataFrame(rows)
    return {'per_ticker': per_ticker, 'summary_df': summary_df}
