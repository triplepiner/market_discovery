"""
Ablation chain + alternative smoothing comparison (Agent B3).

Implements the 6-config ablation chain from the final revision PRD that
isolates the contribution of each pipeline component on real SPY data:

    A: FD derivs, raw-K coords, 5-term library, FD theta,         Linear
    B: GP derivs, raw-K coords, 5-term library, FD theta,         Linear
    C: GP derivs, log-m coords, 2-term library, FD theta,         Linear
    D: GP + SVI,  log-m coords, 2-term library, FD theta,         Linear
    E: GP + SVI,  log-m coords, 2-term library, analytical theta, Linear
    F: GP + SVI,  log-m coords, 2-term library, analytical theta, [2,1] KAN

Plus an alternative-smoothing comparison (replaces SVI in config E):

    - cubic-spline IV smoothing (scipy.interpolate.CubicSpline per expiry)
    - LOESS IV smoothing      (statsmodels lowess per expiry, falls back to
      a quadratic sliding window if statsmodels is unavailable)

Both pipelines are designed for the SPY snapshot at 20260329 (cached at
``outputs/tables/real_chain_SPY_20260329.csv``) but operate on any
DataFrame in the same shape (columns: strike, tau, mid_price, implied_vol,
S0, r).

All functions are wrapped in try/except so a single bad config can't crash
the sweep. CPU only. Seed = 42 throughout.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.data_generation import bs_call_price
from src.gp_derivatives import compute_gp_derivatives, fit_gp_surface
from src.real_data_v2 import (
    build_logm_surface_svi,
    compute_forward_prices,
    get_dividend_yield,
)
from src.real_data_v4 import (
    _central_dk,
    _r2_from,
    bs_theta_analytical,
    reconstruct_sigma_imp_grid,
)
from src.utils import setup_logging

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Snapshot loader
# ---------------------------------------------------------------------------

def load_snapshot(ticker: str = 'SPY', snapshot: str = '20260329',
                  cache_dir: Optional[str] = None) -> dict[str, Any]:
    """Load a date-stamped option-chain snapshot CSV.

    Returns a dict in the same shape as ``real_data.fetch_option_data``:
    ``{ticker, S0, r, option_df, ...}``.
    """
    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs', 'tables',
        )
    path = os.path.join(cache_dir, f'real_chain_{ticker}_{snapshot}.csv')
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Snapshot CSV not found: {path}")
    df = pd.read_csv(path)
    if 'mid_price' not in df.columns and 'mid' in df.columns:
        df = df.rename(columns={'mid': 'mid_price'})
    if len(df) < 10:
        raise ValueError(f"Snapshot has only {len(df)} rows (< 10).")
    S0 = float(df['S0'].iloc[0])
    r = float(df['r'].iloc[0])
    return {
        'ticker': ticker,
        'snapshot': snapshot,
        'S0': S0,
        'r': r,
        'option_df': df,
    }


# ---------------------------------------------------------------------------
# Shared helpers: linear regressions on a Dupire library
# ---------------------------------------------------------------------------

def _r2(y, yhat) -> float:
    y = np.asarray(y, dtype=np.float64).ravel()
    yhat = np.asarray(yhat, dtype=np.float64).ravel()
    return _r2_from(y, yhat)


def _ols(library: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """OLS with NaN masking. Returns (coef, R^2 on the masked rows)."""
    A = np.asarray(library, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64).ravel()
    mask = np.all(np.isfinite(A), axis=1) & np.isfinite(b)
    if mask.sum() < A.shape[1] + 1:
        return np.zeros(A.shape[1]), float('nan')
    Am, bm = A[mask], b[mask]
    coef, *_ = np.linalg.lstsq(Am, bm, rcond=None)
    pred = Am @ coef
    return coef, _r2(bm, pred)


def _bs_call_safe(S0: float, K: float, r: float, sigma: float,
                  tau: float, q: float) -> float:
    """BS call with continuous dividend yield: price a call on S0*exp(-q*tau)."""
    Sd = float(S0) * float(np.exp(-float(q) * float(tau)))
    return float(bs_call_price(float(Sd), float(K), float(r),
                               float(sigma), float(tau)))


# ---------------------------------------------------------------------------
# Build raw-K BS surface on a regular (K, tau) grid (used by configs A, B)
# ---------------------------------------------------------------------------

def _raw_K_surface_from_iv(option_df: pd.DataFrame, S0: float, r: float,
                            q: float, n_K: int = 40, n_tau: int = 15,
                            ) -> dict[str, Any]:
    """Linear-interpolate observed IV to a regular (K, tau) grid then BS-price.

    This mirrors ``src.real_data.construct_smooth_surface`` but with a
    user-controllable ``n_tau`` (real_data uses unique expirations).
    """
    from scipy.interpolate import griddata

    df = option_df.dropna(subset=['strike', 'tau', 'mid_price', 'implied_vol'])
    df = df[(df['mid_price'] > 0) & (df['implied_vol'] > 0)]
    strikes = df['strike'].values.astype(float)
    taus = df['tau'].values.astype(float)
    ivs = df['implied_vol'].values.astype(float)
    if len(df) < 10:
        raise ValueError(f"Too few options ({len(df)}) for raw-K surface.")

    K_grid = np.linspace(strikes.min(), strikes.max(), int(n_K))
    tau_grid = np.linspace(taus.min(), taus.max(), int(n_tau))
    KK, TT = np.meshgrid(K_grid, tau_grid, indexing='ij')

    pts = np.column_stack([strikes, taus])
    iv_grid = griddata(pts, ivs, (KK, TT), method='linear')
    nan_mask = np.isnan(iv_grid)
    if np.any(nan_mask):
        iv_nn = griddata(pts, ivs, (KK, TT), method='nearest')
        iv_grid[nan_mask] = iv_nn[nan_mask]
    iv_grid = np.clip(iv_grid, 0.01, 2.0)

    V = np.zeros_like(KK)
    for i in range(n_K):
        for j in range(n_tau):
            V[i, j] = _bs_call_safe(S0, float(KK[i, j]), r,
                                     float(iv_grid[i, j]),
                                     float(TT[i, j]), q)
    V = np.clip(V, 1e-6, None)
    return {'V': V, 'K_grid': K_grid, 'tau_grid': tau_grid,
            'iv_grid': iv_grid}


# ---------------------------------------------------------------------------
# Derivative engines: FD and GP for (V, K, tau) surfaces
# ---------------------------------------------------------------------------

def _fd_derivs_KT(V: np.ndarray, K_grid: np.ndarray,
                  tau_grid: np.ndarray) -> dict[str, np.ndarray]:
    dK = float(K_grid[1] - K_grid[0])
    dtau = float(tau_grid[1] - tau_grid[0])
    dV_dK = np.gradient(V, dK, axis=0, edge_order=2)
    d2V_dK2 = np.gradient(dV_dK, dK, axis=0, edge_order=2)
    dV_dtau = np.gradient(V, dtau, axis=1, edge_order=2)
    return {'V': V, 'dV_dK': dV_dK, 'd2V_dK2': d2V_dK2, 'dV_dtau': dV_dtau}


def _gp_derivs_KT(V: np.ndarray, K_grid: np.ndarray,
                  tau_grid: np.ndarray, kernel: str = 'rbf',
                  n_subsample: int = 300, seed: int = 42
                  ) -> dict[str, np.ndarray]:
    """GP-fit V(K, tau) and return analytic derivatives w.r.t. K and tau."""
    gp, _ = fit_gp_surface(V, K_grid, tau_grid, n_subsample=n_subsample,
                           seed=seed, kernel=kernel)
    d = compute_gp_derivatives(gp, K_grid, tau_grid)
    return {
        'V': d['V_smooth'],
        'dV_dK': d['dV_dS'],
        'd2V_dK2': d['d2V_dS2'],
        'dV_dtau': d['dV_dt'],
    }


# ---------------------------------------------------------------------------
# 5-term raw-K Dupire library: dV/dtau = c0*V + c1*K + c2*K*dV/dK
#                                       + c3*d2V/dK2 + c4*K^2*d2V/dK2
# Linear regression of FD-theta target.
# ---------------------------------------------------------------------------

def _fit_5term_raw_K(derivs: dict[str, np.ndarray], K_grid: np.ndarray,
                     tau_grid: np.ndarray, trim: int = 3) -> dict[str, Any]:
    V = derivs['V']
    dV_dK = derivs['dV_dK']
    d2V_dK2 = derivs['d2V_dK2']
    dV_dtau = derivs['dV_dtau']

    n_K, n_tau = V.shape
    KK = np.tile(K_grid.reshape(-1, 1), (1, n_tau))

    # trim boundary rows/cols (FD edge effects)
    t = int(max(1, min(trim, n_K // 3, n_tau // 3)))
    sl = (slice(t, n_K - t), slice(t, n_tau - t))

    library = np.column_stack([
        V[sl].ravel(),
        KK[sl].ravel(),
        (KK[sl] * dV_dK[sl]).ravel(),
        d2V_dK2[sl].ravel(),
        (KK[sl] ** 2 * d2V_dK2[sl]).ravel(),
    ])
    target = dV_dtau[sl].ravel()
    coef, r2 = _ols(library, target)
    return {'r2': float(r2), 'coef': coef, 'n_points': int(library.shape[0])}


# ---------------------------------------------------------------------------
# 2-term log-m Dupire library: theta = c1*dC/dk + c2*d2C/dk2  (linear)
# ---------------------------------------------------------------------------

def _fit_2term_logm(dCdk: np.ndarray, d2Cdk2: np.ndarray,
                    theta: np.ndarray) -> dict[str, Any]:
    library = np.column_stack([dCdk.ravel(), d2Cdk2.ravel()])
    target = theta.ravel()
    coef, r2 = _ols(library, target)
    return {'r2': float(r2), 'coef': coef}


# ---------------------------------------------------------------------------
# Log-moneyness coord surface from observed IVs WITHOUT SVI smoothing.
# Uses plain linear griddata interpolation on (k, tau) to a regular grid.
# Used by config C (GP, log-m, 2-term, FD theta).
# ---------------------------------------------------------------------------

def _logm_surface_no_svi(option_df: pd.DataFrame, S0: float, r: float,
                          q: float, n_k: int = 40, n_tau: int = 15,
                          k_range: tuple[float, float] = (-0.25, 0.25),
                          ) -> dict[str, Any]:
    """Build a (k, tau) call surface by linearly interpolating IV in log-m.

    This is the "log-m coords but no SVI" intermediate step. Per the v2
    docstring this corresponds to ``build_logm_surface`` with method
    'linear'.
    """
    from scipy.interpolate import griddata

    df = option_df.dropna(subset=['strike', 'tau', 'mid_price', 'implied_vol'])
    df = df[(df['mid_price'] > 0) & (df['implied_vol'] > 0)]
    K_obs = df['strike'].values.astype(float)
    tau_obs = df['tau'].values.astype(float)
    iv_obs = df['implied_vol'].values.astype(float)
    F_obs = float(S0) * np.exp((float(r) - float(q)) * tau_obs)
    k_obs = np.log(K_obs / F_obs)

    k_grid = np.linspace(float(k_range[0]), float(k_range[1]), int(n_k))
    tau_grid = np.linspace(tau_obs.min(), tau_obs.max(), int(n_tau))
    F_grid = compute_forward_prices(S0, r, q, tau_grid)
    KK, TT = np.meshgrid(k_grid, tau_grid, indexing='ij')

    pts = np.column_stack([k_obs, tau_obs])
    sigma_grid = griddata(pts, iv_obs, (KK, TT), method='linear')
    nan_mask = np.isnan(sigma_grid)
    if np.any(nan_mask):
        sigma_nn = griddata(pts, iv_obs, (KK, TT), method='nearest')
        sigma_grid[nan_mask] = sigma_nn[nan_mask]
    sigma_grid = np.clip(sigma_grid, 0.01, 3.0)

    C = np.zeros_like(KK)
    for i in range(n_k):
        for j in range(n_tau):
            K_val = F_grid[j] * np.exp(k_grid[i])
            C[i, j] = _bs_call_safe(S0, float(K_val), r,
                                     float(sigma_grid[i, j]),
                                     float(tau_grid[j]), q)
    C = np.clip(C, 1e-6, None)
    return {'C_surface': C, 'k_grid': k_grid, 'tau_grid': tau_grid,
            'sigma_surface': sigma_grid}


# ---------------------------------------------------------------------------
# Alternative IV smoothers (replace SVI in config E): cubic spline & LOESS
# ---------------------------------------------------------------------------

def cubic_spline_smooth_surface(option_df: pd.DataFrame, S0: float, r: float,
                                 q: float, n_k: int = 40, n_tau: int = 15,
                                 k_range: tuple[float, float] = (-0.25, 0.25),
                                 ) -> dict[str, Any]:
    """Per-expiration cubic-spline smoothing of IV in log-moneyness.

    For each observed expiration:
      1. compute observed (k, iv) pairs,
      2. fit a CubicSpline to the iv(k) curve (natural BC, clipped extrap),
      3. evaluate at every k in the regular grid (clamped to observed range),
    Then interpolate sigma across tau linearly to the regular tau grid and
    BS-price every cell. Returns the same dict shape as
    ``build_logm_surface_svi``.
    """
    from scipy.interpolate import CubicSpline, griddata

    df = option_df.dropna(subset=['strike', 'tau', 'mid_price', 'implied_vol'])
    df = df[(df['mid_price'] > 0) & (df['implied_vol'] > 0)]
    K_obs = df['strike'].values.astype(float)
    tau_obs = df['tau'].values.astype(float)
    iv_obs = df['implied_vol'].values.astype(float)
    F_obs = float(S0) * np.exp((float(r) - float(q)) * tau_obs)
    k_obs_all = np.log(K_obs / F_obs)

    unique_taus = np.sort(np.unique(np.round(tau_obs, 6)))
    if len(unique_taus) < 3:
        raise ValueError(f"Need 3+ expirations, got {len(unique_taus)}.")
    tau_grid = np.linspace(unique_taus.min(), unique_taus.max(), int(n_tau))
    k_grid = np.linspace(float(k_range[0]), float(k_range[1]), int(n_k))
    F_grid = compute_forward_prices(S0, r, q, tau_grid)

    smooth_pts: list[tuple[float, float, float]] = []
    n_fits = 0
    for tau_val in unique_taus:
        sel = np.isclose(tau_obs, tau_val, atol=1e-6)
        if sel.sum() < 4:
            continue
        k_slice = k_obs_all[sel]
        iv_slice = iv_obs[sel]
        order = np.argsort(k_slice)
        k_sorted = k_slice[order]
        iv_sorted = iv_slice[order]
        # CubicSpline needs strictly increasing x.
        uniq_mask = np.concatenate([[True], np.diff(k_sorted) > 1e-8])
        k_sorted = k_sorted[uniq_mask]
        iv_sorted = iv_sorted[uniq_mask]
        if k_sorted.size < 4:
            continue
        try:
            cs = CubicSpline(k_sorted, iv_sorted, bc_type='natural',
                             extrapolate=False)
            eval_k = np.clip(k_grid, k_sorted.min(), k_sorted.max())
            iv_dense = cs(eval_k)
            # Replace any NaN (clipped extrap) with edge values.
            if np.any(~np.isfinite(iv_dense)):
                iv_dense = np.where(
                    np.isfinite(iv_dense), iv_dense,
                    np.interp(eval_k, k_sorted, iv_sorted))
            iv_dense = np.clip(iv_dense, 0.01, 3.0)
            n_fits += 1
        except Exception as exc:
            logger.debug("CubicSpline failed at tau=%g: %s", tau_val, exc)
            continue
        for kk, ss in zip(k_grid, iv_dense):
            smooth_pts.append((float(kk), float(tau_val), float(ss)))

    if not smooth_pts:
        raise ValueError("CubicSpline produced no usable slices.")
    arr = np.array(smooth_pts)
    KK, TT = np.meshgrid(k_grid, tau_grid, indexing='ij')
    sigma_surface = griddata(arr[:, :2], arr[:, 2], (KK, TT), method='linear')
    nm = np.isnan(sigma_surface)
    if np.any(nm):
        sigma_nn = griddata(arr[:, :2], arr[:, 2], (KK, TT), method='nearest')
        sigma_surface[nm] = sigma_nn[nm]
    sigma_surface = np.clip(sigma_surface, 0.01, 3.0)

    C = np.zeros_like(KK)
    for i in range(n_k):
        for j in range(n_tau):
            K_val = F_grid[j] * np.exp(k_grid[i])
            C[i, j] = _bs_call_safe(S0, float(K_val), r,
                                     float(sigma_surface[i, j]),
                                     float(tau_grid[j]), q)
    C = np.clip(C, 1e-6, None)
    return {
        'C_surface': C, 'k_grid': k_grid, 'tau_grid': tau_grid,
        'sigma_surface': sigma_surface, 'F_grid': F_grid,
        'method': 'cubic_spline', 'n_fits': n_fits,
    }


def loess_smooth_surface(option_df: pd.DataFrame, S0: float, r: float,
                          q: float, n_k: int = 40, n_tau: int = 15,
                          k_range: tuple[float, float] = (-0.25, 0.25),
                          frac: float = 0.5) -> dict[str, Any]:
    """Per-expiration LOESS / local-polynomial smoothing of IV in log-m.

    Tries statsmodels lowess first. Falls back to a quadratic sliding-window
    (window = max(5, frac * n)) if statsmodels is unavailable.
    """
    from scipy.interpolate import griddata
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
        have_sm = True
    except Exception:
        have_sm = False

    df = option_df.dropna(subset=['strike', 'tau', 'mid_price', 'implied_vol'])
    df = df[(df['mid_price'] > 0) & (df['implied_vol'] > 0)]
    K_obs = df['strike'].values.astype(float)
    tau_obs = df['tau'].values.astype(float)
    iv_obs = df['implied_vol'].values.astype(float)
    F_obs = float(S0) * np.exp((float(r) - float(q)) * tau_obs)
    k_obs_all = np.log(K_obs / F_obs)

    unique_taus = np.sort(np.unique(np.round(tau_obs, 6)))
    if len(unique_taus) < 3:
        raise ValueError(f"Need 3+ expirations, got {len(unique_taus)}.")
    tau_grid = np.linspace(unique_taus.min(), unique_taus.max(), int(n_tau))
    k_grid = np.linspace(float(k_range[0]), float(k_range[1]), int(n_k))
    F_grid = compute_forward_prices(S0, r, q, tau_grid)

    smooth_pts: list[tuple[float, float, float]] = []
    n_fits = 0
    method_used = 'lowess' if have_sm else 'quad_window'
    for tau_val in unique_taus:
        sel = np.isclose(tau_obs, tau_val, atol=1e-6)
        if sel.sum() < 5:
            continue
        k_slice = k_obs_all[sel]
        iv_slice = iv_obs[sel]
        order = np.argsort(k_slice)
        k_sorted = k_slice[order]
        iv_sorted = iv_slice[order]
        eval_k = np.clip(k_grid, k_sorted.min(), k_sorted.max())
        try:
            if have_sm:
                # lowess returns sorted (x, y) array. xvals param evaluates at
                # arbitrary points (modern statsmodels).
                try:
                    iv_dense = _lowess(iv_sorted, k_sorted,
                                       frac=float(frac), it=1,
                                       xvals=eval_k, return_sorted=False)
                except TypeError:
                    # Older statsmodels: smooth on observed points then interp.
                    sm = _lowess(iv_sorted, k_sorted, frac=float(frac),
                                 it=1, return_sorted=True)
                    iv_dense = np.interp(eval_k, sm[:, 0], sm[:, 1])
            else:
                # Quadratic sliding-window fallback.
                w = max(5, int(np.ceil(float(frac) * k_sorted.size)))
                w = min(w, k_sorted.size)
                iv_dense = np.empty_like(eval_k, dtype=np.float64)
                for ii, kv in enumerate(eval_k):
                    dists = np.abs(k_sorted - kv)
                    idx = np.argsort(dists)[:w]
                    xx = k_sorted[idx]; yy = iv_sorted[idx]
                    if xx.size >= 3 and (xx.max() - xx.min()) > 1e-8:
                        coefs = np.polyfit(xx, yy, deg=2)
                        iv_dense[ii] = np.polyval(coefs, kv)
                    else:
                        iv_dense[ii] = float(np.mean(yy))
            iv_dense = np.where(np.isfinite(iv_dense), iv_dense,
                                np.interp(eval_k, k_sorted, iv_sorted))
            iv_dense = np.clip(iv_dense, 0.01, 3.0)
            n_fits += 1
        except Exception as exc:
            logger.debug("LOESS failed at tau=%g: %s", tau_val, exc)
            continue
        for kk, ss in zip(k_grid, iv_dense):
            smooth_pts.append((float(kk), float(tau_val), float(ss)))

    if not smooth_pts:
        raise ValueError("LOESS produced no usable slices.")
    arr = np.array(smooth_pts)
    KK, TT = np.meshgrid(k_grid, tau_grid, indexing='ij')
    sigma_surface = griddata(arr[:, :2], arr[:, 2], (KK, TT), method='linear')
    nm = np.isnan(sigma_surface)
    if np.any(nm):
        sigma_nn = griddata(arr[:, :2], arr[:, 2], (KK, TT), method='nearest')
        sigma_surface[nm] = sigma_nn[nm]
    sigma_surface = np.clip(sigma_surface, 0.01, 3.0)

    C = np.zeros_like(KK)
    for i in range(n_k):
        for j in range(n_tau):
            K_val = F_grid[j] * np.exp(k_grid[i])
            C[i, j] = _bs_call_safe(S0, float(K_val), r,
                                     float(sigma_surface[i, j]),
                                     float(tau_grid[j]), q)
    C = np.clip(C, 1e-6, None)
    return {
        'C_surface': C, 'k_grid': k_grid, 'tau_grid': tau_grid,
        'sigma_surface': sigma_surface, 'F_grid': F_grid,
        'method': method_used, 'n_fits': n_fits,
    }


# ---------------------------------------------------------------------------
# Single-config runners
# ---------------------------------------------------------------------------

def _theta_from_C_FD(C: np.ndarray, tau_grid: np.ndarray) -> np.ndarray:
    dtau = float(tau_grid[1] - tau_grid[0])
    return np.gradient(C, dtau, axis=1, edge_order=2)


def run_config_A(snapshot: dict[str, Any], q: float,
                 n_K: int = 40, n_tau: int = 15) -> dict[str, Any]:
    """FD derivs, raw K, 5-term, FD theta, Linear."""
    S0 = snapshot['S0']; r = snapshot['r']
    surf = _raw_K_surface_from_iv(snapshot['option_df'], S0, r, q,
                                   n_K=n_K, n_tau=n_tau)
    derivs = _fd_derivs_KT(surf['V'], surf['K_grid'], surf['tau_grid'])
    fit = _fit_5term_raw_K(derivs, surf['K_grid'], surf['tau_grid'])
    return {'r2': fit['r2'], 'n_points': fit['n_points']}


def run_config_B(snapshot: dict[str, Any], q: float,
                 n_K: int = 40, n_tau: int = 15,
                 n_subsample: int = 300) -> dict[str, Any]:
    """GP (RBF) derivs, raw K, 5-term, FD theta, Linear."""
    S0 = snapshot['S0']; r = snapshot['r']
    surf = _raw_K_surface_from_iv(snapshot['option_df'], S0, r, q,
                                   n_K=n_K, n_tau=n_tau)
    derivs = _gp_derivs_KT(surf['V'], surf['K_grid'], surf['tau_grid'],
                            kernel='rbf', n_subsample=n_subsample)
    fit = _fit_5term_raw_K(derivs, surf['K_grid'], surf['tau_grid'])
    return {'r2': fit['r2'], 'n_points': fit['n_points']}


def run_config_C(snapshot: dict[str, Any], q: float,
                 n_k: int = 40, n_tau: int = 15,
                 k_range: tuple[float, float] = (-0.20, 0.20)
                 ) -> dict[str, Any]:
    """GP derivs, log-m coords, 2-term, FD theta, Linear (no SVI)."""
    S0 = snapshot['S0']; r = snapshot['r']
    surf = _logm_surface_no_svi(snapshot['option_df'], S0, r, q,
                                 n_k=n_k, n_tau=n_tau, k_range=k_range)
    # GP-fit C(k, tau) for derivs in log-m coords (use k_grid as 'S_grid').
    derivs = _gp_derivs_KT(surf['C_surface'], surf['k_grid'], surf['tau_grid'],
                            kernel='rbf', n_subsample=300)
    dCdk = derivs['dV_dK']; d2Cdk2 = derivs['d2V_dK2']
    theta_fd = derivs['dV_dtau']  # GP-smoothed dC/dtau (still "FD-like" target)
    fit = _fit_2term_logm(dCdk, d2Cdk2, theta_fd)
    return {'r2': fit['r2']}


def run_config_D(snapshot: dict[str, Any], q: float,
                 n_k: int = 40, n_tau: int = 15,
                 k_range: tuple[float, float] = (-0.20, 0.20)
                 ) -> dict[str, Any]:
    """SVI + log-m + 2-term + FD theta + Linear."""
    S0 = snapshot['S0']; r = snapshot['r']
    surf = build_logm_surface_svi(snapshot['option_df'], S0, r, q,
                                    n_k=n_k, n_tau=n_tau, k_range=k_range)
    C = surf['C_surface']; k_grid = surf['k_grid']; tau_grid = surf['tau_grid']
    dCdk, d2Cdk2 = _central_dk(C, k_grid)
    theta_fd = _theta_from_C_FD(C, tau_grid)
    fit = _fit_2term_logm(dCdk, d2Cdk2, theta_fd)
    return {'r2': fit['r2']}


def run_config_E(snapshot: dict[str, Any], q: float,
                 n_k: int = 40, n_tau: int = 15,
                 k_range: tuple[float, float] = (-0.20, 0.20)
                 ) -> dict[str, Any]:
    """SVI + log-m + 2-term + ANALYTICAL theta + Linear (the v4 pipeline)."""
    S0 = snapshot['S0']; r = snapshot['r']
    surf = build_logm_surface_svi(snapshot['option_df'], S0, r, q,
                                    n_k=n_k, n_tau=n_tau, k_range=k_range)
    C = surf['C_surface']; k_grid = surf['k_grid']; tau_grid = surf['tau_grid']
    sigma_imp = reconstruct_sigma_imp_grid(surf['svi_params'], k_grid, tau_grid)
    F_grid = compute_forward_prices(S0, r, q, tau_grid)
    K2d = np.outer(np.exp(k_grid), F_grid)
    theta = bs_theta_analytical(S0, K2d, tau_grid, sigma_imp, r, q)
    dCdk, d2Cdk2 = _central_dk(C, k_grid)
    fit = _fit_2term_logm(dCdk, d2Cdk2, theta)
    return {'r2': fit['r2']}


def run_config_F(snapshot: dict[str, Any], q: float,
                 n_k: int = 40, n_tau: int = 15,
                 k_range: tuple[float, float] = (-0.20, 0.20),
                 n_epochs: int = 3000, seed: int = 42) -> dict[str, Any]:
    """SVI + log-m + 2-term + analytical theta + [2,1] KAN."""
    from src.sindy_kan import train_kan_dupire_21
    S0 = snapshot['S0']; r = snapshot['r']
    surf = build_logm_surface_svi(snapshot['option_df'], S0, r, q,
                                    n_k=n_k, n_tau=n_tau, k_range=k_range)
    C = surf['C_surface']; k_grid = surf['k_grid']; tau_grid = surf['tau_grid']
    sigma_imp = reconstruct_sigma_imp_grid(surf['svi_params'], k_grid, tau_grid)
    F_grid = compute_forward_prices(S0, r, q, tau_grid)
    K2d = np.outer(np.exp(k_grid), F_grid)
    theta = bs_theta_analytical(S0, K2d, tau_grid, sigma_imp, r, q)
    dCdk, d2Cdk2 = _central_dk(C, k_grid)
    res = train_kan_dupire_21(dCdk, d2Cdk2, theta, n_epochs=int(n_epochs),
                                seed=int(seed))
    # F gets the in-sample / out-of-sample R^2; we report test_r2 since it
    # matches the headline PRD number (~0.795 was train_r2 from
    # sindy_kan_dupire_real.csv; both are reported in the dict).
    return {'r2': float(res['test_r2']),
            'train_r2': float(res['train_r2'])}


# ---------------------------------------------------------------------------
# Alternative-smoothing variants of config E
# ---------------------------------------------------------------------------

def _run_E_with_surface(surf: dict[str, Any], snapshot: dict[str, Any],
                          q: float) -> dict[str, Any]:
    """Config-E linear 2-term fit given a pre-built (k, tau) surface dict.

    Builds analytical theta from the surface's own ``sigma_surface``.
    """
    S0 = snapshot['S0']; r = snapshot['r']
    C = surf['C_surface']; k_grid = surf['k_grid']; tau_grid = surf['tau_grid']
    sigma_imp = surf.get('sigma_surface')
    if sigma_imp is None:
        raise ValueError("Surface dict missing sigma_surface")
    F_grid = compute_forward_prices(S0, r, q, tau_grid)
    K2d = np.outer(np.exp(k_grid), F_grid)
    theta = bs_theta_analytical(S0, K2d, tau_grid, sigma_imp, r, q)
    dCdk, d2Cdk2 = _central_dk(C, k_grid)
    fit = _fit_2term_logm(dCdk, d2Cdk2, theta)
    # local-vol estimate from c2 = 0.5 * sigma^2
    c2 = float(fit['coef'][1]) if fit['coef'].size > 1 else float('nan')
    sigma_loc = float(np.sqrt(max(0.0, 2.0 * c2)))
    return {'r2': fit['r2'], 'sigma_loc': sigma_loc}


def run_config_E_cubic_spline(snapshot: dict[str, Any], q: float,
                                n_k: int = 40, n_tau: int = 15,
                                k_range: tuple[float, float] = (-0.20, 0.20)
                                ) -> dict[str, Any]:
    surf = cubic_spline_smooth_surface(
        snapshot['option_df'], snapshot['S0'], snapshot['r'], q,
        n_k=n_k, n_tau=n_tau, k_range=k_range)
    out = _run_E_with_surface(surf, snapshot, q)
    out['method'] = surf['method']
    out['n_fits'] = surf['n_fits']
    return out


def run_config_E_loess(snapshot: dict[str, Any], q: float,
                        n_k: int = 40, n_tau: int = 15,
                        k_range: tuple[float, float] = (-0.20, 0.20),
                        frac: float = 0.5) -> dict[str, Any]:
    surf = loess_smooth_surface(
        snapshot['option_df'], snapshot['S0'], snapshot['r'], q,
        n_k=n_k, n_tau=n_tau, k_range=k_range, frac=frac)
    out = _run_E_with_surface(surf, snapshot, q)
    out['method'] = surf['method']
    out['n_fits'] = surf['n_fits']
    return out


# ---------------------------------------------------------------------------
# Top-level orchestrators
# ---------------------------------------------------------------------------

_ABLATION_DESCRIPTORS = [
    ('A', 'FD',       'raw K',          '5-term', 'FD',         'Linear'),
    ('B', 'GP (RBF)', 'raw K',          '5-term', 'FD',         'Linear'),
    ('C', 'GP',       'log-m',          '2-term', 'FD',         'Linear'),
    ('D', 'GP',       'log-m + SVI',    '2-term', 'FD',         'Linear'),
    ('E', 'GP',       'log-m + SVI',    '2-term', 'analytical', 'Linear'),
    ('F', 'GP',       'log-m + SVI',    '2-term', 'analytical', '[2,1] KAN'),
]


def run_ablation_chain(ticker: str = 'SPY', snapshot: str = '20260329',
                        cache_dir: Optional[str] = None,
                        q: Optional[float] = None,
                        n_K: int = 40, n_tau: int = 15,
                        n_k_logm: int = 40,
                        k_range: tuple[float, float] = (-0.20, 0.20),
                        save_csv: Optional[str] = None,
                        ) -> pd.DataFrame:
    """Run all 6 ablation configs and return the summary DataFrame."""
    snap = load_snapshot(ticker, snapshot, cache_dir=cache_dir)
    if q is None:
        try:
            q = float(get_dividend_yield(ticker))
        except Exception:
            q = 0.013 if ticker == 'SPY' else 0.0
    q = float(q)

    runners = {
        'A': lambda: run_config_A(snap, q, n_K=n_K, n_tau=n_tau),
        'B': lambda: run_config_B(snap, q, n_K=n_K, n_tau=n_tau),
        'C': lambda: run_config_C(snap, q, n_k=n_k_logm, n_tau=n_tau,
                                   k_range=k_range),
        'D': lambda: run_config_D(snap, q, n_k=n_k_logm, n_tau=n_tau,
                                   k_range=k_range),
        'E': lambda: run_config_E(snap, q, n_k=n_k_logm, n_tau=n_tau,
                                   k_range=k_range),
        'F': lambda: run_config_F(snap, q, n_k=n_k_logm, n_tau=n_tau,
                                   k_range=k_range),
    }
    rows = []
    for cfg, derivs, coords, lib, theta_src, model in _ABLATION_DESCRIPTORS:
        try:
            res = runners[cfg]()
            r2 = float(res.get('r2', float('nan')))
        except Exception as exc:
            logger.warning("Config %s failed: %s", cfg, exc)
            r2 = float('nan')
        rows.append({
            'config': cfg,
            'derivatives': derivs,
            'coordinates': coords,
            'library': lib,
            'theta_source': theta_src,
            'model': model,
            'R2_SPY': r2,
        })
    df = pd.DataFrame(rows)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        df.to_csv(save_csv, index=False)
        logger.info("Wrote ablation chain to %s", save_csv)
    return df


def run_smoothing_comparison(ticker: str = 'SPY', snapshot: str = '20260329',
                              cache_dir: Optional[str] = None,
                              q: Optional[float] = None,
                              n_k: int = 40, n_tau: int = 15,
                              k_range: tuple[float, float] = (-0.20, 0.20),
                              save_csv: Optional[str] = None,
                              ) -> pd.DataFrame:
    """SVI vs cubic-spline vs LOESS, all inside the config-E pipeline."""
    snap = load_snapshot(ticker, snapshot, cache_dir=cache_dir)
    if q is None:
        try:
            q = float(get_dividend_yield(ticker))
        except Exception:
            q = 0.013 if ticker == 'SPY' else 0.0
    q = float(q)

    rows = []
    # SVI (the actual config E)
    try:
        S0 = snap['S0']; r = snap['r']
        surf = build_logm_surface_svi(snap['option_df'], S0, r, q,
                                       n_k=n_k, n_tau=n_tau, k_range=k_range)
        sigma_imp = reconstruct_sigma_imp_grid(
            surf['svi_params'], surf['k_grid'], surf['tau_grid'])
        surf['sigma_surface'] = sigma_imp  # config-E expects sigma_surface
        out = _run_E_with_surface(surf, snap, q)
        sigma_summary = float(np.nanmedian(sigma_imp))
        notes = (f"n_svi_success={surf.get('n_svi_success', 0)}, "
                 f"n_svi_fallback={surf.get('n_svi_fallback', 0)}")
        rows.append({'method': 'SVI', 'R2_SPY': float(out['r2']),
                     'sigma_loc_summary': float(out['sigma_loc']),
                     'sigma_imp_median': sigma_summary, 'notes': notes})
    except Exception as exc:
        logger.warning("SVI smoothing failed: %s", exc)
        rows.append({'method': 'SVI', 'R2_SPY': float('nan'),
                     'sigma_loc_summary': float('nan'),
                     'sigma_imp_median': float('nan'),
                     'notes': f"error: {str(exc)[:80]}"})

    # Cubic spline
    try:
        surf_cs = cubic_spline_smooth_surface(
            snap['option_df'], snap['S0'], snap['r'], q,
            n_k=n_k, n_tau=n_tau, k_range=k_range)
        out = _run_E_with_surface(surf_cs, snap, q)
        sigma_summary = float(np.nanmedian(surf_cs['sigma_surface']))
        rows.append({'method': 'cubic_spline', 'R2_SPY': float(out['r2']),
                     'sigma_loc_summary': float(out['sigma_loc']),
                     'sigma_imp_median': sigma_summary,
                     'notes': f"n_fits={surf_cs['n_fits']}"})
    except Exception as exc:
        logger.warning("Cubic-spline smoothing failed: %s", exc)
        rows.append({'method': 'cubic_spline', 'R2_SPY': float('nan'),
                     'sigma_loc_summary': float('nan'),
                     'sigma_imp_median': float('nan'),
                     'notes': f"error: {str(exc)[:80]}"})

    # LOESS
    try:
        surf_lo = loess_smooth_surface(
            snap['option_df'], snap['S0'], snap['r'], q,
            n_k=n_k, n_tau=n_tau, k_range=k_range)
        out = _run_E_with_surface(surf_lo, snap, q)
        sigma_summary = float(np.nanmedian(surf_lo['sigma_surface']))
        rows.append({'method': 'LOESS', 'R2_SPY': float(out['r2']),
                     'sigma_loc_summary': float(out['sigma_loc']),
                     'sigma_imp_median': sigma_summary,
                     'notes': f"impl={surf_lo['method']}, n_fits={surf_lo['n_fits']}"})
    except Exception as exc:
        logger.warning("LOESS smoothing failed: %s", exc)
        rows.append({'method': 'LOESS', 'R2_SPY': float('nan'),
                     'sigma_loc_summary': float('nan'),
                     'sigma_imp_median': float('nan'),
                     'notes': f"error: {str(exc)[:80]}"})

    df = pd.DataFrame(rows)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        df.to_csv(save_csv, index=False)
        logger.info("Wrote smoothing comparison to %s", save_csv)
    return df
