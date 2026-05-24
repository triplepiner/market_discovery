"""
Real-data pipeline v3: targeted improvements on top of v2.

Improvements:
  1. Windowed 2-term Dupire applied to the v2 SVI-smoothed log-moneyness surface.
  2. Per-expiration sigma extraction yielding a term structure sigma_loc(tau).
  3. Quadratic sigma^2(k) Dupire model -- allows skew/curvature so a global R^2
     can be meaningful even when a single constant sigma cannot fit a smile.
  4. Sigma-method comparison: SINDy vs mean / median / vega-weighted /
     volume-weighted IVs vs ATM IV truth.
  5. Bootstrap CI for the recovered sigma from the SVI + 2-term Dupire stack.

All public functions are wrapped in try/except where they touch external state
or do nontrivial numerics. The synthetic and ticker-level wrappers are
deterministic for fixed seeds.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.real_data_v2 import (
    build_logm_surface_svi,
    compute_forward_prices,
    compute_liquidity_weights,
    compute_log_moneyness,
    dupire_logm_2term,
)
from src.utils import set_all_seeds, setup_logging

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _central_dk(C: np.ndarray, k_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (dC/dk, d2C/dk2) computed with centered FD on uniform k grid."""
    dk = float(k_grid[1] - k_grid[0])
    dCdk = np.gradient(C, dk, axis=0, edge_order=2)
    d2Cdk2 = np.gradient(dCdk, dk, axis=0, edge_order=2)
    return dCdk, d2Cdk2


def _central_dtau(C: np.ndarray, tau_grid: np.ndarray) -> np.ndarray:
    """Return dC/dtau with centered FD on (possibly non-uniform) tau grid."""
    if len(tau_grid) < 2:
        return np.zeros_like(C)
    # Use np.gradient with explicit coordinates -> handles non-uniform spacing.
    return np.gradient(C, tau_grid, axis=1, edge_order=2)


def _r2_from(target: np.ndarray, pred: np.ndarray) -> float:
    """Unweighted R^2."""
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    if ss_tot < 1e-30:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Improvement 1: windowed 2-term Dupire on the SVI surface
# ---------------------------------------------------------------------------

def windowed_2term_dupire_v2(C_surface: np.ndarray, k_grid: np.ndarray,
                             tau_grid: np.ndarray, window_size: int = 8,
                             stride: int = 3, min_r2: float = 0.5,
                             sigma_bounds: tuple[float, float] = (0.01, 2.0),
                             weights: Optional[np.ndarray] = None,
                             ) -> dict[str, Any]:
    """Slide windows over the SVI-smoothed log-moneyness surface.

    At each (i, j) window the local 2-term Dupire OLS

        dC/dtau = c1 * dC/dk + c2 * d2C/dk2

    is solved and sigma_local = sqrt(2 * c2) extracted. A window is "valid"
    if its OLS R^2 >= ``min_r2`` and sigma_local lies within ``sigma_bounds``.

    Parameters
    ----------
    C_surface : ndarray, shape (n_k, n_tau)
    k_grid, tau_grid : ndarray
    window_size, stride : int
    min_r2 : float
    sigma_bounds : (float, float)
    weights : ndarray, shape (n_k, n_tau), optional

    Returns
    -------
    dict
    """
    C = np.asarray(C_surface, dtype=np.float64)
    n_k, n_tau = C.shape

    if window_size > n_k or window_size > n_tau:
        raise ValueError(
            f"window_size {window_size} too large for surface {C.shape}"
        )

    i_starts = list(range(0, n_k - window_size + 1, max(1, int(stride))))
    j_starts = list(range(0, n_tau - window_size + 1, max(1, int(stride))))
    if not i_starts:
        i_starts = [0]
    if not j_starts:
        j_starts = [0]

    n_iw = len(i_starts)
    n_jw = len(j_starts)
    sigma_grid = np.full((n_iw, n_jw), np.nan)
    r2_grid = np.full((n_iw, n_jw), np.nan)
    drift_grid = np.full((n_iw, n_jw), np.nan)
    valid_grid = np.zeros((n_iw, n_jw), dtype=bool)

    k_centers = np.array([
        0.5 * (k_grid[i] + k_grid[i + window_size - 1]) for i in i_starts
    ])
    tau_centers = np.array([
        0.5 * (tau_grid[j] + tau_grid[j + window_size - 1]) for j in j_starts
    ])

    n_total = n_iw * n_jw
    n_valid = 0
    sig_lo, sig_hi = float(sigma_bounds[0]), float(sigma_bounds[1])

    for ii, i in enumerate(i_starts):
        for jj, j in enumerate(j_starts):
            i_end = i + window_size
            j_end = j + window_size
            C_win = C[i:i_end, j:j_end]
            k_win = k_grid[i:i_end]
            tau_win = tau_grid[j:j_end]
            w_win = weights[i:i_end, j:j_end] if weights is not None else None

            try:
                res = dupire_logm_2term(C_win, k_win, tau_win, weights=w_win)
            except Exception as exc:
                logger.debug("window (%d,%d) regression failed: %s",
                             ii, jj, exc)
                continue

            sigma_w = float(res.get('sigma_loc_discovered', np.nan))
            r2_w = float(res.get('r2_score', np.nan))
            drift_w = float(res.get('drift_discovered', np.nan))

            if not np.isfinite(sigma_w):
                continue

            sigma_grid[ii, jj] = sigma_w
            r2_grid[ii, jj] = r2_w
            drift_grid[ii, jj] = drift_w

            if (np.isfinite(r2_w) and r2_w >= min_r2
                    and sig_lo <= sigma_w <= sig_hi):
                valid_grid[ii, jj] = True
                n_valid += 1

    if n_valid > 0:
        valid_sigmas = sigma_grid[valid_grid]
        sigma_median = float(np.nanmedian(valid_sigmas))
        sigma_mean = float(np.nanmean(valid_sigmas))
    else:
        sigma_median = float('nan')
        sigma_mean = float('nan')

    return {
        'k_centers': k_centers,
        'tau_centers': tau_centers,
        'sigma_local_grid': sigma_grid,
        'r2_grid': r2_grid,
        'drift_grid': drift_grid,
        'is_valid_grid': valid_grid,
        'n_valid': int(n_valid),
        'n_total': int(n_total),
        'sigma_median': sigma_median,
        'sigma_mean': sigma_mean,
        'window_size': int(window_size),
        'stride': int(stride),
        'min_r2': float(min_r2),
        'sigma_bounds': (sig_lo, sig_hi),
    }


def windowed_2term_dupire_v2_on_ticker(option_data: dict[str, Any], ticker: str,
                                       window_size: int = 8, stride: int = 3,
                                       min_r2: float = 0.5,
                                       sigma_bounds: tuple[float, float] = (0.01, 2.0),
                                       use_weights: bool = True,
                                       n_k: int = 40,
                                       k_range: tuple[float, float] = (-0.25, 0.25),
                                       q: Optional[float] = None,
                                       ) -> dict[str, Any]:
    """Build the SVI surface for a ticker and run :func:`windowed_2term_dupire_v2`."""
    out: dict[str, Any] = {'ticker': ticker, 'errors': {}}
    try:
        S0 = float(option_data['S0'])
        r = float(option_data['r'])
        df = option_data['option_df'].copy()
        if q is None:
            try:
                from src.real_data_v2 import get_dividend_yield
                q = get_dividend_yield(ticker)
            except Exception:
                q = 0.0
        out['q'] = float(q)
        surface = build_logm_surface_svi(df, S0, r, q, n_k=n_k, k_range=k_range)
        weights = None
        if use_weights:
            try:
                weights = compute_liquidity_weights(
                    df, surface['k_grid'], surface['tau_grid'], q=q, S0=S0,
                )
            except Exception as exc:
                out['errors']['weights'] = str(exc)
                weights = None
        res = windowed_2term_dupire_v2(
            surface['C_surface'], surface['k_grid'], surface['tau_grid'],
            window_size=window_size, stride=stride, min_r2=min_r2,
            sigma_bounds=sigma_bounds, weights=weights,
        )
        res['surface_method'] = surface.get('method', 'svi_smoothed')
        out['result'] = res
    except Exception as exc:
        logger.warning("%s: windowed_2term_dupire_v2_on_ticker failed: %s",
                       ticker, exc)
        out['errors']['top'] = str(exc)
        out['result'] = None
    return out


# ---------------------------------------------------------------------------
# Improvement 2: per-expiration sigma extraction
# ---------------------------------------------------------------------------

def per_expiration_sigma(C_grid: np.ndarray, k_grid: np.ndarray,
                         tau_grid: np.ndarray,
                         sigma_bounds: tuple[float, float] = (0.01, 2.0),
                         ) -> pd.DataFrame:
    """Extract sigma_loc(tau_i) for each interior expiration via 2-term OLS.

    For each interior tau index i (1..n_tau-2) we compute dC/dtau against
    neighbouring expirations, then regress over the k-axis:

        dC/dtau(k) = c1 * dC/dk(k) + c2 * d2C/dk2(k)

    and report sigma_loc = sqrt(2*c2) if c2 > 0.

    Returns
    -------
    DataFrame with columns: tau, sigma_loc, drift, r2, n_used.
    """
    C = np.asarray(C_grid, dtype=np.float64)
    k = np.asarray(k_grid, dtype=np.float64)
    tau = np.asarray(tau_grid, dtype=np.float64)
    n_k, n_tau = C.shape
    if n_tau < 3:
        raise ValueError(f"Need >=3 expirations, got {n_tau}.")

    dCdk, d2Cdk2 = _central_dk(C, k)
    dCdtau = _central_dtau(C, tau)

    sig_lo, sig_hi = float(sigma_bounds[0]), float(sigma_bounds[1])
    rows: list[dict[str, Any]] = []
    # Skip first/last expirations (forward/backward FD biased).
    for i in range(1, n_tau - 1):
        try:
            tgt = dCdtau[:, i]
            lib = np.column_stack([dCdk[:, i], d2Cdk2[:, i]])
            mask = np.isfinite(tgt) & np.all(np.isfinite(lib), axis=1)
            if mask.sum() < 4:
                rows.append({'tau': float(tau[i]), 'sigma_loc': float('nan'),
                             'drift': float('nan'), 'r2': float('nan'),
                             'n_used': int(mask.sum())})
                continue
            A = lib[mask]
            b = tgt[mask]
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            c1, c2 = float(coef[0]), float(coef[1])
            pred = A @ coef
            r2 = _r2_from(b, pred)
            sigma_loc = float(np.sqrt(max(0.0, 2.0 * c2)))
            if not (sig_lo <= sigma_loc <= sig_hi):
                # Keep value but it's outside reasonable bounds (still reported).
                pass
            rows.append({
                'tau': float(tau[i]),
                'sigma_loc': sigma_loc,
                'drift': c1,
                'r2': float(r2),
                'n_used': int(mask.sum()),
            })
        except Exception as exc:
            logger.debug("per_expiration_sigma failed at tau=%.4f: %s",
                         tau[i], exc)
            rows.append({'tau': float(tau[i]), 'sigma_loc': float('nan'),
                         'drift': float('nan'), 'r2': float('nan'),
                         'n_used': 0})

    return pd.DataFrame(rows)


def per_expiration_sigma_on_ticker(option_data: dict[str, Any], ticker: str,
                                   n_k: int = 40,
                                   k_range: tuple[float, float] = (-0.25, 0.25),
                                   q: Optional[float] = None,
                                   ) -> dict[str, Any]:
    """Build SVI surface for a ticker then extract term structure sigma_loc(tau)."""
    out: dict[str, Any] = {'ticker': ticker, 'errors': {}}
    try:
        S0 = float(option_data['S0'])
        r = float(option_data['r'])
        df = option_data['option_df'].copy()
        if q is None:
            try:
                from src.real_data_v2 import get_dividend_yield
                q = get_dividend_yield(ticker)
            except Exception:
                q = 0.0
        out['q'] = float(q)
        surface = build_logm_surface_svi(df, S0, r, q, n_k=n_k, k_range=k_range)
        term_df = per_expiration_sigma(
            surface['C_surface'], surface['k_grid'], surface['tau_grid'],
        )
        # Market avg IV per expiration (simple mean of options at that expiry).
        market_rows: list[dict[str, Any]] = []
        if 'tau' in df.columns and 'implied_vol' in df.columns:
            for tau_val, sub in df.groupby(np.round(df['tau'].values, 6)):
                ivs = sub['implied_vol'].values
                ivs = ivs[np.isfinite(ivs) & (ivs > 0)]
                if len(ivs) == 0:
                    continue
                market_rows.append({
                    'tau': float(tau_val),
                    'market_iv_mean': float(np.mean(ivs)),
                    'market_iv_median': float(np.median(ivs)),
                    'n_options': int(len(ivs)),
                })
        market_df = pd.DataFrame(market_rows)
        out['term_structure'] = term_df
        out['market_iv_per_tau'] = market_df
    except Exception as exc:
        logger.warning("%s: per_expiration_sigma_on_ticker failed: %s",
                       ticker, exc)
        out['errors']['top'] = str(exc)
        out['term_structure'] = None
        out['market_iv_per_tau'] = None
    return out


# ---------------------------------------------------------------------------
# Improvement 3: quadratic sigma^2(k) Dupire model
# ---------------------------------------------------------------------------

def quadratic_dupire_logm(C_surface: np.ndarray, k_grid: np.ndarray,
                          tau_grid: np.ndarray,
                          weights: Optional[np.ndarray] = None,
                          k_eval: Optional[list[float]] = None,
                          ) -> dict[str, Any]:
    """Fit Dupire with sigma^2(k) = alpha + beta*k + gamma*k^2.

    The PDE becomes::

        dC/dtau = 0.5*alpha*d2C/dk2 + 0.5*beta*k*d2C/dk2
                  + 0.5*gamma*k^2*d2C/dk2 + drift*dC/dk

    Library columns: [dC/dk, d2C/dk2, k*d2C/dk2, k^2*d2C/dk2].
    Coefficients map back to (drift, 0.5*alpha, 0.5*beta, 0.5*gamma).
    """
    C = np.asarray(C_surface, dtype=np.float64)
    k = np.asarray(k_grid, dtype=np.float64)
    tau = np.asarray(tau_grid, dtype=np.float64)
    n_k, n_tau = C.shape
    if n_tau < 2:
        raise ValueError(f"Need >=2 expirations, got {n_tau}.")

    dCdk, d2Cdk2 = _central_dk(C, k)
    dCdtau = _central_dtau(C, tau)

    KK = np.tile(k.reshape(-1, 1), (1, n_tau))

    lib = np.column_stack([
        dCdk.ravel(),
        d2Cdk2.ravel(),
        (KK * d2Cdk2).ravel(),
        (KK * KK * d2Cdk2).ravel(),
    ])
    tgt = dCdtau.ravel()
    mask = np.all(np.isfinite(lib), axis=1) & np.isfinite(tgt)
    lib = lib[mask]
    tgt = tgt[mask]

    if weights is not None:
        w = np.clip(np.asarray(weights, dtype=np.float64).ravel()[mask],
                    0.0, None)
        sw = np.sqrt(w)
        A = lib * sw[:, None]
        b = tgt * sw
    else:
        A = lib
        b = tgt

    cond = float(np.linalg.cond(lib))
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    drift = float(coef[0])
    half_alpha = float(coef[1])
    half_beta = float(coef[2])
    half_gamma = float(coef[3])
    alpha = 2.0 * half_alpha
    beta = 2.0 * half_beta
    gamma = 2.0 * half_gamma

    pred = lib @ coef
    r2 = _r2_from(tgt, pred)

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


def quadratic_dupire_on_ticker(option_data: dict[str, Any], ticker: str,
                               n_k: int = 40,
                               k_range: tuple[float, float] = (-0.25, 0.25),
                               use_weights: bool = True,
                               q: Optional[float] = None,
                               ) -> dict[str, Any]:
    """Build SVI surface + liquidity weights, then fit quadratic sigma^2(k) Dupire."""
    out: dict[str, Any] = {'ticker': ticker, 'errors': {}}
    try:
        S0 = float(option_data['S0'])
        r = float(option_data['r'])
        df = option_data['option_df'].copy()
        if q is None:
            try:
                from src.real_data_v2 import get_dividend_yield
                q = get_dividend_yield(ticker)
            except Exception:
                q = 0.0
        out['q'] = float(q)
        surface = build_logm_surface_svi(df, S0, r, q, n_k=n_k, k_range=k_range)
        weights = None
        if use_weights:
            try:
                weights = compute_liquidity_weights(
                    df, surface['k_grid'], surface['tau_grid'], q=q, S0=S0,
                )
            except Exception as exc:
                out['errors']['weights'] = str(exc)
                weights = None

        res = quadratic_dupire_logm(
            surface['C_surface'], surface['k_grid'], surface['tau_grid'],
            weights=weights,
        )

        # ATM IV from raw option data (|k|<0.05).
        try:
            F_obs = float(S0) * np.exp((r - q) * df['tau'].values)
            k_obs = compute_log_moneyness(df['strike'].values, F_obs)
            atm_mask = np.abs(k_obs) < 0.05
            ivs = df['implied_vol'].values
            mask = atm_mask & np.isfinite(ivs) & (ivs > 0)
            atm_iv = float(np.mean(ivs[mask])) if mask.sum() > 0 else float('nan')
        except Exception as exc:
            out['errors']['atm_iv'] = str(exc)
            atm_iv = float('nan')

        res['atm_iv_market'] = atm_iv
        sig_at_zero = res['sigma_at_k_dict'].get(0.0, float('nan'))
        if np.isfinite(sig_at_zero) and np.isfinite(atm_iv) and atm_iv > 0:
            res['sigma_at_zero_rel_err'] = float(abs(sig_at_zero - atm_iv) / atm_iv)
        else:
            res['sigma_at_zero_rel_err'] = float('nan')

        out['result'] = res
    except Exception as exc:
        logger.warning("%s: quadratic_dupire_on_ticker failed: %s", ticker, exc)
        out['errors']['top'] = str(exc)
        out['result'] = None
    return out


# ---------------------------------------------------------------------------
# Improvement 4: sigma method comparison
# ---------------------------------------------------------------------------

def _vega_per_option(S0: float, K: np.ndarray, r: float, q: float,
                     sigma: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """BS vega = S0*exp(-q*tau)*sqrt(tau)*phi(d1)."""
    K = np.asarray(K, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    safe_sig = np.where(sigma > 1e-8, sigma, 1e-8)
    safe_tau = np.where(tau > 1e-8, tau, 1e-8)
    F = S0 * np.exp((r - q) * safe_tau)
    d1 = (np.log(F / K) + 0.5 * safe_sig ** 2 * safe_tau) / (safe_sig * np.sqrt(safe_tau))
    phi = norm.pdf(d1)
    vega = S0 * np.exp(-q * safe_tau) * np.sqrt(safe_tau) * phi
    return vega


def compare_sigma_methods(option_data: dict[str, Any], ticker: str,
                          sigma_sindy: float,
                          q: Optional[float] = None,
                          ) -> pd.DataFrame:
    """Compare SINDy sigma against IV averages and ATM IV truth.

    Returns a DataFrame with one row per method.
    """
    df = option_data['option_df'].copy()
    S0 = float(option_data['S0'])
    r = float(option_data['r'])
    if q is None:
        try:
            from src.real_data_v2 import get_dividend_yield
            q = get_dividend_yield(ticker)
        except Exception:
            q = 0.0

    ivs = df['implied_vol'].values.astype(np.float64)
    K = df['strike'].values.astype(np.float64)
    tau = df['tau'].values.astype(np.float64)
    volume = (df['volume'].values.astype(np.float64) if 'volume' in df.columns
              else np.ones(len(df)))

    mask = np.isfinite(ivs) & (ivs > 0) & np.isfinite(tau) & (tau > 0)
    ivs_ok = ivs[mask]
    K_ok = K[mask]
    tau_ok = tau[mask]
    vol_ok = np.clip(volume[mask], 0.0, None)

    # ATM truth: mean of IVs with |k|<0.05.
    try:
        F_ok = S0 * np.exp((r - q) * tau_ok)
        k_ok = compute_log_moneyness(K_ok, F_ok)
        atm_mask = np.abs(k_ok) < 0.05
        atm_iv = float(np.mean(ivs_ok[atm_mask])) if atm_mask.sum() > 0 else float('nan')
    except Exception:
        atm_iv = float('nan')

    sigma_mean = float(np.mean(ivs_ok))
    sigma_median = float(np.median(ivs_ok))

    # Vega weights.
    try:
        vega = _vega_per_option(S0, K_ok, r, q, ivs_ok, tau_ok)
        vega = np.clip(vega, 0.0, None)
        w_vega = vega.sum()
        sigma_vega = float(np.sum(ivs_ok * vega) / w_vega) if w_vega > 0 else float('nan')
    except Exception:
        sigma_vega = float('nan')

    # Volume weights.
    try:
        w_vol = vol_ok.sum()
        sigma_vol = float(np.sum(ivs_ok * vol_ok) / w_vol) if w_vol > 0 else float('nan')
    except Exception:
        sigma_vol = float('nan')

    rows = []
    for method, sigma_val in [
        ('sindy', float(sigma_sindy)),
        ('mean', sigma_mean),
        ('median', sigma_median),
        ('vega_weighted', sigma_vega),
        ('volume_weighted', sigma_vol),
        ('atm_truth', atm_iv),
    ]:
        if np.isfinite(atm_iv) and atm_iv > 0 and np.isfinite(sigma_val):
            abs_diff = abs(sigma_val - atm_iv)
            rel_err = abs_diff / atm_iv
        else:
            abs_diff = float('nan')
            rel_err = float('nan')
        rows.append({
            'method': method,
            'sigma': float(sigma_val),
            'abs_diff_to_atm': float(abs_diff),
            'rel_err': float(rel_err),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Improvement 5: bootstrap CIs on sigma_recovered
# ---------------------------------------------------------------------------

def bootstrap_sigma_v2(option_data: dict[str, Any], ticker: str,
                       n_bootstrap: int = 100, seed: int = 42,
                       q: Optional[float] = None,
                       n_k: int = 40,
                       k_range: tuple[float, float] = (-0.25, 0.25),
                       ) -> dict[str, Any]:
    """Resample the option chain with replacement and recompute sigma each time.

    Returns dict with mean, std, 2.5%/97.5% CI bounds, and success counts.
    """
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
            res = dupire_logm_2term(
                surface['C_surface'], surface['k_grid'], surface['tau_grid'],
            )
            sigma_b = float(res.get('sigma_loc_discovered', float('nan')))
            if np.isfinite(sigma_b) and sigma_b > 0:
                sigmas.append(sigma_b)
        except Exception as exc:
            logger.debug("bootstrap iter %d failed: %s", b, exc)
            continue

    n_success = len(sigmas)
    n_total = int(n_bootstrap)
    if n_success == 0:
        return {
            'sigma_mean': float('nan'),
            'sigma_std': float('nan'),
            'ci_low': float('nan'),
            'ci_high': float('nan'),
            'n_success': 0,
            'n_total': n_total,
            'sigmas': [],
        }
    arr = np.asarray(sigmas, dtype=np.float64)
    sigma_mean = float(np.mean(arr))
    sigma_std = float(np.std(arr, ddof=1)) if n_success > 1 else 0.0
    ci_low = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    return {
        'sigma_mean': sigma_mean,
        'sigma_std': sigma_std,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'n_success': n_success,
        'n_total': n_total,
        'sigmas': sigmas,
    }
