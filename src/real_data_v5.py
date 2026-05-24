"""
Real-data pipeline v5: final-polish wiring of the three remaining experiments
to the analytical-theta Dupire target introduced in v4.

The v4 module replaced finite-difference theta with the closed-form
Black-Scholes theta evaluated pointwise from the SVI-smoothed implied vol
surface. That single change lifted the global 2-term Dupire R^2 from ~-0.03 to
0.55-0.63 across SPY/QQQ/AAPL/MSFT. Three experiments in the v3/v4 stack were
still using FD-theta:

  * Windowed 2-term Dupire (``windowed_2term_dupire_v2`` from v3 -- only 2/55
    valid windows on SPY when run on FD theta).
  * Quadratic sigma^2(k) Dupire (``quadratic_dupire_logm`` from v3 -- baseline
    used FD-theta; v4's ``quadratic_dupire_analytical_theta`` already lifted
    it, this wrapper just exposes it under v5 naming with an explicit sign
    convention note).
  * Per-expiration sigma extraction (``per_expiration_sigma`` from v3 -- FD
    dC/dtau collapses 6/21 expirations on SPY into sigma_loc=0).

Sign convention note (see Fix 2 diagnostic in the report):
    The diagnostic confirmed that with K = F(tau)*exp(k) the standard
    convention k<0 <=> K<F (downside) holds, and the analytical BS theta is
    positive everywhere. The library and PDE form used by both v3's
    ``quadratic_dupire_logm`` and v4's ``quadratic_dupire_analytical_theta``
    is ``dC/dtau = drift * dC/dk + (0.5*alpha + 0.5*beta*k + 0.5*gamma*k^2) *
    d2C/dk2`` with sigma^2(k) = alpha + beta*k + gamma*k^2. On a synthetic
    surface with sigma^2(k) = 0.04 - 0.01*k + 0.05*k^2 the quadratic v4
    function recovers beta ~ -0.014, gamma ~ +0.098 -- i.e. the textbook
    equity-skew signs are correctly recovered, so no sign flip is applied.
    The empirical beta>0/gamma<0 reported by v4 on SPY/QQQ/AAPL is a genuine
    finding (the SVI-smoothed surface within k in [-0.25, 0.25] does not
    exhibit a pure textbook equity-skew shape).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.real_data_v2 import (
    build_logm_surface_svi,
    compute_forward_prices,
    compute_liquidity_weights,
)
from src.real_data_v4 import (
    _central_dk,
    _r2_from,
    bs_theta_analytical,
    reconstruct_sigma_imp_grid,
)
from src.utils import set_all_seeds, setup_logging

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Fix 1 -- windowed 2-term Dupire with analytical theta
# ---------------------------------------------------------------------------

def windowed_2term_dupire_v4(C_surface: np.ndarray,
                             sigma_imp_surface: np.ndarray,
                             k_grid: np.ndarray,
                             tau_grid: np.ndarray,
                             S0: float, r: float, q: float = 0.0,
                             window_size: int = 8, stride: int = 3,
                             min_r2: float = 0.5,
                             sigma_bounds: tuple[float, float] = (0.01, 2.0),
                             ) -> dict[str, Any]:
    """Slide windows over the SVI-smoothed log-moneyness surface.

    At each window we:
      1. Extract local C, sigma_imp, k_local, tau_local.
      2. Compute analytical theta on the LOCAL grid via
         :func:`bs_theta_analytical`.
      3. Run 2-term OLS with theta as the target on the local k-derivatives.

    A window is valid iff R^2 >= ``min_r2`` and sigma_local lies in
    ``sigma_bounds``.

    Parameters
    ----------
    C_surface, sigma_imp_surface : ndarray, shape (n_k, n_tau)
    k_grid : ndarray, shape (n_k,)
    tau_grid : ndarray, shape (n_tau,)
    S0, r : float
    q : float, default 0
    window_size, stride : int
    min_r2 : float
    sigma_bounds : (float, float)

    Returns
    -------
    dict
        ``k_centers``, ``tau_centers``, ``sigma_local_grid``, ``r2_grid``,
        ``drift_grid``, ``is_valid_grid``, ``n_valid``, ``n_total``,
        ``sigma_median``, ``sigma_mean``, ``window_size``, ``stride``,
        ``min_r2``, ``sigma_bounds``.
    """
    C = np.asarray(C_surface, dtype=np.float64)
    sig = np.asarray(sigma_imp_surface, dtype=np.float64)
    k_grid = np.asarray(k_grid, dtype=np.float64)
    tau_grid = np.asarray(tau_grid, dtype=np.float64)
    n_k, n_tau = C.shape
    if sig.shape != C.shape:
        raise ValueError(
            f"sigma_imp shape {sig.shape} != C shape {C.shape}"
        )

    window_size = int(window_size)
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

    sig_lo, sig_hi = float(sigma_bounds[0]), float(sigma_bounds[1])
    n_total = n_iw * n_jw
    n_valid = 0

    for ii, i in enumerate(i_starts):
        for jj, j in enumerate(j_starts):
            i_end = i + window_size
            j_end = j + window_size
            try:
                C_win = C[i:i_end, j:j_end]
                sig_win = sig[i:i_end, j:j_end]
                k_win = k_grid[i:i_end]
                tau_win = tau_grid[j:j_end]

                # Local strikes K(tau) = F(tau) * exp(k)
                F_win = compute_forward_prices(S0, r, q, tau_win)
                K_win = np.outer(np.exp(k_win), F_win)
                theta_win = bs_theta_analytical(
                    S0, K_win, tau_win, sig_win, r, q,
                )

                dCdk, d2Cdk2 = _central_dk(C_win, k_win)
                lib = np.column_stack(
                    [dCdk.ravel(), d2Cdk2.ravel()]
                )
                tgt = theta_win.ravel()
                mask = (np.all(np.isfinite(lib), axis=1)
                        & np.isfinite(tgt))
                if mask.sum() < 4:
                    continue
                A = lib[mask]
                b = tgt[mask]
                coef, *_ = np.linalg.lstsq(A, b, rcond=None)
                c1 = float(coef[0])
                c2 = float(coef[1])
                pred = A @ coef
                r2_w = _r2_from(b, pred)
                sigma_w = float(np.sqrt(max(0.0, 2.0 * c2)))
            except Exception as exc:
                logger.debug(
                    "windowed v4 window (%d,%d) failed: %s", ii, jj, exc,
                )
                continue

            sigma_grid[ii, jj] = sigma_w
            r2_grid[ii, jj] = r2_w
            drift_grid[ii, jj] = c1
            if (np.isfinite(sigma_w) and np.isfinite(r2_w)
                    and r2_w >= float(min_r2)
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


def windowed_v4_on_ticker(option_data: dict[str, Any], ticker: str,
                          window_size: int = 8, stride: int = 3,
                          min_r2: float = 0.5,
                          sigma_bounds: tuple[float, float] = (0.01, 2.0),
                          n_k: int = 40,
                          k_range: tuple[float, float] = (-0.25, 0.25),
                          q: Optional[float] = None,
                          ) -> dict[str, Any]:
    """Build SVI surface + reconstruct sigma_imp grid + windowed v4 Dupire."""
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
        sigma_imp = reconstruct_sigma_imp_grid(
            surface['svi_params'], surface['k_grid'], surface['tau_grid'],
        )
        res = windowed_2term_dupire_v4(
            surface['C_surface'], sigma_imp,
            surface['k_grid'], surface['tau_grid'],
            S0, r, q,
            window_size=window_size, stride=stride, min_r2=min_r2,
            sigma_bounds=sigma_bounds,
        )
        res['surface_method'] = surface.get('method', 'svi_smoothed')
        out['result'] = res
    except Exception as exc:
        logger.warning("%s: windowed_v4_on_ticker failed: %s", ticker, exc)
        out['errors']['top'] = str(exc)
        out['result'] = None
    return out


# ---------------------------------------------------------------------------
# Fix 2 -- quadratic sigma^2(k) Dupire with analytical theta
# ---------------------------------------------------------------------------

def quadratic_dupire_v4(C_surface: np.ndarray,
                        sigma_imp_surface: np.ndarray,
                        k_grid: np.ndarray, tau_grid: np.ndarray,
                        S0: float, r: float, q: float = 0.0,
                        weights: Optional[np.ndarray] = None,
                        k_eval: Optional[list[float]] = None,
                        ) -> dict[str, Any]:
    """Quadratic sigma^2(k) Dupire fit with the analytical-theta target.

    Library matches :func:`quadratic_dupire_logm` from v3:

        [dC/dk, d2C/dk2, k * d2C/dk2, k^2 * d2C/dk2]

    The target is the analytical BS theta evaluated pointwise from the
    SVI-smoothed implied vol surface (NOT the FD ``dC/dtau``).

    The PDE form:

        dC/dtau = drift*dC/dk + (0.5*alpha + 0.5*beta*k + 0.5*gamma*k^2)*d2C/dk2

    so sigma^2(k) = alpha + beta*k + gamma*k^2, and the OLS coefficients map
    back as ``drift = coef[0]``, ``alpha = 2*coef[1]``,
    ``beta = 2*coef[2]``, ``gamma = 2*coef[3]``.

    Sign convention
    ---------------
    The diagnostic (see module docstring) confirmed the convention k<0 <=>
    K<F is in force and BS theta is positive on the SPY surface. The synthetic
    test ``test_quadratic_v4_synthetic_skew`` confirms that, given a true
    sigma^2(k) = 0.04 - 0.01*k + 0.05*k^2, the routine recovers
    beta < 0, gamma > 0 in the correct signs. No sign flip is applied.
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

    F_grid = compute_forward_prices(S0, r, q, tau_grid)
    K_2d = np.outer(np.exp(k_grid), F_grid)
    theta = bs_theta_analytical(S0, K_2d, tau_grid, sigma_imp, r, q)

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
    except Exception as exc:
        logger.warning("quadratic_dupire_v4 lstsq failed: %s", exc)
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
        sigma_at_k[float(kv)] = float(np.sqrt(sig2)) if sig2 > 0 \
            else float('nan')

    return {
        'r2_score': float(r2),
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'drift': drift,
        'sigma_at_k_dict': sigma_at_k,
        'condition_number': cond,
        'sign_convention_note': (
            'Synthetic sigma^2(k) = 0.04 - 0.01*k + 0.05*k^2 is recovered '
            'with beta<0 and gamma>0; no sign flip applied to empirical fits.'
        ),
    }


def quadratic_v4_on_ticker(option_data: dict[str, Any], ticker: str,
                           n_k: int = 40,
                           k_range: tuple[float, float] = (-0.25, 0.25),
                           use_weights: bool = True,
                           q: Optional[float] = None,
                           ) -> dict[str, Any]:
    """Wrapper: SVI surface + reconstruct sigma_imp + ``quadratic_dupire_v4``."""
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
        sigma_imp = reconstruct_sigma_imp_grid(
            surface['svi_params'], surface['k_grid'], surface['tau_grid'],
        )
        weights = None
        if use_weights:
            try:
                weights = compute_liquidity_weights(
                    df, surface['k_grid'], surface['tau_grid'], q=q, S0=S0,
                )
            except Exception as exc:
                out['errors']['weights'] = str(exc)
                weights = None
        res = quadratic_dupire_v4(
            surface['C_surface'], sigma_imp,
            surface['k_grid'], surface['tau_grid'],
            S0, r, q, weights=weights,
        )
        out['result'] = res
    except Exception as exc:
        logger.warning("%s: quadratic_v4_on_ticker failed: %s", ticker, exc)
        out['errors']['top'] = str(exc)
        out['result'] = None
    return out


# ---------------------------------------------------------------------------
# Fix 5 -- per-expiration sigma with analytical theta
# ---------------------------------------------------------------------------

def per_expiration_sigma_v4(C_grid: np.ndarray, sigma_imp_grid: np.ndarray,
                            k_grid: np.ndarray, tau_grid: np.ndarray,
                            S0: float, r: float, q: float = 0.0,
                            min_k_points: int = 10,
                            sigma_bounds: tuple[float, float] = (0.01, 2.0),
                            ) -> pd.DataFrame:
    """Per-expiration sigma_loc via 2-term OLS with analytical theta.

    For each expiration tau_i:
        target  = theta(k, tau_i)  (analytical, from sigma_imp)
        library = [dC/dk(k, tau_i), d2C/dk2(k, tau_i)]
        sigma_loc(tau_i) = sqrt(2 * c2) if c2 > 0.

    Returns a DataFrame with columns:
        tau, n_used, c1, c2, sigma_loc, r2, market_avg_iv.
    """
    C = np.asarray(C_grid, dtype=np.float64)
    sig = np.asarray(sigma_imp_grid, dtype=np.float64)
    k = np.asarray(k_grid, dtype=np.float64)
    tau = np.asarray(tau_grid, dtype=np.float64)
    n_k, n_tau = C.shape
    if sig.shape != C.shape:
        raise ValueError(
            f"sigma_imp shape {sig.shape} != C shape {C.shape}"
        )

    F_grid = compute_forward_prices(S0, r, q, tau)
    K_2d = np.outer(np.exp(k), F_grid)
    theta = bs_theta_analytical(S0, K_2d, tau, sig, r, q)
    dCdk, d2Cdk2 = _central_dk(C, k)

    sig_lo, sig_hi = float(sigma_bounds[0]), float(sigma_bounds[1])
    rows: list[dict[str, Any]] = []
    for i in range(n_tau):
        try:
            tgt = theta[:, i]
            lib = np.column_stack([dCdk[:, i], d2Cdk2[:, i]])
            mask = np.isfinite(tgt) & np.all(np.isfinite(lib), axis=1)
            n_used = int(mask.sum())
            market_iv = float(np.nanmean(sig[:, i])) \
                if np.any(np.isfinite(sig[:, i])) else float('nan')
            if n_used < int(min_k_points):
                rows.append({
                    'tau': float(tau[i]), 'n_used': n_used,
                    'c1': float('nan'), 'c2': float('nan'),
                    'sigma_loc': float('nan'), 'r2': float('nan'),
                    'market_avg_iv': market_iv,
                })
                continue
            A = lib[mask]
            b = tgt[mask]
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            c1 = float(coef[0])
            c2 = float(coef[1])
            pred = A @ coef
            r2 = _r2_from(b, pred)
            sigma_loc = float(np.sqrt(max(0.0, 2.0 * c2)))
            # Report even if outside bounds; downstream can filter.
            rows.append({
                'tau': float(tau[i]),
                'n_used': n_used,
                'c1': c1,
                'c2': c2,
                'sigma_loc': sigma_loc,
                'r2': float(r2),
                'market_avg_iv': market_iv,
            })
        except Exception as exc:
            logger.debug(
                "per_expiration_sigma_v4 failed at tau=%.4f: %s",
                float(tau[i]), exc,
            )
            rows.append({
                'tau': float(tau[i]), 'n_used': 0,
                'c1': float('nan'), 'c2': float('nan'),
                'sigma_loc': float('nan'), 'r2': float('nan'),
                'market_avg_iv': float('nan'),
            })

    df_out = pd.DataFrame(rows)
    # Annotate sigma in/out of bounds for convenience but do not drop rows.
    df_out['sigma_in_bounds'] = ((df_out['sigma_loc'] >= sig_lo)
                                  & (df_out['sigma_loc'] <= sig_hi))
    return df_out


def per_expiration_sigma_v4_on_ticker(option_data: dict[str, Any],
                                       ticker: str,
                                       n_k: int = 40,
                                       k_range: tuple[float, float] = (-0.25, 0.25),
                                       q: Optional[float] = None,
                                       min_k_points: int = 10,
                                       ) -> dict[str, Any]:
    """Build SVI surface + sigma_imp grid + ``per_expiration_sigma_v4``."""
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
        sigma_imp = reconstruct_sigma_imp_grid(
            surface['svi_params'], surface['k_grid'], surface['tau_grid'],
        )
        term_df = per_expiration_sigma_v4(
            surface['C_surface'], sigma_imp,
            surface['k_grid'], surface['tau_grid'],
            S0, r, q, min_k_points=min_k_points,
        )
        out['term_structure'] = term_df
    except Exception as exc:
        logger.warning("%s: per_expiration_sigma_v4_on_ticker failed: %s",
                       ticker, exc)
        out['errors']['top'] = str(exc)
        out['term_structure'] = None
    return out


# ---------------------------------------------------------------------------
# Module-level orchestrator (optional convenience)
# ---------------------------------------------------------------------------

def run_v5_experiments_on_ticker(option_data: dict[str, Any], ticker: str,
                                  n_k: int = 40,
                                  k_range: tuple[float, float] = (-0.25, 0.25),
                                  window_size: int = 8, stride: int = 3,
                                  min_r2: float = 0.5,
                                  sigma_bounds: tuple[float, float] = (0.01, 2.0),
                                  seed: int = 42,
                                  ) -> dict[str, Any]:
    """Run windowed v4, quadratic v4, and per-expiration v4 for one ticker."""
    set_all_seeds(seed)
    out: dict[str, Any] = {'ticker': ticker, 'errors': {}}
    try:
        S0 = float(option_data['S0'])
        r = float(option_data['r'])
        df = option_data['option_df'].copy()
        try:
            from src.real_data_v2 import get_dividend_yield
            q = get_dividend_yield(ticker)
        except Exception:
            q = 0.0
        out['q'] = float(q)
        surface = build_logm_surface_svi(df, S0, r, q, n_k=n_k, k_range=k_range)
        sigma_imp = reconstruct_sigma_imp_grid(
            surface['svi_params'], surface['k_grid'], surface['tau_grid'],
        )
        try:
            weights = compute_liquidity_weights(
                df, surface['k_grid'], surface['tau_grid'], q=q, S0=S0,
            )
        except Exception as exc:
            out['errors']['weights'] = str(exc)
            weights = None

        try:
            win = windowed_2term_dupire_v4(
                surface['C_surface'], sigma_imp,
                surface['k_grid'], surface['tau_grid'],
                S0, r, q, window_size=window_size, stride=stride,
                min_r2=min_r2, sigma_bounds=sigma_bounds,
            )
            out['windowed'] = win
        except Exception as exc:
            out['errors']['windowed'] = str(exc)
            out['windowed'] = None

        try:
            quad = quadratic_dupire_v4(
                surface['C_surface'], sigma_imp,
                surface['k_grid'], surface['tau_grid'],
                S0, r, q, weights=weights,
            )
            out['quadratic'] = quad
        except Exception as exc:
            out['errors']['quadratic'] = str(exc)
            out['quadratic'] = None

        try:
            term_df = per_expiration_sigma_v4(
                surface['C_surface'], sigma_imp,
                surface['k_grid'], surface['tau_grid'],
                S0, r, q,
            )
            out['per_expiration'] = term_df
        except Exception as exc:
            out['errors']['per_expiration'] = str(exc)
            out['per_expiration'] = None
    except Exception as exc:
        out['errors']['top'] = str(exc)
    return out
