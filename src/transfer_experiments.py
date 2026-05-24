"""
Transfer / generalization experiments for the SINDy-KAN-Dupire model.

Three experiments implemented here, all addressing PRD revision B2:

1. ``ticker_transfer`` -- train a [2, 1] KAN-Dupire on ticker A (snapshot
   date d_train), evaluate on ticker B (same date or another date).
2. ``temporal_transfer`` -- same as (1) but with A == B and different
   snapshot dates.
3. ``leave_one_expiration_out`` -- LOO-CV over the unique tau values of a
   single ticker/date; train on all-except-one and evaluate on the held
   out maturity, plus a "mild" tau < 0.75 vs tau >= 0.75 split.

The maturity-failure analysis additionally produces
``outputs/figures/paper/maturity_transfer_analysis.{png,pdf}``: per-
expiration R^2 vs tau, serif font, 300 DPI, colorblind palette.

Everything runs CPU-only on the cached real_chain_*.csv files committed
to ``outputs/tables/``. Seed = 42 throughout. Inner errors are caught and
logged so a single bad pair does not abort the whole sweep.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from src.utils import setup_logging
from src.real_data_v2 import (
    build_logm_surface_svi,
    compute_forward_prices,
    get_dividend_yield,
)
from src.real_data_v4 import (
    bs_theta_analytical,
    reconstruct_sigma_imp_grid,
    _central_dk,
    _r2_from,
)
from src.sindy_kan import train_kan_dupire_21

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_CACHE_DIR = os.path.join('outputs', 'tables')


def _load_chain(ticker: str, date: str,
                cache_dir: str = _CACHE_DIR) -> dict[str, Any]:
    """Load a cached ``real_chain_{ticker}_{date}.csv`` and shape it.

    Returns a dict with keys ``ticker``, ``S0``, ``r``, ``q``, ``option_df``
    -- the same minimal shape ``sindy_kan_dupire_on_ticker`` expects.
    """
    path = os.path.join(cache_dir, f'real_chain_{ticker}_{date}.csv')
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if 'mid_price' not in df.columns and 'mid' in df.columns:
        df = df.rename(columns={'mid': 'mid_price'})
    S0 = float(df['S0'].iloc[0])
    r = float(df['r'].iloc[0])
    try:
        q = float(get_dividend_yield(ticker))
    except Exception:
        q = 0.0
    return {
        'ticker': ticker, 'date': date,
        'S0': S0, 'r': r, 'q': q, 'option_df': df,
    }


def _surface_from_chain(chain: dict[str, Any],
                         n_k: int = 40, n_tau: int = 15,
                         k_range: tuple[float, float] = (-0.20, 0.20),
                         ) -> dict[str, Any]:
    """Build a Dupire-space (k, tau) grid + theta target from a chain dict."""
    S0 = float(chain['S0']); r = float(chain['r']); q = float(chain['q'])
    df = chain['option_df']
    surf = build_logm_surface_svi(df, S0, r, q,
                                    n_k=n_k, k_range=k_range, n_tau=n_tau)
    C = surf['C_surface']
    k_grid = surf['k_grid']
    tau_grid = surf['tau_grid']
    sigma_imp = reconstruct_sigma_imp_grid(surf['svi_params'], k_grid, tau_grid)
    F_grid = compute_forward_prices(S0, r, q, tau_grid)
    K2d = np.outer(np.exp(k_grid), F_grid)
    theta = bs_theta_analytical(S0, K2d, tau_grid, sigma_imp, r, q)
    dCdk, d2Cdk2 = _central_dk(C, k_grid)
    return {
        'C': C, 'k': k_grid, 'tau': tau_grid,
        'sigma_imp': sigma_imp, 'theta': theta,
        'dCdk': dCdk, 'd2Cdk2': d2Cdk2,
        'S0': S0, 'r': r, 'q': q,
    }


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return _r2_from(np.asarray(y_true, dtype=np.float64),
                    np.asarray(y_pred, dtype=np.float64))


def _predict_kan(res: dict[str, Any], dCdk: np.ndarray,
                  d2Cdk2: np.ndarray) -> np.ndarray:
    """Apply a trained [2, 1] KAN to *unstandardized* (dCdk, d2Cdk2) arrays."""
    xp = res['x_std_params']; yp = res['y_std_params']
    a = np.asarray(dCdk, dtype=np.float64).ravel()
    b = np.asarray(d2Cdk2, dtype=np.float64).ravel()
    a_std = (2.0 * (a - xp['center'][0]) / xp['range'][0])
    b_std = (2.0 * (b - xp['center'][1]) / xp['range'][1])
    X = torch.tensor(np.column_stack([a_std, b_std]).astype(np.float32))
    model = res['model']
    model.eval()
    with torch.no_grad():
        pred_std = model(X).numpy()
    return pred_std * float(yp['std']) + float(yp['mean'])


def _train_on_mask(dCdk: np.ndarray, d2Cdk2: np.ndarray, theta: np.ndarray,
                    mask: np.ndarray, n_epochs: int = 2000,
                    seed: int = 42) -> dict[str, Any]:
    return train_kan_dupire_21(
        dCdk[mask], d2Cdk2[mask], theta[mask],
        n_epochs=int(n_epochs), seed=int(seed))


# ---------------------------------------------------------------------------
# Experiment A: cross-ticker transfer
# ---------------------------------------------------------------------------


def ticker_transfer(train_ticker: str, test_ticker: str,
                     train_date: str, test_date: Optional[str] = None,
                     n_epochs: int = 2000, seed: int = 42,
                     ) -> dict[str, Any]:
    """Train on ticker A and evaluate on ticker B (default same date)."""
    if test_date is None:
        test_date = train_date
    train_surf = _surface_from_chain(_load_chain(train_ticker, train_date))
    test_surf = _surface_from_chain(_load_chain(test_ticker, test_date))

    a_tr, b_tr, t_tr = (train_surf['dCdk'].ravel(),
                          train_surf['d2Cdk2'].ravel(),
                          train_surf['theta'].ravel())
    ok = np.isfinite(a_tr) & np.isfinite(b_tr) & np.isfinite(t_tr)
    res = train_kan_dupire_21(a_tr[ok], b_tr[ok], t_tr[ok],
                                 n_epochs=int(n_epochs), seed=int(seed))

    a_te = test_surf['dCdk'].ravel()
    b_te = test_surf['d2Cdk2'].ravel()
    t_te = test_surf['theta'].ravel()
    ok_te = np.isfinite(a_te) & np.isfinite(b_te) & np.isfinite(t_te)
    pred = _predict_kan(res, a_te[ok_te], b_te[ok_te])
    r2 = _r2(t_te[ok_te], pred)
    return {
        'train_ticker': train_ticker, 'test_ticker': test_ticker,
        'train_date': train_date, 'test_date': test_date,
        'experiment': 'ticker_transfer',
        'R2_test': float(r2),
        'R2_train': float(res['train_r2']),
        'n_train': int(ok.sum()),
        'n_test': int(ok_te.sum()),
    }


# ---------------------------------------------------------------------------
# Experiment B: temporal transfer
# ---------------------------------------------------------------------------


def temporal_transfer(ticker: str, train_date: str, test_date: str,
                       n_epochs: int = 2000, seed: int = 42,
                       ) -> dict[str, Any]:
    """Train on a snapshot date and evaluate on a later snapshot of the same
    ticker. Returns the same row schema as ``ticker_transfer`` but the row
    is tagged ``experiment == 'temporal_transfer'``."""
    out = ticker_transfer(ticker, ticker, train_date, test_date,
                            n_epochs=n_epochs, seed=seed)
    out['experiment'] = 'temporal_transfer'
    return out


# ---------------------------------------------------------------------------
# Experiment C: leave-one-expiration-out + mild maturity split
# ---------------------------------------------------------------------------


def leave_one_expiration_out(ticker: str, date: str, n_epochs: int = 1500,
                              seed: int = 42) -> pd.DataFrame:
    """For each tau column of the SVI-smoothed surface, train on the others
    and evaluate on the held-out tau. Returns a DataFrame with columns
    ``tau_left_out``, ``R2``, ``n_test_points``."""
    surf = _surface_from_chain(_load_chain(ticker, date))
    tau_grid = surf['tau']
    dCdk = surf['dCdk']
    d2Cdk2 = surf['d2Cdk2']
    theta = surf['theta']
    n_k, n_tau = dCdk.shape

    rows = []
    tau_idx_mask = np.zeros((n_k, n_tau), dtype=bool)
    for j in range(n_tau):
        mask_train = np.ones((n_k, n_tau), dtype=bool)
        mask_train[:, j] = False
        mask_test = ~mask_train
        try:
            res = _train_on_mask(dCdk, d2Cdk2, theta, mask_train,
                                  n_epochs=n_epochs, seed=seed)
            a_te = dCdk[mask_test].ravel()
            b_te = d2Cdk2[mask_test].ravel()
            t_te = theta[mask_test].ravel()
            ok = np.isfinite(a_te) & np.isfinite(b_te) & np.isfinite(t_te)
            pred = _predict_kan(res, a_te[ok], b_te[ok])
            r2 = _r2(t_te[ok], pred)
        except Exception as exc:
            logger.warning("LOO tau_idx=%d failed: %s", j, exc)
            r2 = float('nan'); ok = np.array([False])
        rows.append({
            'ticker': ticker, 'date': date,
            'tau_left_out': float(tau_grid[j]),
            'R2': float(r2),
            'n_test_points': int(ok.sum()),
        })
    return pd.DataFrame(rows)


def mild_maturity_split(ticker: str, date: str, tau_threshold: float = 0.75,
                          n_epochs: int = 2000, seed: int = 42,
                          ) -> dict[str, Any]:
    """Train on tau < threshold, evaluate on tau >= threshold."""
    surf = _surface_from_chain(_load_chain(ticker, date))
    tau_grid = surf['tau']
    n_k = surf['dCdk'].shape[0]
    tau_mat = np.tile(tau_grid.reshape(1, -1), (n_k, 1))
    mask_train = tau_mat < float(tau_threshold)
    mask_test = tau_mat >= float(tau_threshold)
    if mask_train.sum() < 20 or mask_test.sum() < 5:
        return {
            'experiment': 'mild_maturity_split',
            'train_ticker': ticker, 'test_ticker': ticker,
            'train_date': date, 'test_date': date,
            'R2_test': float('nan'),
            'R2_train': float('nan'),
            'n_train': int(mask_train.sum()),
            'n_test': int(mask_test.sum()),
            'tau_threshold': float(tau_threshold),
        }
    res = _train_on_mask(surf['dCdk'], surf['d2Cdk2'], surf['theta'],
                          mask_train, n_epochs=n_epochs, seed=seed)
    a_te = surf['dCdk'][mask_test].ravel()
    b_te = surf['d2Cdk2'][mask_test].ravel()
    t_te = surf['theta'][mask_test].ravel()
    ok = np.isfinite(a_te) & np.isfinite(b_te) & np.isfinite(t_te)
    pred = _predict_kan(res, a_te[ok], b_te[ok])
    r2 = _r2(t_te[ok], pred)
    return {
        'experiment': 'mild_maturity_split',
        'train_ticker': ticker, 'test_ticker': ticker,
        'train_date': date, 'test_date': date,
        'R2_test': float(r2),
        'R2_train': float(res['train_r2']),
        'n_train': int(mask_train.sum()),
        'n_test': int(ok.sum()),
        'tau_threshold': float(tau_threshold),
    }


# ---------------------------------------------------------------------------
# Figure: per-expiration R^2 vs tau
# ---------------------------------------------------------------------------


def plot_maturity_transfer_analysis(loo_df: pd.DataFrame,
                                      save_path: str = 'outputs/figures/paper/maturity_transfer_analysis',
                                      ) -> Optional[str]:
    """Per-expiration R^2 vs tau, serif font, 300 DPI, colorblind palette.

    Accepts the long-form ``loo_df`` produced by ``leave_one_expiration_out``
    -- if a ``ticker`` column is present, one curve is drawn per ticker.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'legend.fontsize': 9,
        })

        # Okabe-Ito colorblind-safe palette.
        palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
                   '#F0E442', '#56B4E9', '#E69F00', '#000000']

        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
        if 'ticker' in loo_df.columns and loo_df['ticker'].nunique() > 1:
            for i, (tk, sub) in enumerate(loo_df.groupby('ticker')):
                sub = sub.sort_values('tau_left_out')
                ax.plot(sub['tau_left_out'], sub['R2'],
                         marker='o', lw=1.5, ms=6,
                         color=palette[i % len(palette)], label=str(tk))
            ax.legend(loc='lower left', frameon=True)
        else:
            sub = loo_df.sort_values('tau_left_out')
            ax.plot(sub['tau_left_out'], sub['R2'],
                     marker='o', lw=1.8, ms=7, color=palette[0],
                     label='leave-one-tau-out R$^2$')
            ax.legend(loc='lower left', frameon=True)

        ax.axhline(0.0, color='gray', lw=1, ls='--', alpha=0.6)
        ax.set_xlabel(r'held-out maturity $\tau$ (years)')
        ax.set_ylabel(r'leave-one-out test $R^2$')
        ax.set_title(r'Per-expiration generalization of the [2, 1] KAN-Dupire')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        png = f"{save_path}.png"; pdf = f"{save_path}.pdf"
        fig.savefig(png, dpi=300, bbox_inches='tight')
        fig.savefig(pdf, bbox_inches='tight')
        plt.close(fig)
        return png
    except Exception as exc:
        logger.warning("plot_maturity_transfer_analysis failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Orchestrator: run all three experiments end-to-end
# ---------------------------------------------------------------------------


def run_all_transfer_experiments(date: str = '20260329',
                                   second_date: str = '20260523',
                                   n_epochs: int = 1500, seed: int = 42,
                                   ) -> dict[str, Any]:
    """Run cross-ticker (4 pairs), temporal (SPY, QQQ), and mild-maturity
    transfer, plus per-expiration LOO on SPY.

    Returns a dict with the assembled DataFrames; also writes:

    * ``outputs/tables/transfer_experiments.csv``
    * ``outputs/tables/per_expiration_loo.csv``
    * ``outputs/figures/paper/maturity_transfer_analysis.{png,pdf}``
    """
    transfer_rows: list[dict[str, Any]] = []

    # --- A) Ticker transfer pairs
    pairs = [('SPY', 'QQQ'), ('QQQ', 'SPY'),
              ('AAPL', 'MSFT'), ('MSFT', 'AAPL')]
    for tr, te in pairs:
        try:
            row = ticker_transfer(tr, te, train_date=date, test_date=date,
                                    n_epochs=n_epochs, seed=seed)
            transfer_rows.append(row)
            logger.info("ticker %s -> %s: R2=%.3f", tr, te, row['R2_test'])
        except Exception as exc:
            logger.warning("ticker_transfer %s->%s failed: %s", tr, te, exc)
            transfer_rows.append({
                'experiment': 'ticker_transfer',
                'train_ticker': tr, 'test_ticker': te,
                'train_date': date, 'test_date': date,
                'R2_test': float('nan'), 'R2_train': float('nan'),
                'n_train': 0, 'n_test': 0,
            })

    # --- B) Temporal transfer
    for tk in ('SPY', 'QQQ'):
        try:
            row = temporal_transfer(tk, train_date=date,
                                      test_date=second_date,
                                      n_epochs=n_epochs, seed=seed)
            transfer_rows.append(row)
            logger.info("temporal %s %s->%s: R2=%.3f",
                        tk, date, second_date, row['R2_test'])
        except Exception as exc:
            logger.warning("temporal_transfer %s failed: %s", tk, exc)
            transfer_rows.append({
                'experiment': 'temporal_transfer',
                'train_ticker': tk, 'test_ticker': tk,
                'train_date': date, 'test_date': second_date,
                'R2_test': float('nan'), 'R2_train': float('nan'),
                'n_train': 0, 'n_test': 0,
            })

    # --- C2) Mild maturity split on SPY
    try:
        row = mild_maturity_split('SPY', date=date, tau_threshold=0.75,
                                    n_epochs=n_epochs, seed=seed)
        transfer_rows.append(row)
        logger.info("mild_maturity SPY tau>=0.75: R2=%.3f", row['R2_test'])
    except Exception as exc:
        logger.warning("mild_maturity_split SPY failed: %s", exc)

    transfer_df = pd.DataFrame(transfer_rows)
    out_dir = os.path.join('outputs', 'tables')
    os.makedirs(out_dir, exist_ok=True)
    transfer_df.to_csv(os.path.join(out_dir, 'transfer_experiments.csv'),
                        index=False)

    # --- C1) Per-expiration LOO on SPY
    try:
        loo_df = leave_one_expiration_out('SPY', date=date,
                                            n_epochs=n_epochs, seed=seed)
    except Exception as exc:
        logger.warning("leave_one_expiration_out SPY failed: %s", exc)
        loo_df = pd.DataFrame(columns=['ticker', 'date', 'tau_left_out',
                                         'R2', 'n_test_points'])
    loo_df.to_csv(os.path.join(out_dir, 'per_expiration_loo.csv'), index=False)

    # --- Figure
    fig_path = plot_maturity_transfer_analysis(
        loo_df, save_path='outputs/figures/paper/maturity_transfer_analysis')

    return {
        'transfer_df': transfer_df,
        'loo_df': loo_df,
        'figure_path': fig_path,
    }


# ---------------------------------------------------------------------------
# Improvement 5: per-expiration coefficients + regime transfer
# ---------------------------------------------------------------------------


def _fit_dupire_2term(dCdk: np.ndarray, d2Cdk2: np.ndarray,
                       theta: np.ndarray) -> tuple[float, float]:
    """OLS 2-term Dupire fit. Returns (coef_drift, coef_diffusion).

    The model is ``theta = c_drift * dC/dk + c_diff * d2C/dk2`` with no
    intercept -- matches the SINDy-Dupire library form. NaNs are masked.
    """
    a = np.asarray(dCdk, dtype=np.float64).ravel()
    b = np.asarray(d2Cdk2, dtype=np.float64).ravel()
    t = np.asarray(theta, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(t)
    a, b, t = a[ok], b[ok], t[ok]
    if a.size < 3:
        return float('nan'), float('nan')
    A = np.column_stack([a, b])
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    return float(coef[0]), float(coef[1])


def leave_one_expiration_coefficients(ticker: str, date: str,
                                        n_epochs: int = 1500,
                                        seed: int = 42) -> pd.DataFrame:
    """Per-expiration LOO that also extracts discovered coefficients.

    For each held-out tau column, fits a [2,1] KAN on the rest, computes the
    test R^2 on the held-out maturity, and -- in addition -- fits a linear
    2-term Dupire regression on the held-out slice to extract physical
    ``coef_drift`` and ``coef_diffusion`` values. Also records the
    market-average implied volatility at that maturity from the raw chain.

    Returns DataFrame with columns: ticker, tau, R2, coef_drift,
    coef_diffusion, market_avg_iv, n_strikes.
    """
    chain = _load_chain(ticker, date)
    surf = _surface_from_chain(chain)
    tau_grid = surf['tau']
    dCdk = surf['dCdk']
    d2Cdk2 = surf['d2Cdk2']
    theta = surf['theta']
    n_k, n_tau = dCdk.shape

    # Raw-chain market IVs by expiration tau.
    raw = chain['option_df']
    iv_col = 'implied_vol' if 'implied_vol' in raw.columns else None

    rows = []
    for j in range(n_tau):
        tau_j = float(tau_grid[j])
        mask_train = np.ones((n_k, n_tau), dtype=bool); mask_train[:, j] = False
        mask_test = ~mask_train
        try:
            res = _train_on_mask(dCdk, d2Cdk2, theta, mask_train,
                                  n_epochs=n_epochs, seed=seed)
            a_te = dCdk[mask_test].ravel()
            b_te = d2Cdk2[mask_test].ravel()
            t_te = theta[mask_test].ravel()
            ok = np.isfinite(a_te) & np.isfinite(b_te) & np.isfinite(t_te)
            pred = _predict_kan(res, a_te[ok], b_te[ok])
            r2 = _r2(t_te[ok], pred)
        except Exception as exc:
            logger.warning("LOO coef tau_idx=%d failed: %s", j, exc)
            r2 = float('nan')

        # Discover coefficients on the held-out slice using OLS 2-term Dupire.
        try:
            c_drift, c_diff = _fit_dupire_2term(
                dCdk[:, j], d2Cdk2[:, j], theta[:, j])
        except Exception as exc:
            logger.warning("coef fit tau_idx=%d failed: %s", j, exc)
            c_drift, c_diff = float('nan'), float('nan')

        # Market-average IV for this expiration: find the raw-chain rows whose
        # tau is closest to tau_j (the SVI grid tau may not exactly equal a
        # raw tau, so use nearest-tau matching with a small tolerance).
        market_iv = float('nan')
        n_strikes = 0
        if iv_col is not None and 'tau' in raw.columns:
            try:
                tau_diffs = np.abs(raw['tau'].values - tau_j)
                # Match rows within 1 day = ~0.0027 yr.
                tol = max(0.005, 0.05 * tau_j)
                mask = tau_diffs <= tol
                if mask.sum() == 0:
                    # Fall back to the unique tau closest to tau_j.
                    nearest = float(raw['tau'].iloc[int(np.argmin(tau_diffs))])
                    mask = np.abs(raw['tau'].values - nearest) < 1e-9
                ivs = raw[iv_col].values[mask]
                ivs = ivs[np.isfinite(ivs)]
                if ivs.size > 0:
                    market_iv = float(np.mean(ivs))
                    n_strikes = int(ivs.size)
            except Exception as exc:
                logger.warning("market_iv lookup tau=%.3f failed: %s",
                               tau_j, exc)

        rows.append({
            'ticker': ticker, 'tau': tau_j, 'R2': float(r2),
            'coef_drift': float(c_drift),
            'coef_diffusion': float(c_diff),
            'market_avg_iv': float(market_iv),
            'n_strikes': int(n_strikes),
        })
    return pd.DataFrame(rows)


def regime_transfer_atm_otm(ticker: str = 'SPY', date: str = '20260329',
                              k_threshold: float = 0.05,
                              n_epochs: int = 1500,
                              seed: int = 42) -> list[dict[str, Any]]:
    """Train on one |k| regime and evaluate on the other.

    Splits the SVI-smoothed (k, tau) grid by |k| < k_threshold (ATM) vs
    |k| >= k_threshold (OTM). Runs both transfer directions.

    Returns a list of two row dicts ready to append to
    ``generalization_analysis.csv`` with columns: experiment, train_regime,
    test_regime, R2_test, R2_train, n_train, n_test.
    """
    surf = _surface_from_chain(_load_chain(ticker, date))
    k_grid = surf['k']
    dCdk = surf['dCdk']; d2Cdk2 = surf['d2Cdk2']; theta = surf['theta']
    n_k, n_tau = dCdk.shape

    atm_k = np.abs(k_grid) < float(k_threshold)
    otm_k = ~atm_k
    atm_mask = np.tile(atm_k.reshape(-1, 1), (1, n_tau))
    otm_mask = np.tile(otm_k.reshape(-1, 1), (1, n_tau))

    def _run(train_mask, test_mask, train_name, test_name):
        out = {
            'experiment': 'regime_transfer',
            'ticker': ticker, 'date': date,
            'train_regime': train_name, 'test_regime': test_name,
            'R2_test': float('nan'), 'R2_train': float('nan'),
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
            'k_threshold': float(k_threshold),
        }
        if train_mask.sum() < 20 or test_mask.sum() < 5:
            return out
        try:
            res = _train_on_mask(dCdk, d2Cdk2, theta, train_mask,
                                  n_epochs=n_epochs, seed=seed)
            out['R2_train'] = float(res['train_r2'])
            a_te = dCdk[test_mask].ravel()
            b_te = d2Cdk2[test_mask].ravel()
            t_te = theta[test_mask].ravel()
            ok = np.isfinite(a_te) & np.isfinite(b_te) & np.isfinite(t_te)
            pred = _predict_kan(res, a_te[ok], b_te[ok])
            out['R2_test'] = float(_r2(t_te[ok], pred))
            out['n_test'] = int(ok.sum())
        except Exception as exc:
            logger.warning("regime_transfer %s->%s failed: %s",
                           train_name, test_name, exc)
        return out

    rows = [
        _run(atm_mask, otm_mask, 'ATM', 'OTM'),
        _run(otm_mask, atm_mask, 'OTM', 'ATM'),
    ]
    return rows


# ---------------------------------------------------------------------------
# Figures for Improvement 5
# ---------------------------------------------------------------------------


def plot_coefficient_vs_maturity(
        coef_df: pd.DataFrame,
        save_path: str = 'outputs/figures/paper/coefficient_vs_maturity'
        ) -> Optional[str]:
    """Dual-axis plot of discovered diffusion coefficient vs market IV term.

    Left axis: discovered diffusion coefficient (#0072B2 blue).
    Right axis: market-average implied volatility (#D55E00 orange).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'legend.fontsize': 9,
        })
        df = coef_df.sort_values('tau').reset_index(drop=True)
        fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
        c_blue = '#0072B2'
        c_orange = '#D55E00'

        ax1.plot(df['tau'], df['coef_diffusion'], marker='o', lw=2,
                  ms=6, color=c_blue, label='discovered diffusion coef')
        ax1.set_xlabel(r'maturity $\tau$ (years)')
        ax1.set_ylabel('discovered diffusion coefficient', color=c_blue)
        ax1.tick_params(axis='y', labelcolor=c_blue)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0.0, color='gray', lw=0.8, ls=':', alpha=0.5)

        ax2 = ax1.twinx()
        ax2.plot(df['tau'], df['market_avg_iv'], marker='s', lw=2,
                  ms=6, color=c_orange, label='market avg IV')
        ax2.set_ylabel('market average implied volatility',
                        color=c_orange)
        ax2.tick_params(axis='y', labelcolor=c_orange)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                    loc='best', frameon=True)

        ax1.set_title('Discovered diffusion coefficient vs market IV '
                       'term structure')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        png = f"{save_path}.png"; pdf = f"{save_path}.pdf"
        fig.savefig(png, dpi=300, bbox_inches='tight')
        fig.savefig(pdf, bbox_inches='tight')
        plt.close(fig)
        return png
    except Exception as exc:
        logger.warning("plot_coefficient_vs_maturity failed: %s", exc)
        return None


def plot_transfer_heatmap(
        loo_df: pd.DataFrame,
        save_path: str = 'outputs/figures/paper/transfer_heatmap'
        ) -> Optional[str]:
    """Per-expiration LOO R^2 with regime-colored points and annotations.

    x-axis: tau (years). y-axis: held-out R^2. Points colored green where
    R^2 > 0.5, orange where 0 < R^2 <= 0.5, red where R^2 <= 0.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'legend.fontsize': 9,
        })
        # Use 'tau' if present, else 'tau_left_out'.
        tau_col = 'tau' if 'tau' in loo_df.columns else 'tau_left_out'
        df = loo_df.sort_values(tau_col).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8.0, 4.8))

        def _color(r2):
            if not np.isfinite(r2):
                return '#999999'
            if r2 > 0.5:
                return '#009E73'  # green
            if r2 > 0.0:
                return '#D55E00'  # orange
            return '#CC0033'      # red

        colors = [_color(r) for r in df['R2']]
        ax.scatter(df[tau_col], df['R2'], c=colors, s=70,
                    edgecolor='black', linewidth=0.5, zorder=3)
        ax.plot(df[tau_col], df['R2'], color='#444444', lw=0.8,
                  alpha=0.4, zorder=2)
        ax.axhline(0.0, color='gray', lw=1, ls='--', alpha=0.6)

        # Annotate each point with the tau value (1-2 dec digits).
        if 'date' in df.columns:
            label_src = df['date'].astype(str)
        else:
            label_src = [f"{t:.2f}" for t in df[tau_col]]
        for x, y, lab in zip(df[tau_col], df['R2'], label_src):
            if np.isfinite(y):
                ax.annotate(str(lab)[-4:] if isinstance(lab, str)
                              and len(str(lab)) >= 4 else f"{float(x):.2f}",
                              (x, y),
                              xytext=(4, 4), textcoords='offset points',
                              fontsize=7, alpha=0.75)

        # Legend proxies.
        from matplotlib.lines import Line2D
        legend_items = [
            Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='#009E73', markersize=8,
                    label=r'$R^2 > 0.5$'),
            Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='#D55E00', markersize=8,
                    label=r'$0 < R^2 \leq 0.5$'),
            Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='#CC0033', markersize=8,
                    label=r'$R^2 \leq 0$'),
        ]
        ax.legend(handles=legend_items, loc='lower left', frameon=True)
        ax.set_xlabel(r'held-out maturity $\tau$ (years)')
        ax.set_ylabel(r'leave-one-out test $R^2$')
        ax.set_title(r'Per-expiration leave-one-out $R^2$ (SPY)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        png = f"{save_path}.png"; pdf = f"{save_path}.pdf"
        fig.savefig(png, dpi=300, bbox_inches='tight')
        fig.savefig(pdf, bbox_inches='tight')
        plt.close(fig)
        return png
    except Exception as exc:
        logger.warning("plot_transfer_heatmap failed: %s", exc)
        return None


def run_generalization_analysis(ticker: str = 'SPY', date: str = '20260329',
                                  n_epochs: int = 1500, seed: int = 42,
                                  ) -> dict[str, Any]:
    """Orchestrate Improvement 5: per-expiration coefs + regime transfer.

    Writes:
      * outputs/tables/per_expiration_coefficients.csv
      * outputs/tables/generalization_analysis.csv
      * outputs/figures/paper/coefficient_vs_maturity.{png,pdf}
      * outputs/figures/paper/transfer_heatmap.{png,pdf}

    Returns a dict with ``coef_df``, ``regime_df``, and figure paths.
    """
    out_dir = os.path.join('outputs', 'tables')
    fig_dir = os.path.join('outputs', 'figures', 'paper')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # A) per-expiration coefficient table
    coef_df = leave_one_expiration_coefficients(
        ticker, date, n_epochs=n_epochs, seed=seed)
    coef_path = os.path.join(out_dir, 'per_expiration_coefficients.csv')
    coef_df.to_csv(coef_path, index=False)

    # C) regime transfer
    regime_rows = regime_transfer_atm_otm(
        ticker=ticker, date=date, n_epochs=n_epochs, seed=seed)
    regime_df = pd.DataFrame(regime_rows)
    gen_path = os.path.join(out_dir, 'generalization_analysis.csv')
    regime_df.to_csv(gen_path, index=False)

    # Figures (A) and (B).
    coef_fig = plot_coefficient_vs_maturity(
        coef_df, save_path=os.path.join(fig_dir, 'coefficient_vs_maturity'))

    # B) Use the existing per_expiration_loo.csv if present; otherwise build
    # one on the fly from coef_df (it already carries R^2 per tau).
    loo_csv = os.path.join(out_dir, 'per_expiration_loo.csv')
    if os.path.isfile(loo_csv):
        loo_df = pd.read_csv(loo_csv)
    else:
        loo_df = coef_df.rename(columns={'tau': 'tau_left_out'})[
            ['ticker', 'tau_left_out', 'R2']].copy()
    heatmap_fig = plot_transfer_heatmap(
        loo_df, save_path=os.path.join(fig_dir, 'transfer_heatmap'))

    return {
        'coef_df': coef_df,
        'regime_df': regime_df,
        'coef_csv': coef_path,
        'generalization_csv': gen_path,
        'coef_figure': coef_fig,
        'heatmap_figure': heatmap_fig,
    }
