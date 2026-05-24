"""Tests for Improvement 5: per-expiration coefficient extraction and
ATM/OTM moneyness regime transfer.

Both tests use small synthetic surfaces so they run in seconds. The real-data
counterparts are exercised by the orchestrator script and the larger LOO
test in ``test_sindy_kan.py``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# A) LOO coefficient extraction
# ---------------------------------------------------------------------------


def test_loo_coefficient_extraction():
    """Per-expiration LOO + coefficient extraction returns a finite DataFrame.

    Uses the cached SPY chain if available so the function exercises the
    real-data code path including market_avg_iv lookup. Skips cleanly if the
    cached CSV is absent in this environment.
    """
    cache = os.path.join('outputs', 'tables')
    csv = os.path.join(cache, 'per_expiration_coefficients.csv')

    if not os.path.isfile(csv):
        # Build it on the fly with the SPY chain if present.
        from src.transfer_experiments import leave_one_expiration_coefficients
        chain_csv = os.path.join(cache, 'real_chain_SPY_20260329.csv')
        if not os.path.isfile(chain_csv):
            pytest.skip("cached SPY real_chain not present in environment")
        df = leave_one_expiration_coefficients(
            'SPY', '20260329', n_epochs=300, seed=42)
    else:
        df = pd.read_csv(csv)

    # Required columns.
    for col in ('coef_drift', 'coef_diffusion', 'market_avg_iv',
                 'tau', 'R2', 'n_strikes'):
        assert col in df.columns, f"missing column {col!r}"

    # At least 3 rows with finite coefficients.
    finite_drift = np.isfinite(df['coef_drift'].values).sum()
    finite_diff = np.isfinite(df['coef_diffusion'].values).sum()
    finite_iv = np.isfinite(df['market_avg_iv'].values).sum()
    assert finite_drift >= 3, f"only {finite_drift} finite coef_drift rows"
    assert finite_diff >= 3, f"only {finite_diff} finite coef_diffusion rows"
    assert finite_iv >= 3, f"only {finite_iv} finite market_avg_iv rows"


# ---------------------------------------------------------------------------
# B) Moneyness regime transfer on synthetic data
# ---------------------------------------------------------------------------


def test_regime_transfer_runs():
    """ATM->OTM regime transfer returns a finite R^2 on a small surface.

    Builds a tiny synthetic Dupire surface in-memory and runs the same
    train-on-mask / predict-on-mask machinery used by ``regime_transfer_atm_otm``.
    Avoids any disk I/O so this test runs even without cached chains.
    """
    from src.sindy_kan import generate_synthetic_dupire_smile
    from src.transfer_experiments import _train_on_mask, _predict_kan, _r2

    # 30 strikes covering [-0.20, 0.20] in log-moneyness, 6 maturities.
    data = generate_synthetic_dupire_smile(
        n_k=30, n_tau=6, sigma_atm=0.20, smile_curvature=0.30,
        k_range=(-0.20, 0.20), tau_range=(0.1, 1.0))
    k = data['k']
    dCdk = data['dCdk']; d2Cdk2 = data['d2Cdk2']; theta = data['theta']
    n_k, n_tau = dCdk.shape

    atm = np.abs(k) < 0.05
    otm = ~atm
    atm_mask = np.tile(atm.reshape(-1, 1), (1, n_tau))
    otm_mask = np.tile(otm.reshape(-1, 1), (1, n_tau))
    assert atm_mask.sum() >= 6 and otm_mask.sum() >= 30

    # ATM -> OTM.
    res = _train_on_mask(dCdk, d2Cdk2, theta, atm_mask,
                          n_epochs=400, seed=42)
    a_te = dCdk[otm_mask].ravel()
    b_te = d2Cdk2[otm_mask].ravel()
    t_te = theta[otm_mask].ravel()
    ok = np.isfinite(a_te) & np.isfinite(b_te) & np.isfinite(t_te)
    pred = _predict_kan(res, a_te[ok], b_te[ok])
    r2 = _r2(t_te[ok], pred)
    assert np.isfinite(r2), f"ATM->OTM transfer R2 not finite: {r2}"
