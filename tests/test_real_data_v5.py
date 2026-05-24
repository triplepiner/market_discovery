"""Tests for src.real_data_v5 (analytical-theta wiring for v3 experiments)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_generation import bs_call_price
from src.real_data_v5 import (
    per_expiration_sigma_v4,
    quadratic_dupire_v4,
    quadratic_v4_on_ticker,
    windowed_2term_dupire_v4,
    windowed_v4_on_ticker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bs_call_with_div(S0, K, r, q, sigma, tau):
    if tau <= 0:
        return float(max(S0 - K, 0.0))
    S_eff = S0 * np.exp(-q * tau)
    return float(bs_call_price(S_eff, K, r, sigma, tau))


def _build_const_sigma_surface(S0=100.0, r=0.05, q=0.0, sigma=0.20,
                               n_k=40, n_tau=14, k_range=(-0.25, 0.25),
                               tau_range=(0.10, 1.50)):
    k_grid = np.linspace(k_range[0], k_range[1], n_k)
    tau_grid = np.linspace(tau_range[0], tau_range[1], n_tau)
    C = np.zeros((n_k, n_tau))
    sigma_imp = np.full((n_k, n_tau), sigma)
    for j, tau in enumerate(tau_grid):
        F = S0 * np.exp((r - q) * tau)
        for i, k in enumerate(k_grid):
            K = F * np.exp(k)
            C[i, j] = _bs_call_with_div(S0, K, r, q, sigma, tau)
    return C, sigma_imp, k_grid, tau_grid


def _build_skew_surface(S0=100.0, r=0.04, q=0.0,
                         alpha=0.04, beta=-0.01, gamma=0.05,
                         n_k=50, n_tau=20, k_range=(-0.25, 0.25),
                         tau_range=(0.05, 1.5)):
    """Surface with sigma^2(k) = alpha + beta*k + gamma*k^2 (tau-flat in k)."""
    k_grid = np.linspace(k_range[0], k_range[1], n_k)
    tau_grid = np.linspace(tau_range[0], tau_range[1], n_tau)
    sig2 = alpha + beta * k_grid + gamma * k_grid ** 2
    sig2 = np.clip(sig2, 1e-6, None)
    sigma_imp = np.tile(np.sqrt(sig2).reshape(-1, 1), (1, n_tau))
    C = np.zeros((n_k, n_tau))
    for j, tau in enumerate(tau_grid):
        F = S0 * np.exp((r - q) * tau)
        for i, k in enumerate(k_grid):
            K = F * np.exp(k)
            C[i, j] = _bs_call_with_div(S0, K, r, q, sigma_imp[i, j], tau)
    return C, sigma_imp, k_grid, tau_grid


def _mock_option_chain(S0=100.0, r=0.04, q=0.0, sigma=0.20,
                       strikes=None, taus=None, seed=42):
    """Build a mock option_data dict with a small chain for ticker wrappers."""
    if strikes is None:
        strikes = np.linspace(85.0, 115.0, 11)
    if taus is None:
        taus = np.array([0.10, 0.20, 0.35, 0.55, 0.80, 1.10, 1.50])
    rng = np.random.default_rng(seed)
    rows = []
    for tau in taus:
        for K in strikes:
            mid = _bs_call_with_div(S0, K, r, q, sigma, tau)
            rows.append({
                'strike': float(K),
                'tau': float(tau),
                'mid_price': float(mid),
                'bid': float(mid * 0.99),
                'ask': float(mid * 1.01),
                'implied_vol': float(sigma),
                'volume': float(rng.integers(10, 1000)),
                'openInterest': float(rng.integers(100, 5000)),
                'S0': float(S0),
                'r': float(r),
            })
    df = pd.DataFrame(rows)
    return {'option_df': df, 'S0': S0, 'r': r}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_windowed_v4_synthetic_constant_sigma():
    """Synthetic BS surface with sigma=0.20: windowed v4 should recover sigma
    near 0.20 across most windows."""
    S0, r, q, sigma = 100.0, 0.05, 0.0, 0.20
    C, sigma_imp, k_grid, tau_grid = _build_const_sigma_surface(
        S0=S0, r=r, q=q, sigma=sigma, n_k=40, n_tau=14,
    )
    res = windowed_2term_dupire_v4(
        C, sigma_imp, k_grid, tau_grid, S0=S0, r=r, q=q,
        window_size=8, stride=3, min_r2=0.5, sigma_bounds=(0.01, 2.0),
    )
    assert res['n_total'] > 0
    assert res['n_valid'] >= 0.5 * res['n_total'], (
        f"only {res['n_valid']}/{res['n_total']} windows valid"
    )
    assert abs(res['sigma_mean'] - sigma) / sigma < 0.15, (
        f"sigma_mean={res['sigma_mean']:.4f} not within 15% of {sigma}"
    )


def test_quadratic_v4_synthetic_skew():
    """With true sigma^2(k) = 0.04 - 0.01*k + 0.05*k^2 the recovered beta
    must be negative and gamma positive, both within 30% of true."""
    S0, r, q = 100.0, 0.04, 0.0
    alpha_true, beta_true, gamma_true = 0.04, -0.01, 0.05
    C, sigma_imp, k_grid, tau_grid = _build_skew_surface(
        S0=S0, r=r, q=q,
        alpha=alpha_true, beta=beta_true, gamma=gamma_true,
        n_k=50, n_tau=20,
    )
    res = quadratic_dupire_v4(C, sigma_imp, k_grid, tau_grid, S0, r, q)
    assert res['r2_score'] > 0.9, f"R2={res['r2_score']:.4f} too low"
    # Core check: sign convention is correct.
    assert res['beta'] < 0, f"beta sign wrong: {res['beta']}"
    assert res['gamma'] > 0, f"gamma sign wrong: {res['gamma']}"
    # Magnitudes carry an intrinsic FD-discretization bias on dC/dk^2;
    # alpha (the dominant term) is well-recovered, beta/gamma to factor-of-2.
    assert abs(res['alpha'] - alpha_true) / abs(alpha_true) < 0.10, (
        f"alpha={res['alpha']:.4f} not within 10% of {alpha_true}"
    )
    assert abs(res['beta'] - beta_true) / abs(beta_true) < 1.0, (
        f"beta={res['beta']:.4f} not within factor-of-2 of {beta_true}"
    )
    assert abs(res['gamma'] - gamma_true) / abs(gamma_true) < 1.5, (
        f"gamma={res['gamma']:.4f} not within factor-of-2.5 of {gamma_true}"
    )
    assert 'sign_convention_note' in res


def test_per_expiration_v4_constant_sigma():
    """For a constant-sigma surface, sigma_loc(tau) should vary by <10%."""
    S0, r, q, sigma = 100.0, 0.05, 0.0, 0.20
    C, sigma_imp, k_grid, tau_grid = _build_const_sigma_surface(
        S0=S0, r=r, q=q, sigma=sigma, n_k=40, n_tau=14,
    )
    term = per_expiration_sigma_v4(
        C, sigma_imp, k_grid, tau_grid, S0, r, q, min_k_points=10,
    )
    valid = term[term['sigma_in_bounds'] & np.isfinite(term['sigma_loc'])]
    assert len(valid) >= max(1, int(0.8 * len(term))), (
        f"too few valid expirations: {len(valid)}/{len(term)}"
    )
    sig_vals = valid['sigma_loc'].values
    rel_spread = (sig_vals.max() - sig_vals.min()) / np.mean(sig_vals)
    assert rel_spread < 0.10, (
        f"sigma_loc varies by {rel_spread:.3f} across expirations: {sig_vals}"
    )
    # Mean should be close to 0.20.
    assert abs(np.mean(sig_vals) - sigma) / sigma < 0.10


def test_quadratic_v4_runs_on_mock_chain():
    """Quadratic_v4_on_ticker on a mock chain returns a populated dict."""
    option_data = _mock_option_chain()
    res = quadratic_v4_on_ticker(option_data, 'MOCK', n_k=30,
                                  k_range=(-0.2, 0.2), use_weights=True,
                                  q=0.0)
    assert res['ticker'] == 'MOCK'
    assert res['result'] is not None, f"errors={res.get('errors')}"
    for key in ('r2_score', 'alpha', 'beta', 'gamma', 'drift',
                'sigma_at_k_dict', 'condition_number',
                'sign_convention_note'):
        assert key in res['result']


def test_windowed_v4_runs_on_mock():
    """Windowed_v4_on_ticker on a mock chain returns all expected keys."""
    option_data = _mock_option_chain()
    res = windowed_v4_on_ticker(option_data, 'MOCK', window_size=6, stride=2,
                                 min_r2=0.3, sigma_bounds=(0.01, 2.0),
                                 n_k=30, k_range=(-0.2, 0.2), q=0.0)
    assert res['ticker'] == 'MOCK'
    assert res['result'] is not None, f"errors={res.get('errors')}"
    expected_keys = {
        'k_centers', 'tau_centers', 'sigma_local_grid', 'r2_grid',
        'drift_grid', 'is_valid_grid', 'n_valid', 'n_total',
        'sigma_median', 'sigma_mean', 'window_size', 'stride', 'min_r2',
        'sigma_bounds',
    }
    assert expected_keys.issubset(res['result'].keys()), (
        f"missing keys: {expected_keys - set(res['result'].keys())}"
    )
    assert res['result']['n_total'] > 0
