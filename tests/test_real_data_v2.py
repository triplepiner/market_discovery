"""Tests for src.real_data_v2 (log-moneyness Dupire pipeline)."""

import numpy as np
import pandas as pd
import pytest

from src.data_generation import bs_call_price
from src.real_data_v2 import (
    KNOWN_DIVIDEND_YIELDS,
    build_logm_surface,
    build_logm_surface_svi,
    compute_forward_prices,
    compute_liquidity_weights,
    compute_log_moneyness,
    dupire_logm_2term,
    dupire_logm_2term_windowed,
    filter_atm,
    fit_svi_slice,
    get_dividend_yield,
    run_improved_real_pipeline,
    run_improved_pipeline_all_tickers,
    weighted_stlsq,
)
from src.sindy_discovery import stlsq


# ---------------------------------------------------------------------------
# Helpers: synthetic option chain
# ---------------------------------------------------------------------------

def _make_synthetic_chain(S0=100.0, r=0.05, q=0.0, sigma=0.20, seed=42):
    """Generate a synthetic BS option chain on a (K, tau) grid."""
    rng = np.random.default_rng(seed)
    taus = np.array([0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0])
    strikes = np.linspace(0.7 * S0, 1.3 * S0, 25)

    rows = []
    for tau in taus:
        for K in strikes:
            S_eff = S0 * np.exp(-q * tau)
            C = float(bs_call_price(S_eff, K, r, sigma, tau))
            if C <= 1e-3:
                continue
            spread = max(C * 0.02, 0.01)
            rows.append({
                'strike': K,
                'tau': tau,
                'expiration': f"exp_{tau:.4f}",
                'bid': C - spread / 2,
                'ask': C + spread / 2,
                'mid_price': C,
                'implied_vol': sigma,
                'volume': rng.integers(100, 5000),
                'openInterest': rng.integers(500, 20000),
                'S0': S0,
                'r': r,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_prices_correct():
    """F(tau=1) = S0*exp(r-q) within 1e-10."""
    S0, r, q = 100.0, 0.05, 0.01
    tau = np.array([0.5, 1.0, 2.0])
    F = compute_forward_prices(S0, r, q, tau)
    expected = S0 * np.exp((r - q) * tau)
    assert np.allclose(F, expected, atol=1e-10)
    assert abs(F[1] - S0 * np.exp(r - q)) < 1e-10


def test_log_moneyness_centered_for_atm():
    """For K=S0=100, F~100, k should be ~ 0."""
    S0, r, q, tau = 100.0, 0.05, 0.05, 1.0  # r == q so F == S0
    F = compute_forward_prices(S0, r, q, tau)
    k = compute_log_moneyness(np.array([100.0]), np.array([F]))
    assert abs(k[0]) < 1e-10

    # Non-zero (r-q): F = S0 * exp(0.04) ~ 104.08, K=100 -> k ~ -0.04
    F2 = compute_forward_prices(S0, 0.05, 0.01, 1.0)
    k2 = compute_log_moneyness(np.array([100.0]), np.array([F2]))
    assert abs(k2[0] - (-0.04)) < 1e-6


def test_svi_recovers_constant_iv():
    """Flat IV -> SVI fit gives flat w(k), implied sigma near input."""
    sigma_true = 0.20
    tau = 0.5
    k_obs = np.linspace(-0.2, 0.2, 30)
    w_obs = np.full_like(k_obs, sigma_true ** 2 * tau)
    fit = fit_svi_slice(k_obs, w_obs)
    assert fit is not None
    # b should be small (no skew/smile).
    assert fit['b'] < 0.05
    # Reconstruct sigma at k=0.
    w_atm = fit['a'] + fit['b'] * (fit['rho'] * (-fit['m'])
                                   + np.sqrt(fit['m'] ** 2 + fit['s'] ** 2))
    sigma_atm = np.sqrt(max(w_atm, 1e-12) / tau)
    assert abs(sigma_atm - sigma_true) < 0.02


def test_logm_surface_built_synthetic():
    """Synthetic BS option chain -> log-moneyness surface built, k centered ~0."""
    df = _make_synthetic_chain(S0=100.0, r=0.05, q=0.01, sigma=0.20)
    surface = build_logm_surface(df, S0=100.0, r=0.05, q=0.01,
                                 n_k=30, k_range=(-0.2, 0.2))
    assert surface['C_surface'].shape == (30, len(surface['tau_grid']))
    # k_grid is centered around 0.
    assert surface['k_grid'].min() == -0.2
    assert surface['k_grid'].max() == 0.2
    assert np.all(np.isfinite(surface['C_surface']))
    assert np.all(surface['C_surface'] > 0)


def test_weighted_stlsq_equals_stlsq_when_uniform():
    """With weights all 1, weighted_stlsq matches unweighted stlsq."""
    rng = np.random.default_rng(0)
    n = 200
    library = rng.standard_normal((n, 5))
    true_coef = np.array([1.0, 0.0, -0.5, 0.0, 0.3])
    target = library @ true_coef + 0.01 * rng.standard_normal(n)

    coef_a, active_a = stlsq(library, target, threshold=0.05)
    coef_b, active_b, _ = weighted_stlsq(library, target, weights=None,
                                         threshold=0.05)
    assert np.allclose(coef_a, coef_b, atol=1e-10)
    assert np.array_equal(active_a, active_b)

    # Also test explicit ones.
    coef_c, active_c, _ = weighted_stlsq(library, target,
                                         weights=np.ones(n), threshold=0.05)
    assert np.allclose(coef_a, coef_c, atol=1e-10)


def test_dupire_logm_2term_synthetic():
    """Generate Dupire-consistent data -> recover sigma within 5%, R^2 > 0.99."""
    # Build a clean BS surface in (k, tau), then run 2-term Dupire on it.
    S0, r, q, sigma_true = 100.0, 0.05, 0.0, 0.20
    n_k, n_tau = 50, 40
    k_grid = np.linspace(-0.25, 0.25, n_k)
    tau_grid = np.linspace(0.05, 1.0, n_tau)

    C = np.zeros((n_k, n_tau))
    for j, tau in enumerate(tau_grid):
        F = S0 * np.exp((r - q) * tau)
        Ks = F * np.exp(k_grid)
        C[:, j] = bs_call_price(S0, Ks, r, sigma_true, tau)

    res = dupire_logm_2term(C, k_grid, tau_grid)
    assert res['r2_score'] > 0.99
    sigma_err = abs(res['sigma_loc_discovered'] - sigma_true) / sigma_true
    assert sigma_err < 0.05, f"sigma error too large: {sigma_err}"


def test_atm_filter_nonempty():
    """ATM filter on synthetic chain produces >= 30 options."""
    df = _make_synthetic_chain(S0=100.0, r=0.05, q=0.01, sigma=0.20)
    atm = filter_atm(df, k_low=-0.08, k_high=0.08, S0=100.0, r=0.05, q=0.01)
    assert len(atm) >= 30, f"only {len(atm)} ATM options"


def test_improved_pipeline_runs_on_mock():
    """Full pipeline runs on a mock per_ticker_results entry without error."""
    df = _make_synthetic_chain(S0=100.0, r=0.05, q=0.0, sigma=0.20)
    option_data = {
        'S0': 100.0,
        'r': 0.05,
        'ticker': 'TEST',
        'option_df': df,
        'implied_vols': np.full(len(df), 0.20),
    }
    res = run_improved_real_pipeline(option_data, 'TEST',
                                     use_svi=True, use_weights=True,
                                     run_atm=True, run_windowed=True)
    assert res['ticker'] == 'TEST'
    assert res['full_range'] is not None
    assert np.isfinite(res['full_range']['r2_score'])
    assert res['full_range']['sigma_loc_discovered'] > 0


def test_known_dividend_yields_present():
    """Sanity: dividend yield table has SPY/QQQ/AAPL/MSFT."""
    for t in ('SPY', 'QQQ', 'AAPL', 'MSFT'):
        assert t in KNOWN_DIVIDEND_YIELDS
    # Fallback returns known value when yfinance fails / missing.
    q_spy = KNOWN_DIVIDEND_YIELDS['SPY']
    assert 0 <= q_spy < 0.1


def test_run_improved_pipeline_all_tickers_smoke():
    """Multi-ticker orchestrator returns a DataFrame with one row per ticker."""
    df1 = _make_synthetic_chain(S0=100.0, r=0.05, q=0.0, sigma=0.20, seed=1)
    df2 = _make_synthetic_chain(S0=200.0, r=0.05, q=0.005, sigma=0.25, seed=2)
    per_ticker = {
        'A': {'option_data': {'S0': 100.0, 'r': 0.05, 'option_df': df1,
                              'implied_vols': np.full(len(df1), 0.20)}},
        'B': {'option_data': {'S0': 200.0, 'r': 0.05, 'option_df': df2,
                              'implied_vols': np.full(len(df2), 0.25)}},
    }
    out = run_improved_pipeline_all_tickers(per_ticker, use_svi=False,
                                            use_weights=False,
                                            run_atm=False,
                                            run_windowed=False)
    assert 'summary_df' in out
    assert len(out['summary_df']) == 2
    assert set(out['summary_df']['ticker']) == {'A', 'B'}
