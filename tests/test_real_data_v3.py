"""Tests for src.real_data_v3 (improvements 1-5 on top of v2 pipeline)."""

import numpy as np
import pandas as pd
import pytest

from src.data_generation import bs_call_price
from src.real_data_v2 import (
    build_logm_surface_svi,
    compute_forward_prices,
)
from src.real_data_v3 import (
    bootstrap_sigma_v2,
    compare_sigma_methods,
    per_expiration_sigma,
    quadratic_dupire_logm,
    windowed_2term_dupire_v2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bs_call_with_div(S0, K, r, q, sigma, tau):
    if tau <= 0:
        return float(max(S0 - K, 0.0))
    S_eff = S0 * np.exp(-q * tau)
    return float(bs_call_price(S_eff, K, r, sigma, tau))


def _make_synthetic_chain(S0=100.0, r=0.05, q=0.0, sigma=0.20, seed=42,
                          n_taus=8, n_strikes=25):
    """Synthetic BS option chain on a (K, tau) grid (constant sigma)."""
    rng = np.random.default_rng(seed)
    taus = np.linspace(0.05, 1.0, n_taus)
    strikes = np.linspace(0.7 * S0, 1.3 * S0, n_strikes)

    rows = []
    for tau in taus:
        for K in strikes:
            C = _bs_call_with_div(S0, K, r, q, sigma, tau)
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
                'volume': int(rng.integers(100, 5000)),
                'openInterest': int(rng.integers(500, 20000)),
                'S0': S0,
                'r': r,
            })
    return pd.DataFrame(rows)


def _make_synthetic_chain_quadratic(S0=100.0, r=0.05, q=0.0, sigma0=0.20,
                                    skew_coef=0.05, seed=42):
    """Chain where sigma(k) = sqrt(sigma0^2 + skew_coef * k^2)."""
    rng = np.random.default_rng(seed)
    taus = np.linspace(0.05, 1.0, 8)
    strikes = np.linspace(0.7 * S0, 1.3 * S0, 30)
    rows = []
    for tau in taus:
        F = S0 * np.exp((r - q) * tau)
        for K in strikes:
            k = np.log(K / F)
            sig2 = sigma0 ** 2 + skew_coef * k * k
            sig = float(np.sqrt(max(sig2, 1e-6)))
            C = _bs_call_with_div(S0, K, r, q, sig, tau)
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
                'implied_vol': sig,
                'volume': int(rng.integers(100, 5000)),
                'openInterest': int(rng.integers(500, 20000)),
                'S0': S0,
                'r': r,
            })
    return pd.DataFrame(rows)


def _build_constant_sigma_surface(S0=100.0, r=0.05, q=0.0, sigma=0.20,
                                  n_k=40, n_tau=12):
    """Analytic BS call price grid on uniform (k, tau)."""
    k_grid = np.linspace(-0.25, 0.25, n_k)
    tau_grid = np.linspace(0.05, 1.0, n_tau)
    C = np.zeros((n_k, n_tau))
    for j, tau in enumerate(tau_grid):
        F = S0 * np.exp((r - q) * tau)
        for i, k in enumerate(k_grid):
            K = F * np.exp(k)
            C[i, j] = _bs_call_with_div(S0, K, r, q, sigma, tau)
    return C, k_grid, tau_grid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_windowed_v2_on_synthetic_constant_sigma():
    """Constant sigma=0.20 BS surface -> windowed mean sigma within 20%."""
    sigma_true = 0.20
    C, k_grid, tau_grid = _build_constant_sigma_surface(
        sigma=sigma_true, n_k=40, n_tau=12,
    )
    res = windowed_2term_dupire_v2(
        C, k_grid, tau_grid, window_size=8, stride=3,
        min_r2=0.5, sigma_bounds=(0.01, 2.0),
    )
    assert res['n_valid'] > 0
    assert np.isfinite(res['sigma_mean'])
    rel = abs(res['sigma_mean'] - sigma_true) / sigma_true
    assert rel < 0.20, (
        f"windowed sigma_mean={res['sigma_mean']:.4f} vs true={sigma_true} "
        f"rel_err={rel:.3f} (n_valid={res['n_valid']}/{res['n_total']})"
    )


def test_per_expiration_synthetic_constant():
    """Per-expiration sigma should be approximately flat for constant-sigma surface."""
    sigma_true = 0.20
    C, k_grid, tau_grid = _build_constant_sigma_surface(
        sigma=sigma_true, n_k=40, n_tau=10,
    )
    df = per_expiration_sigma(C, k_grid, tau_grid)
    assert len(df) >= 1
    valid = df.dropna(subset=['sigma_loc'])
    assert len(valid) >= 1
    # Each per-tau sigma should be within 25% of true (FD biased a bit).
    for _, row in valid.iterrows():
        rel = abs(row['sigma_loc'] - sigma_true) / sigma_true
        assert rel < 0.25, (
            f"tau={row['tau']:.3f} sigma_loc={row['sigma_loc']:.4f}, rel={rel:.3f}"
        )


def test_quadratic_dupire_synthetic_constant():
    """Constant sigma=0.20 -> alpha~0.04, beta~0, gamma~0, R^2>0.95."""
    sigma_true = 0.20
    C, k_grid, tau_grid = _build_constant_sigma_surface(
        sigma=sigma_true, n_k=40, n_tau=12,
    )
    res = quadratic_dupire_logm(C, k_grid, tau_grid)
    assert res['r2_score'] > 0.95, f"R^2={res['r2_score']:.4f} below 0.95"
    assert abs(res['alpha'] - sigma_true ** 2) < 0.02, (
        f"alpha={res['alpha']:.4f} vs expected {sigma_true ** 2:.4f}"
    )
    # |beta| should be small (no skew).
    assert abs(res['beta']) < 0.05, f"|beta|={abs(res['beta']):.4f}"
    # |gamma| should be small (no curvature).
    assert abs(res['gamma']) < 0.10, f"|gamma|={abs(res['gamma']):.4f}"


def test_compare_sigma_methods_returns_6_methods():
    """compare_sigma_methods returns 6 rows (sindy, mean, median, vega, volume, atm_truth)."""
    df = _make_synthetic_chain(sigma=0.20)
    option_data = {
        'option_df': df,
        'S0': 100.0,
        'r': 0.05,
    }
    out = compare_sigma_methods(option_data, ticker='TEST', sigma_sindy=0.21)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 6
    methods = set(out['method'].tolist())
    assert methods == {'sindy', 'mean', 'median', 'vega_weighted',
                       'volume_weighted', 'atm_truth'}
    # SINDy row reports its value.
    sindy_row = out[out['method'] == 'sindy'].iloc[0]
    assert abs(sindy_row['sigma'] - 0.21) < 1e-9


def test_bootstrap_returns_valid_ci():
    """Bootstrap with n=20 gives ci_low <= mean <= ci_high."""
    df = _make_synthetic_chain(sigma=0.20, n_taus=6, n_strikes=20)
    option_data = {
        'option_df': df,
        'S0': 100.0,
        'r': 0.05,
    }
    res = bootstrap_sigma_v2(option_data, ticker='TEST', n_bootstrap=20,
                             seed=42, q=0.0, n_k=20)
    assert res['n_success'] >= 5, (
        f"too few successful bootstrap iterations: {res['n_success']}/{res['n_total']}"
    )
    assert np.isfinite(res['sigma_mean'])
    assert res['ci_low'] <= res['sigma_mean'] <= res['ci_high'], (
        f"ci=[{res['ci_low']:.4f},{res['ci_high']:.4f}] mean={res['sigma_mean']:.4f}"
    )


def _build_quadratic_sigma_surface(S0=100.0, r=0.05, q=0.0, sigma0=0.20,
                                   skew_coef=0.05, n_k=40, n_tau=12):
    """Analytic BS surface where IV(k) = sqrt(sigma0^2 + skew_coef*k^2)."""
    k_grid = np.linspace(-0.25, 0.25, n_k)
    tau_grid = np.linspace(0.05, 1.0, n_tau)
    C = np.zeros((n_k, n_tau))
    for j, tau in enumerate(tau_grid):
        F = S0 * np.exp((r - q) * tau)
        for i, k in enumerate(k_grid):
            K = F * np.exp(k)
            sig2 = sigma0 ** 2 + skew_coef * k * k
            sig = float(np.sqrt(max(sig2, 1e-6)))
            C[i, j] = _bs_call_with_div(S0, K, r, q, sig, tau)
    return C, k_grid, tau_grid


def test_quadratic_synthetic_skew():
    """sigma^2(k)=0.04+0.05*k^2 surface -> recovered gamma close to 0.05.

    Uses the analytic BS surface (no SVI round-trip) so the underlying
    sigma(k) is exactly quadratic. Dupire's equation uses *local* sigma
    rather than implied sigma so the recovered gamma will not equal the
    skew coefficient exactly, but should have the right sign and order
    of magnitude.
    """
    sigma0 = 0.20
    skew = 0.05
    C, k_grid, tau_grid = _build_quadratic_sigma_surface(
        sigma0=sigma0, skew_coef=skew, n_k=40, n_tau=12,
    )
    res = quadratic_dupire_logm(C, k_grid, tau_grid)
    # alpha (= sigma^2(k=0)) should be near sigma0^2.
    assert abs(res['alpha'] - sigma0 ** 2) < 0.03, (
        f"alpha={res['alpha']:.4f} vs expected {sigma0 ** 2:.4f}"
    )
    # Dupire's local vol differs from implied vol; the curvature picked up
    # by 2-term OLS is roughly 2 * skew because the local-vol curvature is
    # amplified relative to the implied-vol curvature.
    # PRD prediction: gamma ~= 0.05 * 2 = 0.10, within 30%.
    expected_gamma = 2.0 * skew  # 0.10
    assert res['gamma'] > 0, f"gamma={res['gamma']:.4f} should be positive"
    rel = abs(res['gamma'] - expected_gamma) / expected_gamma
    assert rel < 0.30, (
        f"gamma={res['gamma']:.4f} vs expected {expected_gamma:.4f} rel={rel:.3f}"
    )
    # R^2 should be high (quadratic library captures quadratic surface).
    assert res['r2_score'] > 0.90, f"R^2={res['r2_score']:.4f}"
