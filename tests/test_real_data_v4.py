"""Tests for src.real_data_v4 (analytical-theta Dupire stack)."""

import numpy as np
import pandas as pd
import pytest

from src.data_generation import bs_call_price
from src.real_data_v4 import (
    bs_theta_analytical,
    direct_dupire_local_vol,
    dupire_2term_analytical_theta,
    reconstruct_sigma_imp_grid,
    run_v4_experiments_on_ticker,
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
                               n_k=40, n_tau=12, k_range=(-0.25, 0.25),
                               tau_range=(0.10, 1.0)):
    """Analytic BS call grid on uniform (k, tau)."""
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


def _build_smile_surface(S0=100.0, r=0.05, q=0.0, sigma0=0.20, skew=0.005,
                         n_k=40, n_tau=12, k_range=(-0.25, 0.25),
                         tau_range=(0.10, 1.0)):
    """Analytic BS surface where sigma^2(k) = sigma0^2 + skew*k^2 (in IV space)."""
    k_grid = np.linspace(k_range[0], k_range[1], n_k)
    tau_grid = np.linspace(tau_range[0], tau_range[1], n_tau)
    C = np.zeros((n_k, n_tau))
    sigma_imp = np.zeros((n_k, n_tau))
    for j, tau in enumerate(tau_grid):
        F = S0 * np.exp((r - q) * tau)
        for i, k in enumerate(k_grid):
            sig2 = sigma0 ** 2 + skew * k * k
            sig = float(np.sqrt(max(sig2, 1e-6)))
            sigma_imp[i, j] = sig
            K = F * np.exp(k)
            C[i, j] = _bs_call_with_div(S0, K, r, q, sig, tau)
    return C, sigma_imp, k_grid, tau_grid


def _make_mock_chain(S0=100.0, r=0.05, sigma=0.20, seed=42):
    rng = np.random.default_rng(seed)
    taus = np.linspace(0.05, 1.0, 8)
    strikes = np.linspace(0.7 * S0, 1.3 * S0, 25)
    rows = []
    for tau in taus:
        for K in strikes:
            C = _bs_call_with_div(S0, K, r, 0.0, sigma, tau)
            if C <= 1e-3:
                continue
            spread = max(C * 0.02, 0.01)
            rows.append({
                'strike': K, 'tau': tau,
                'expiration': f"exp_{tau:.4f}",
                'bid': C - spread / 2, 'ask': C + spread / 2,
                'mid_price': C, 'implied_vol': sigma,
                'volume': int(rng.integers(100, 5000)),
                'openInterest': int(rng.integers(500, 20000)),
                'S0': S0, 'r': r,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_analytical_theta_matches_fd_on_synthetic_bs():
    """Analytical theta vs FD theta agree within 0.5% RMS away from boundaries.

    Dupire's analytical theta is dC/dtau at *fixed strike K*. Computing FD
    theta at fixed log-moneyness k differs by a chain-rule term because
    K = F(tau)*exp(k) varies with tau. We therefore build the test surface
    on a fixed (K, tau) grid so FD along tau directly approximates the
    analytical theta.
    """
    sigma = 0.20
    S0, r, q = 100.0, 0.05, 0.0
    tau_grid = np.linspace(0.10, 1.20, 32)
    K_grid_1d = np.linspace(80.0, 130.0, 40)
    n_k, n_tau = len(K_grid_1d), len(tau_grid)
    sigma_imp = np.full((n_k, n_tau), sigma)
    C = np.zeros((n_k, n_tau))
    for j, tau in enumerate(tau_grid):
        for i, K in enumerate(K_grid_1d):
            C[i, j] = _bs_call_with_div(S0, K, r, q, sigma, tau)

    # Pass 2D K grid (constant along tau axis) so bs_theta_analytical knows
    # the strike is the same at every column.
    K_2d = np.tile(K_grid_1d.reshape(-1, 1), (1, n_tau))
    theta_a = bs_theta_analytical(S0, K_2d, tau_grid, sigma_imp, r, q)
    theta_fd = np.gradient(C, tau_grid, axis=1, edge_order=2)

    # Compare on the interior (skip tau boundaries).
    inter = slice(2, -2)
    diff = theta_a[:, inter] - theta_fd[:, inter]
    rms = float(np.sqrt(np.mean(diff ** 2)))
    scale = float(np.sqrt(np.mean(theta_a[:, inter] ** 2)))
    rel_rms = rms / scale
    assert rel_rms < 0.005, (
        f"analytical vs FD theta relative RMS = {rel_rms:.4f} > 0.005"
    )


def test_dupire_analytical_theta_recovers_sigma():
    """sigma=0.20 -> recovered sigma within 5%, R^2 > 0.99."""
    sigma_true = 0.20
    C, sigma_imp, k_grid, tau_grid = _build_const_sigma_surface(
        sigma=sigma_true, n_k=40, n_tau=12,
    )
    S0, r, q = 100.0, 0.05, 0.0
    res = dupire_2term_analytical_theta(
        C, sigma_imp, k_grid, tau_grid, S0, r, q,
    )
    assert res['r2_score'] > 0.99, f"R^2={res['r2_score']:.5f} below 0.99"
    rel = abs(res['sigma_loc_discovered'] - sigma_true) / sigma_true
    assert rel < 0.05, (
        f"sigma_loc={res['sigma_loc_discovered']:.4f} vs true={sigma_true}, "
        f"rel={rel:.4f}"
    )


def test_direct_dupire_on_constant_sigma():
    """Constant sigma=0.20 -> direct formula recovers ~0.20 everywhere."""
    sigma_true = 0.20
    C, sigma_imp, k_grid, tau_grid = _build_const_sigma_surface(
        sigma=sigma_true, n_k=40, n_tau=12,
    )
    S0, r, q = 100.0, 0.05, 0.0
    res = direct_dupire_local_vol(C, sigma_imp, k_grid, tau_grid, S0, r, q)
    sigma_grid = res['sigma_loc_grid']
    # Look at interior pixels (avoid boundary FD artifacts).
    inter = sigma_grid[3:-3, 1:-1]
    finite = inter[np.isfinite(inter)]
    assert finite.size > 0
    med = float(np.median(finite))
    rel = abs(med - sigma_true) / sigma_true
    assert rel < 0.10, (
        f"median sigma_loc={med:.4f} vs true={sigma_true}, rel={rel:.4f}; "
        f"valid_pct={res['n_valid_pct']:.3f}"
    )
    assert res['n_valid_pct'] > 0.5, (
        f"only {res['n_valid_pct']:.3f} valid pixels"
    )


def test_direct_dupire_smile_recovered():
    """sigma^2(k) = sigma0^2 + skew*k^2 -> direct sigma_loc grid is non-flat.

    The local-vol curvature is roughly double the implied-vol curvature, so a
    moderate IV-side skew already gives a clearly non-flat sigma_loc(k) when
    recovered via the direct Dupire formula.
    """
    C, sigma_imp, k_grid, tau_grid = _build_smile_surface(
        sigma0=0.20, skew=0.50, n_k=40, n_tau=12,
    )
    S0, r, q = 100.0, 0.05, 0.0
    res = direct_dupire_local_vol(C, sigma_imp, k_grid, tau_grid, S0, r, q)
    sigma_grid = res['sigma_loc_grid']
    inter = sigma_grid[3:-3, 1:-1]
    finite = inter[np.isfinite(inter)]
    assert finite.size > 0
    # Variance across the surface must be non-trivial (parabolic shape).
    var_sigma = float(np.var(finite))
    assert var_sigma > 1e-3, (
        f"direct sigma_loc variance {var_sigma:.6f} too small for a smile"
    )
    # Median sigma should still be in a sensible range.
    med = float(np.median(finite))
    assert 0.15 < med < 0.40, f"median sigma_loc {med:.4f} outside [0.15, 0.40]"


def test_v4_runs_on_mock_chain():
    """run_v4_experiments_on_ticker on a small mock chain returns expected keys."""
    df = _make_mock_chain(sigma=0.20)
    option_data = {
        'option_df': df,
        'S0': 100.0,
        'r': 0.05,
        'implied_vols': np.full(len(df), 0.20),
    }
    res = run_v4_experiments_on_ticker(
        option_data, ticker='TEST', n_bootstrap=5, seed=42, n_k=20,
    )
    expected = {'ticker', 'q', 'global_2term', 'quadratic', 'windowed',
                'bootstrap', 'sigma_comparison', 'direct_dupire',
                'svi_stats', 'errors'}
    assert expected.issubset(set(res.keys())), (
        f"missing keys: {expected - set(res.keys())}"
    )
    # Sanity: the global 2-term should not crash and should produce a number.
    g2 = res.get('global_2term')
    assert g2 is not None
    assert np.isfinite(g2.get('sigma_loc_discovered', float('nan')))


def test_reconstruct_sigma_imp_grid_smoke():
    """reconstruct_sigma_imp_grid handles empty fits + valid fits."""
    k_grid = np.linspace(-0.2, 0.2, 10)
    tau_grid = np.linspace(0.1, 1.0, 5)
    # Empty list -> default 0.20 grid.
    out_empty = reconstruct_sigma_imp_grid([], k_grid, tau_grid)
    assert out_empty.shape == (10, 5)
    assert np.allclose(out_empty, 0.20)

    # A plausible SVI fit -> finite positive sigma surface.
    fit = {'a': 0.01, 'b': 0.05, 'rho': -0.3, 'm': 0.0, 's': 0.2,
           'tau': 0.5, 'method': 'svi'}
    out = reconstruct_sigma_imp_grid([fit], k_grid, tau_grid)
    assert out.shape == (10, 5)
    assert np.all(np.isfinite(out))
    assert np.all(out > 0)
