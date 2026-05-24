"""
Tests for src.real_data_publication.

Builds small synthetic surfaces that mirror the per_ticker_results structure
returned by src.real_data.run_real_data_experiment, then validates that the
new publication-readiness functions (GP-SINDy, GP-Dupire, windowed local
vol extraction) execute end-to-end and return sensible outputs.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generation import bs_call_price
from src.real_data_publication import (
    run_gp_sindy_on_real_data,
    compare_derivative_methods_on_real_data,
    run_gp_dupire_on_real_data,
    compare_dupire_methods,
    windowed_local_vol_extraction,
)


def _make_bs_surface(S0=100.0, r=0.05, sigma=0.2,
                    K_min=80.0, K_max=120.0, n_K=30,
                    tau_min=0.05, tau_max=1.0, n_tau=30):
    """Build an analytical BS call surface arranged as (K, tau)."""
    K_grid = np.linspace(K_min, K_max, n_K, dtype=np.float64)
    tau_grid = np.linspace(tau_min, tau_max, n_tau, dtype=np.float64)
    C = np.zeros((n_K, n_tau), dtype=np.float64)
    for j, tau in enumerate(tau_grid):
        C[:, j] = bs_call_price(S0, K_grid, r, sigma, float(tau))
    return C, K_grid, tau_grid


def _make_mock_per_ticker_results(tickers=('SPY', 'QQQ'), sigma=0.2):
    """Build a per_ticker_results-style dict from analytical BS surfaces."""
    out = {}
    for t in tickers:
        C, K_grid, tau_grid = _make_bs_surface(
            S0=100.0, r=0.05, sigma=sigma,
            K_min=80.0, K_max=120.0, n_K=25,
            tau_min=0.1, tau_max=1.0, n_tau=25,
        )
        out[t] = {
            'ticker': t,
            'avg_implied_vol': sigma,
            'sigma_effective': sigma,
            'data_source': 'mock',
            'option_data': {
                'ticker': t,
                'S0': 100.0,
                'r': 0.05,
                'data_source': 'mock',
            },
            'surface_data': {
                'V_surface': C,
                'K_grid': K_grid,
                'tau_grid': tau_grid,
                'S0': 100.0,
                'r': 0.05,
            },
        }
    return out


# ---------------------------------------------------------------------------
# 1. GP-SINDy on real-data wrapper
# ---------------------------------------------------------------------------

def test_gp_sindy_real_runs_without_error():
    """run_gp_sindy_on_real_data executes and returns the documented keys."""
    per_ticker = _make_mock_per_ticker_results(tickers=('SPY',))
    out = run_gp_sindy_on_real_data(per_ticker, standardize=True)

    assert 'SPY' in out
    entry = out['SPY']
    if 'error' in entry:
        pytest.fail(
            f"GP-SINDy unexpectedly failed: {entry['error']}: {entry.get('message')}"
        )
    expected = {
        'ticker', 'gp_r2', 'gp_coefficients', 'gp_active_terms',
        'sigma_discovered', 'gp_pde', 'gp_n_active',
    }
    assert expected.issubset(set(entry.keys()))
    assert np.isfinite(entry['gp_r2'])
    assert len(entry['gp_coefficients']) == 5


# ---------------------------------------------------------------------------
# 2. Comparative DataFrame
# ---------------------------------------------------------------------------

def test_compare_derivative_methods_returns_dataframe():
    """compare_derivative_methods_on_real_data returns a DataFrame with the
    expected columns and one row per ticker."""
    per_ticker = _make_mock_per_ticker_results(tickers=('SPY', 'QQQ'))
    df = compare_derivative_methods_on_real_data(per_ticker, standardize=True)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    expected_cols = {
        'ticker', 'r2_fd', 'r2_savgol', 'r2_gp',
        'sigma_fd', 'sigma_savgol', 'sigma_gp',
        'coeffs_fd', 'coeffs_savgol', 'coeffs_gp',
    }
    assert expected_cols.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# 3. GP-Dupire on real data
# ---------------------------------------------------------------------------

def test_gp_dupire_real_runs():
    """run_gp_dupire_on_real_data runs to completion on mock data."""
    per_ticker = _make_mock_per_ticker_results(tickers=('SPY',))
    out = run_gp_dupire_on_real_data(per_ticker, standardize=True)
    assert 'SPY' in out
    entry = out['SPY']
    if 'error' in entry:
        pytest.fail(
            f"GP-Dupire failed: {entry['error']}: {entry.get('message')}"
        )
    assert np.isfinite(entry['r2_score'])
    assert len(entry['discovered_coefficients']) == 5
    # On a clean BS surface GP-Dupire should at least pick up SOME sigma.
    assert np.isfinite(entry['sigma_discovered']) or entry['n_active'] >= 1


def test_compare_dupire_methods_returns_dataframe():
    """compare_dupire_methods returns a DataFrame contrasting FD vs GP."""
    per_ticker = _make_mock_per_ticker_results(tickers=('SPY',))
    df = compare_dupire_methods(per_ticker)
    assert isinstance(df, pd.DataFrame)
    assert {'ticker', 'r2_fd', 'r2_gp',
            'sigma_fd', 'sigma_gp'}.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# 4. Windowed local-vol extraction shape
# ---------------------------------------------------------------------------

def test_windowed_local_vol_shape():
    """windowed_local_vol_extraction returns a sigma_local_grid of the
    expected (n_K_windows, n_tau_windows) shape."""
    per_ticker = _make_mock_per_ticker_results(tickers=('SPY',))
    surface = per_ticker['SPY']['surface_data']
    window_size = 11
    stride = 3
    res = windowed_local_vol_extraction(
        surface, ticker='SPY',
        window_size=window_size, stride=stride, min_r2=0.0,
    )

    n_K, n_tau = surface['V_surface'].shape
    expected_rows = len(range(0, n_K - window_size + 1, stride))
    expected_cols = len(range(0, n_tau - window_size + 1, stride))

    assert res['sigma_local_grid'].shape == (expected_rows, expected_cols)
    assert res['r2_grid'].shape == (expected_rows, expected_cols)
    assert res['n_total_windows'] == expected_rows * expected_cols
    assert 0 <= res['n_valid_windows'] <= res['n_total_windows']


# ---------------------------------------------------------------------------
# 5. Constant-sigma BS surface -> recovered local sigma roughly constant
# ---------------------------------------------------------------------------

def test_windowed_local_vol_constant_sigma():
    """On a constant-sigma analytical surface, the recovered local sigma
    should be approximately constant with mean within 30% of the true value."""
    true_sigma = 0.20
    # A larger, denser surface gives the windowed GP enough signal.
    C, K_grid, tau_grid = _make_bs_surface(
        S0=100.0, r=0.05, sigma=true_sigma,
        K_min=80.0, K_max=120.0, n_K=40,
        tau_min=0.1, tau_max=1.5, n_tau=40,
    )
    surface = {
        'V_surface': C,
        'K_grid': K_grid,
        'tau_grid': tau_grid,
        'S0': 100.0,
        'r': 0.05,
    }

    res = windowed_local_vol_extraction(
        surface, ticker='SYNTH',
        window_size=15, stride=5, min_r2=0.3,
    )

    sigmas = res['sigma_local_grid']
    finite = sigmas[np.isfinite(sigmas)]
    assert finite.size >= 1, (
        f"No valid windows recovered (n_valid={res['n_valid_windows']} / "
        f"{res['n_total_windows']})."
    )

    mean_sigma = float(np.mean(finite))
    rel_err = abs(mean_sigma - true_sigma) / true_sigma
    assert rel_err < 0.30, (
        f"Recovered local sigma mean {mean_sigma:.4f} differs from "
        f"true {true_sigma:.4f} by {rel_err:.2%} (>30%)."
    )
