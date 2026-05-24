"""
Tests for Dupire CV-based approach selection (PRD Part A).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generation import bs_call_price
from src.real_data_publication import (
    dupire_cv_select_approach,
    _DUPIRE_CV_APPROACHES,
)


def _make_mock_option_data(n_expirations=4, n_strikes_per_exp=18,
                           S0=100.0, r=0.05, sigma=0.20, seed=42):
    """Build a synthetic option_data dict mimicking real chain output."""
    rng = np.random.default_rng(seed)
    taus = np.linspace(0.10, 0.80, n_expirations)
    K_min, K_max = 0.85 * S0, 1.15 * S0

    rows = []
    for tau in taus:
        strikes = np.linspace(K_min, K_max, n_strikes_per_exp)
        for K in strikes:
            iv = float(sigma + 0.01 * rng.standard_normal())
            iv = max(iv, 0.05)
            mid = float(bs_call_price(S0, float(K), r, iv, float(tau)))
            rows.append({
                'strike': float(K),
                'expiration': f"exp_{tau:.4f}",
                'tau': float(tau),
                'bid': mid - 0.02,
                'ask': mid + 0.02,
                'mid_price': mid,
                'implied_vol': iv,
                'volume': 100,
                'openInterest': 200,
                'S0': S0,
                'r': r,
            })

    return {
        'ticker': 'MOCK',
        'S0': S0,
        'r': r,
        'option_df': pd.DataFrame(rows),
        'data_source': 'mock',
    }


def test_cv_runs_on_mock_chain():
    """dupire_cv_select_approach returns a dict with the required keys."""
    option_data = _make_mock_option_data(n_expirations=4)
    res = dupire_cv_select_approach(option_data, ticker='MOCK',
                                    normalize_moneyness=False, seed=42)

    expected_keys = {'ticker', 'approaches_tested', 'mean_errors',
                     'best_approach', 'per_fold_errors_df',
                     'normalize_moneyness'}
    assert expected_keys.issubset(set(res.keys()))
    assert res['ticker'] == 'MOCK'
    assert isinstance(res['mean_errors'], dict)
    assert isinstance(res['per_fold_errors_df'], pd.DataFrame)
    # Should have produced at least some per-fold rows.
    assert len(res['per_fold_errors_df']) > 0


def test_best_approach_in_allowed_list():
    """Returned best_approach is one of the four PRD approaches."""
    option_data = _make_mock_option_data(n_expirations=4)
    res = dupire_cv_select_approach(option_data, ticker='MOCK',
                                    normalize_moneyness=False, seed=42)
    assert res['best_approach'] in set(_DUPIRE_CV_APPROACHES)


def test_moneyness_normalization_doesnt_crash():
    """normalize_moneyness=True completes without error."""
    option_data = _make_mock_option_data(n_expirations=4)
    res = dupire_cv_select_approach(option_data, ticker='MOCK',
                                    normalize_moneyness=True, seed=42)
    assert res['normalize_moneyness'] is True
    assert 'best_approach' in res
