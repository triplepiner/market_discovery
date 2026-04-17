"""
Tests for real data analysis module (Improvements 1-4).

Tests analyze_discovered_pde, dividend_yield_discovery, merton_real_data_bridge,
iv_regime_analysis, and helper functions.
"""

import numpy as np
import pandas as pd
import pytest

from src.real_data_analysis import (
    analyze_discovered_pde,
    dividend_yield_discovery,
    merton_real_data_bridge,
    iv_regime_analysis,
    _cosine_similarity,
)


class TestAnalyzeDiscoveredPde:
    def test_returns_expected_keys(self):
        """analyze_discovered_pde returns all expected keys."""
        sindy_result = {
            'discovered_coefficients': np.array([0.05, -0.01, 0.5, -0.04, -0.018]),
        }
        result = analyze_discovered_pde(sindy_result, S0=100, r=0.05,
                                         avg_iv=0.2, ticker='TEST')

        expected_keys = {
            'ticker', 'S0', 'r_fetched', 'avg_iv',
            'discovered_coefficients', 'sigma_discovered', 'sigma_ratio',
            'r_discovered', 'q_implied', 'r_plausible', 'q_plausible',
            'sigma_plausible', 'jump_signature', 'bs_theory_coefficients',
            'term_comparison',
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_effective_sigma_positive(self):
        """If S2*d2V/dS2 coefficient is negative, sigma_discovered is positive."""
        sindy_result = {
            'discovered_coefficients': np.array([0.05, 0.0, 0.0, -0.05, -0.02]),
        }
        result = analyze_discovered_pde(sindy_result, S0=100, r=0.05,
                                         avg_iv=0.2, ticker='TEST')

        assert result['sigma_discovered'] is not None
        assert result['sigma_discovered'] > 0
        # sigma = sqrt(-2 * -0.02) = sqrt(0.04) = 0.2
        np.testing.assert_allclose(result['sigma_discovered'], 0.2, atol=1e-10)

    def test_sigma_none_when_positive_coeff(self):
        """sigma_discovered is None when S2*d2V/dS2 is positive or zero."""
        sindy_result = {
            'discovered_coefficients': np.array([0.05, 0.0, 0.0, -0.05, 0.01]),
        }
        result = analyze_discovered_pde(sindy_result, S0=100, r=0.05,
                                         avg_iv=0.2, ticker='TEST')
        assert result['sigma_discovered'] is None

    def test_term_comparison_length(self):
        """term_comparison has 5 entries (one per library term)."""
        sindy_result = {
            'discovered_coefficients': np.array([0.05, 0.0, 0.0, -0.05, -0.02]),
        }
        result = analyze_discovered_pde(sindy_result, S0=100, r=0.05,
                                         avg_iv=0.2, ticker='TEST')
        assert len(result['term_comparison']) == 5


class TestDividendYieldDiscovery:
    def test_dividend_yield_reasonable(self):
        """Known dividend yield case: S*dV/dS coeff = -(r-q), q should match."""
        # r = 0.05, q = 0.015 -> c_SdVdS = -(0.05 - 0.015) = -0.035
        sindy_result = {
            'discovered_coefficients': np.array([0.05, 0.0, 0.0, -0.035, -0.02]),
        }
        result = dividend_yield_discovery(sindy_result, r_fetched=0.05,
                                           ticker='TEST')

        # q_implied = r - (-c3) = 0.05 - 0.035 = 0.015
        np.testing.assert_allclose(result['q_implied'], 0.015, atol=1e-10)
        assert bool(result['plausible']) is True

    def test_known_ticker_q_actual(self):
        """Known tickers (SPY, AAPL) have fallback dividend yields."""
        sindy_result = {
            'discovered_coefficients': np.array([0.05, 0.0, 0.0, -0.04, -0.02]),
        }
        result = dividend_yield_discovery(sindy_result, r_fetched=0.05,
                                           ticker='SPY')
        # Should have q_actual from fallback
        assert result['q_actual'] is not None
        assert result['q_actual'] > 0

    def test_implausible_q(self):
        """Extremely large coefficient gives implausible q."""
        sindy_result = {
            'discovered_coefficients': np.array([0.05, 0.0, 0.0, 5.0, -0.02]),
        }
        result = dividend_yield_discovery(sindy_result, r_fetched=0.05,
                                           ticker='TEST')
        # q = 0.05 - (-5.0) = 5.05 -> implausible
        assert bool(result['plausible']) is False


class TestMertonFingerprint:
    def test_bridge_returns_dataframe(self):
        """merton_real_data_bridge returns a DataFrame with expected columns."""
        merton = {
            'discovered_coefficients': np.array([0.053, -0.067, 1.905, -0.051, -0.021]),
            'true_bs_coefficients': np.array([0.05, 0.0, 0.0, -0.05, -0.02]),
            'params': {'lam': 0.1},
        }
        real = {
            'SPY': {
                'sindy_result': {
                    'discovered_coefficients': np.array([0.04, -0.5, 1.0, -0.03, -0.01]),
                },
            },
        }
        result = merton_real_data_bridge(merton, real)

        assert 'bridge_df' in result
        assert 'summary' in result
        df = result['bridge_df']
        assert len(df) == 1
        assert 'cos_sim_bs' in df.columns
        assert 'cos_sim_merton' in df.columns
        assert 'closer_to' in df.columns

    def test_cosine_similarity_range(self):
        """All cosine similarities are in [-1, 1]."""
        merton = {
            'discovered_coefficients': np.array([0.053, -0.067, 1.905, -0.051, -0.021]),
            'true_bs_coefficients': np.array([0.05, 0.0, 0.0, -0.05, -0.02]),
            'params': {'lam': 0.1},
        }
        real = {
            'SPY': {
                'sindy_result': {
                    'discovered_coefficients': np.array([-0.4, -253, -18, 0.43, 0]),
                },
            },
            'QQQ': {
                'sindy_result': {
                    'discovered_coefficients': np.array([-0.06, -115, 30, 0.23, 0]),
                },
            },
        }
        result = merton_real_data_bridge(merton, real)
        df = result['bridge_df']

        for _, row in df.iterrows():
            assert -1.0 <= row['cos_sim_bs'] <= 1.0, \
                f"cos_sim_bs out of range: {row['cos_sim_bs']}"
            assert -1.0 <= row['cos_sim_merton'] <= 1.0, \
                f"cos_sim_merton out of range: {row['cos_sim_merton']}"

    def test_empty_real_results(self):
        """Bridge handles empty real results gracefully."""
        merton = {
            'discovered_coefficients': np.array([0.053, -0.067, 1.905, -0.051, -0.021]),
            'true_bs_coefficients': np.array([0.05, 0.0, 0.0, -0.05, -0.02]),
            'params': {'lam': 0.1},
        }
        result = merton_real_data_bridge(merton, {})
        assert result['bridge_df'].empty


class TestIvRegimeHandlesSparseData:
    def test_sparse_data_no_crash(self):
        """iv_regime_analysis with only 20 options skips all slices, no crash."""
        rows = []
        S0 = 100.0
        r = 0.05
        for tau_days in [30, 60]:
            tau_val = tau_days / 365.25
            for K in np.linspace(90, 110, 10):
                rows.append({
                    'strike': K,
                    'expiration': f'2026-{5 + tau_days // 30:02d}-01',
                    'tau': tau_val,
                    'bid': 5.0,
                    'ask': 5.5,
                    'mid_price': 5.25,
                    'implied_vol': 0.25,
                    'volume': 200,
                    'openInterest': 500,
                    'S0': S0,
                    'r': r,
                })

        option_data = {
            'option_df': pd.DataFrame(rows),
            'S0': S0,
            'r': r,
            'data_source': 'test',
        }

        result = iv_regime_analysis(option_data, S0, r, 'TEST')

        assert 'maturity_regimes' in result
        assert 'moneyness_regimes' in result
        # All slices should be skipped (20 options total, <30 per slice)
        for mr in result['maturity_regimes']:
            assert mr['skipped'] is True
        for mr in result['moneyness_regimes']:
            assert mr['skipped'] is True


class TestCosineSimHelper:
    def test_identical_vectors(self):
        """Cosine similarity of a vector with itself is 1."""
        v = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-10

    def test_orthogonal_vectors(self):
        """Cosine similarity of orthogonal vectors is 0."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b)) < 1e-10

    def test_zero_vector(self):
        """Cosine similarity with zero vector is 0."""
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 0.0])
        assert _cosine_similarity(a, b) == 0.0
