"""Tests for real data module."""
import numpy as np
import os
import tempfile
import pytest
from unittest import mock

import pandas as pd

from src.real_data import _generate_mock_data, construct_smooth_surface, fetch_option_data


class TestMockData:
    def test_mock_data_structure(self):
        data = _generate_mock_data('SPY')
        assert data['data_source'] == 'mock'
        assert data['S0'] > 0
        assert len(data['strikes']) > 0
        assert len(data['implied_vols']) > 0

    def test_mock_surface_construction(self):
        data = _generate_mock_data('SPY')
        surface = construct_smooth_surface(data)
        assert surface['V_surface'].shape[0] > 5
        assert surface['V_surface'].shape[1] >= 3
        assert np.all(surface['V_surface'] >= 0)


class TestIVFiltering:
    def test_iv_filtering(self):
        """Verify that NaN, negative, and >3.0 implied vols are removed."""
        rows = []
        S0, r = 100.0, 0.05
        base_tau = 0.25
        # Good rows
        for iv in [0.2, 0.3, 0.5, 1.0, 2.5]:
            rows.append({
                'strike': 100.0,
                'expiration': '2026-06-01',
                'tau': base_tau,
                'bid': 5.0,
                'ask': 5.5,
                'mid_price': 5.25,
                'implied_vol': iv,
                'volume': 200,
                'openInterest': 500,
                'S0': S0,
                'r': r,
            })
        # Bad rows: NaN, negative, >3.0
        for bad_iv in [np.nan, -0.1, 3.5]:
            rows.append({
                'strike': 100.0,
                'expiration': '2026-06-01',
                'tau': base_tau,
                'bid': 5.0,
                'ask': 5.5,
                'mid_price': 5.25,
                'implied_vol': bad_iv,
                'volume': 200,
                'openInterest': 500,
                'S0': S0,
                'r': r,
            })

        df = pd.DataFrame(rows)
        # The IV filtering in construct_smooth_surface removes iv<=0 and iv>2.0
        # But _dataframe_to_result keeps them in the df; filtering happens at
        # the surface construction level. Verify the surface builder filters
        # correctly -- it drops rows with invalid iv.
        from src.real_data import _dataframe_to_result
        result = _dataframe_to_result(df, 'TEST', data_source='test')

        # The raw result should have all 8 rows
        assert result['n_options'] == 8

        # Implied vols array should contain the bad values too (they are raw)
        ivs = result['implied_vols']
        assert np.any(np.isnan(ivs))  # NaN is in there

        # Now verify construct_smooth_surface filters them out.
        # We need enough expirations (>=3) for surface construction,
        # so build a richer dataset.
        rich_rows = []
        for tau_days in [30, 60, 90, 120]:
            tau_val = tau_days / 365.25
            for K in np.linspace(80, 120, 15):
                # Good IV
                rich_rows.append({
                    'strike': K,
                    'expiration': f'2026-{4 + tau_days // 30:02d}-01',
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
            # Bad IV rows for each expiration
            for bad_iv in [np.nan, -0.5, 3.1]:
                rich_rows.append({
                    'strike': 100.0,
                    'expiration': f'2026-{4 + tau_days // 30:02d}-01',
                    'tau': tau_val,
                    'bid': 5.0,
                    'ask': 5.5,
                    'mid_price': 5.25,
                    'implied_vol': bad_iv,
                    'volume': 200,
                    'openInterest': 500,
                    'S0': S0,
                    'r': r,
                })

        rich_df = pd.DataFrame(rich_rows)
        rich_result = _dataframe_to_result(rich_df, 'TEST', data_source='test')

        # Surface construction should succeed (bad IVs removed internally)
        surface = construct_smooth_surface(rich_result)
        assert surface['V_surface'].shape[0] > 0
        assert surface['V_surface'].shape[1] >= 3
        # All surface IV values should be in valid range
        assert np.all(surface['iv_surface'] >= 0.01)
        assert np.all(surface['iv_surface'] <= 2.0)


class TestCacheLoading:
    def test_cache_loading(self):
        """Write a fake cache CSV to tempdir, verify it loads instead of fetching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build a minimal valid DataFrame
            rows = []
            S0, r = 100.0, 0.05
            for tau_days in [30, 60, 90]:
                tau_val = tau_days / 365.25
                for K in np.linspace(80, 120, 10):
                    rows.append({
                        'strike': K,
                        'expiration': f'2026-{4 + tau_days // 30:02d}-01',
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

            df = pd.DataFrame(rows)

            # Write cache file with today's date stamp
            from datetime import datetime, timezone
            today_str = datetime.now(timezone.utc).strftime('%Y%m%d')
            cache_path = os.path.join(
                tmpdir, f'real_chain_TEST_{today_str}.csv'
            )
            df.to_csv(cache_path, index=False)

            # Fetch should load from cache, not hit yfinance
            result = fetch_option_data('TEST', cache_dir=tmpdir)
            assert result['data_source'] == 'cached'
            assert result['n_options'] == len(df)
            assert result['S0'] == S0
            assert result['r'] == r


class TestFallbackOnImportError:
    def test_fallback_on_import_error(self):
        """Ensure mock data is returned when yfinance import fails."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'yfinance':
                raise ImportError("Mocked: yfinance not installed")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use empty cache dir so cache is not found
            with mock.patch('builtins.__import__', side_effect=mock_import):
                result = fetch_option_data('SPY', cache_dir=tmpdir)

            assert result['data_source'] == 'mock'
            assert result['S0'] > 0
            assert result['n_options'] > 0
            assert len(result['strikes']) > 0
            assert len(result['implied_vols']) > 0
            # Verify new keys are present
            assert 'bid_ask_spread_pct' in result
            assert 'n_expirations_raw' in result
            assert 'n_expirations_filtered' in result
            assert 'n_contracts_raw' in result
            assert 'n_contracts_filtered' in result
            assert 'strike_range' in result
            assert 'tau_range' in result
            assert 'iv_range' in result


# ---------------------------------------------------------------------------
# Agent B3 ablation-chain + alternative-smoothing tests
# ---------------------------------------------------------------------------

class TestAblationConfigsRun:
    """Smoke tests for the 6-config ablation chain on a small mock surface."""

    def _make_mock_snapshot(self):
        """Build a small option DataFrame from the const-sigma synthetic
        surface so ablation runners don't need yfinance or the real cache."""
        from src.sindy_kan import generate_synthetic_dupire_constsig

        d = generate_synthetic_dupire_constsig(
            sigma=0.20, r=0.05, q=0.0, S0=100.0,
            n_k=20, n_tau=20, k_range=(-0.20, 0.20),
            tau_range=(0.05, 1.0))
        # Convert to a long-form options DataFrame mimicking real cache shape.
        from src.real_data_v2 import compute_forward_prices
        k_grid = d['k']; tau_grid = d['tau']; C = d['C']
        sigma_imp = d['sigma_imp']
        F_grid = compute_forward_prices(100.0, 0.05, 0.0, tau_grid)
        rows = []
        for i, k in enumerate(k_grid):
            for j, tau in enumerate(tau_grid):
                K = float(F_grid[j] * np.exp(float(k)))
                rows.append({
                    'strike': K,
                    'expiration': f'2026-{j+1:02d}-15',
                    'tau': float(tau),
                    'bid': float(C[i, j]) * 0.99,
                    'ask': float(C[i, j]) * 1.01,
                    'mid_price': float(C[i, j]),
                    'implied_vol': float(sigma_imp[i, j]),
                    'volume': 200,
                    'openInterest': 500,
                    'S0': 100.0,
                    'r': 0.05,
                })
        df = pd.DataFrame(rows)
        return {'ticker': 'MOCK', 'snapshot': 'mock',
                'S0': 100.0, 'r': 0.05, 'option_df': df}

    def test_config_A_runs(self):
        from src.ablation_chain import run_config_A
        snap = self._make_mock_snapshot()
        res = run_config_A(snap, q=0.0, n_K=20, n_tau=15)
        assert 'r2' in res
        assert np.isfinite(res['r2']), f"Config A R^2 not finite: {res['r2']}"

    def test_config_E_runs(self):
        from src.ablation_chain import run_config_E
        snap = self._make_mock_snapshot()
        res = run_config_E(snap, q=0.0, n_k=20, n_tau=15,
                           k_range=(-0.15, 0.15))
        assert np.isfinite(res['r2']), f"Config E R^2 not finite: {res['r2']}"
        # On a small synthetic surface with sparse expirations the SVI fit
        # introduces some noise; we only require a strictly positive R^2 to
        # confirm the pipeline is recovering the dominant Dupire structure.
        assert res['r2'] > 0.0, f"Config E R^2 non-positive: {res['r2']}"

    def test_config_F_runs(self):
        from src.ablation_chain import run_config_F
        snap = self._make_mock_snapshot()
        # Fewer epochs to keep the test fast; we only assert finiteness.
        res = run_config_F(snap, q=0.0, n_k=20, n_tau=15,
                           k_range=(-0.15, 0.15), n_epochs=300, seed=42)
        assert 'r2' in res and 'train_r2' in res
        assert np.isfinite(res['r2']), f"Config F R^2 not finite: {res['r2']}"
        assert np.isfinite(res['train_r2'])


class TestCubicSplineSmoothingRuns:
    """Smoke test: cubic-spline IV smoothing returns a finite surface."""

    def test_cubic_spline_smoothing_runs(self):
        from src.ablation_chain import cubic_spline_smooth_surface
        from src.sindy_kan import generate_synthetic_dupire_smile
        from src.real_data_v2 import compute_forward_prices

        d = generate_synthetic_dupire_smile(
            sigma_atm=0.20, smile_curvature=0.10, r=0.05, q=0.0, S0=100.0,
            n_k=20, n_tau=20, k_range=(-0.20, 0.20),
            tau_range=(0.05, 1.0))
        F_grid = compute_forward_prices(100.0, 0.05, 0.0, d['tau'])
        rows = []
        for i, k in enumerate(d['k']):
            for j, tau in enumerate(d['tau']):
                K = float(F_grid[j] * np.exp(float(k)))
                rows.append({
                    'strike': K,
                    'expiration': f'2026-{j+1:02d}-15',
                    'tau': float(tau),
                    'mid_price': float(d['C'][i, j]),
                    'implied_vol': float(d['sigma_imp'][i, j]),
                    'S0': 100.0,
                    'r': 0.05,
                })
        df = pd.DataFrame(rows)
        surf = cubic_spline_smooth_surface(
            df, S0=100.0, r=0.05, q=0.0,
            n_k=20, n_tau=10, k_range=(-0.15, 0.15))
        assert surf['C_surface'].shape == (20, 10)
        assert np.all(np.isfinite(surf['C_surface']))
        assert np.all(np.isfinite(surf['sigma_surface']))
        assert np.all(surf['sigma_surface'] > 0)
        assert np.all(surf['C_surface'] > 0)
        # n_fits should equal the number of distinct expirations in the input.
        assert surf['n_fits'] >= 3
