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
