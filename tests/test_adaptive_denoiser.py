"""
Tests for the adaptive denoiser module.

Tests noise estimation accuracy, strategy selection logic, and end-to-end
adaptive SINDy discovery.
"""

import numpy as np
import pytest

from src.utils import set_all_seeds
from src.data_generation import generate_price_surface, add_noise


class TestNoiseEstimationClean:
    def test_noise_estimation_clean(self):
        """estimate_noise_level on clean data returns < 0.02."""
        from src.adaptive_denoiser import estimate_noise_level

        set_all_seeds(42)
        V, S_grid, t_grid = generate_price_surface(
            n_S=30, n_t=30, K=100, r=0.05, sigma=0.2, T=1.0
        )

        est = estimate_noise_level(V, S_grid, t_grid)
        assert est < 0.02, f"Clean noise estimate = {est:.4f}, expected < 0.02"


class TestNoiseEstimation5pct:
    def test_noise_estimation_5pct(self):
        """On 5% noisy data, estimate returns between 0.02 and 0.10."""
        from src.adaptive_denoiser import estimate_noise_level

        set_all_seeds(42)
        V_clean, S_grid, t_grid = generate_price_surface(
            n_S=50, n_t=50, K=100, r=0.05, sigma=0.2, T=1.0
        )
        V_noisy = add_noise(V_clean, 0.05, seed=42)

        est = estimate_noise_level(V_noisy, S_grid, t_grid)
        assert 0.02 < est < 0.10, \
            f"5% noise estimate = {est:.4f}, expected in (0.02, 0.10)"


class TestNoiseEstimation20pct:
    def test_noise_estimation_20pct(self):
        """On 20% noisy data, estimate returns between 0.10 and 0.35."""
        from src.adaptive_denoiser import estimate_noise_level

        set_all_seeds(42)
        V_clean, S_grid, t_grid = generate_price_surface(
            n_S=50, n_t=50, K=100, r=0.05, sigma=0.2, T=1.0
        )
        V_noisy = add_noise(V_clean, 0.20, seed=42)

        est = estimate_noise_level(V_noisy, S_grid, t_grid)
        assert 0.10 < est < 0.35, \
            f"20% noise estimate = {est:.4f}, expected in (0.10, 0.35)"


class TestStrategySelection:
    def test_strategy_selection(self):
        """Correct strategies for different noise levels (recalibrated with GP)."""
        from src.adaptive_denoiser import select_derivative_strategy

        # FD: < 0.5% noise
        strat_clean, _ = select_derivative_strategy(0.002)
        assert strat_clean == 'fd'

        # SavGol: narrow band 0.5% - 1% (safety margin before GP)
        strat_low, _ = select_derivative_strategy(0.007)
        assert strat_low == 'savgol'

        # GP: 1% - 12% (dominates here per R²(clean) crossover analysis)
        strat_med, _ = select_derivative_strategy(0.025)
        assert strat_med == 'gp'

        strat_5pct, _ = select_derivative_strategy(0.05)
        assert strat_5pct == 'gp'

        strat_10pct, _ = select_derivative_strategy(0.10)
        assert strat_10pct == 'gp'

        # Weak SINDy takes over at >= 12% noise (GP collapses by 15%)
        strat_15, _ = select_derivative_strategy(0.15)
        assert strat_15 == 'weak'

        strat_30, _ = select_derivative_strategy(0.30)
        assert strat_30 == 'weak'

        # Unreliable: >= 50% noise
        strat_extreme, _ = select_derivative_strategy(0.60)
        assert strat_extreme == 'unreliable'


class TestGpSelectedForModerateNoise:
    def test_gp_selected_for_moderate_noise(self):
        """At 10% noise, strategy should be 'gp' (GP beats SavGol and Weak)."""
        from src.adaptive_denoiser import select_derivative_strategy

        strat, params = select_derivative_strategy(0.10)
        assert strat == 'gp', f"Expected 'gp' at 10% noise, got '{strat}'"
        assert 'n_subsample' in params, \
            f"GP params should include n_subsample, got {params}"


class TestRecalibrationReturnsDataframe:
    def test_recalibration_returns_dataframe(self):
        """recalibrate_adaptive_with_gp() returns DataFrame with expected columns.

        Runs a tiny 2-noise-level sweep on a small grid so the test stays
        fast; only checks structure, not correctness of the threshold values.
        """
        import pandas as pd
        from src.adaptive_denoiser import recalibrate_adaptive_with_gp

        df, recommended = recalibrate_adaptive_with_gp(
            n_S=20, n_t=20, seed=42,
            noise_levels=[0.0, 0.05],
            n_subsample=80,
        )

        assert isinstance(df, pd.DataFrame), "Expected a pandas DataFrame"
        expected_cols = {
            'noise_pct', 'r2_fd', 'r2_savgol', 'r2_gp', 'r2_weak',
            'runtime_fd_s', 'runtime_savgol_s',
            'runtime_gp_s', 'runtime_weak_s',
        }
        assert expected_cols.issubset(set(df.columns)), \
            f"Missing columns: {expected_cols - set(df.columns)}"
        assert len(df) == 2, f"Expected 2 rows, got {len(df)}"

        # Check recommended dict structure
        assert isinstance(recommended, dict)
        assert 'gp_crossover' in recommended
        assert 'weak_crossover' in recommended
        assert 'thresholds' in recommended
        assert 'fd' in recommended['thresholds']
        assert 'gp' in recommended['thresholds']


class TestAdaptiveRunsWithoutError:
    def test_adaptive_runs_without_error(self):
        """adaptive_sindy_discover completes on 5% noisy 20x20 data."""
        from src.adaptive_denoiser import adaptive_sindy_discover

        set_all_seeds(42)
        V_clean, S_grid, t_grid = generate_price_surface(
            n_S=20, n_t=20, K=100, r=0.05, sigma=0.2, T=1.0
        )
        V_noisy = add_noise(V_clean, 0.05, seed=42)

        result = adaptive_sindy_discover(
            V_noisy, S_grid, t_grid,
            true_sigma=0.2, true_r=0.05, seed=42,
        )

        # Check it returns the standard format
        assert 'discovered_coefficients' in result
        assert 'r2_score' in result
        assert 'estimated_noise' in result
        assert 'selected_strategy' in result
        assert len(result['discovered_coefficients']) == 5
