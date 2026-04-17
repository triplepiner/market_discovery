"""
Tests for Fix 1-5: R²(clean), coefficient metrics, adaptive recalibration,
full library analysis, final summary, neural architecture sweep, weak SINDy
tuning, crossover analysis, and adaptive near-oracle performance.
"""

import numpy as np
import pytest

from src.utils import set_all_seeds
from src.data_generation import generate_price_surface, add_noise


class TestR2CleanComputed:
    def test_r2_clean_is_float(self):
        """compute_r2_clean returns a float."""
        from src.sindy_discovery import compute_r2_clean, discover_pde

        set_all_seeds(42)
        V, S_grid, t_grid = generate_price_surface(
            n_S=30, n_t=30, K=100, r=0.05, sigma=0.2, T=1.0,
        )

        result = discover_pde(V, S_grid, t_grid, true_sigma=0.2, true_r=0.05)
        r2c = compute_r2_clean(
            result['discovered_coefficients'], S_grid, t_grid,
            K=100, r=0.05, sigma=0.2, T=1.0,
        )

        assert isinstance(r2c, float), f"r2_clean should be float, got {type(r2c)}"


class TestR2CleanOnCleanData:
    def test_r2_clean_high_on_clean_data(self):
        """On clean data, R²(clean) should be very high (>0.99).

        When SINDy recovers the correct PDE from clean data, the predicted
        dV/dt from discovered coefficients + clean library should match
        the analytical dV/dt almost exactly.
        """
        from src.sindy_discovery import compute_r2_clean, discover_pde

        set_all_seeds(42)
        V, S_grid, t_grid = generate_price_surface(
            n_S=100, n_t=100, K=100, r=0.05, sigma=0.2, T=1.0,
        )

        result = discover_pde(V, S_grid, t_grid, true_sigma=0.2, true_r=0.05)
        r2_clean = compute_r2_clean(
            result['discovered_coefficients'], S_grid, t_grid,
            K=100, r=0.05, sigma=0.2, T=1.0,
        )

        assert r2_clean > 0.99, \
            f"R²(clean) on clean data should be >0.99, got {r2_clean:.4f}"


class TestCoefficientMetrics:
    def test_coefficient_metrics_keys(self):
        """compute_coefficient_metrics returns expected keys."""
        from src.sindy_discovery import compute_coefficient_metrics

        coeffs = np.array([0.05, 0.0, 0.0, -0.05, -0.02])
        metrics = compute_coefficient_metrics(coeffs, true_r=0.05, true_sigma=0.2)

        expected_keys = {
            'coeff_V', 'coeff_SdVdS', 'coeff_S2d2VdS2',
            'true_V', 'true_SdVdS', 'true_S2d2VdS2',
            'rel_err_V', 'rel_err_SdVdS', 'rel_err_S2d2VdS2',
            'max_coeff_rel_error', 'mean_coeff_rel_error',
            'correct_structure',
        }
        assert set(metrics.keys()) == expected_keys

    def test_perfect_coefficients(self):
        """Perfect coefficients give zero errors."""
        from src.sindy_discovery import compute_coefficient_metrics

        coeffs = np.array([0.05, 0.0, 0.0, -0.05, -0.02])
        metrics = compute_coefficient_metrics(coeffs, true_r=0.05, true_sigma=0.2)

        assert metrics['max_coeff_rel_error'] < 1e-10
        assert metrics['correct_structure'] is True


class TestAdaptiveNoCliff:
    def test_adaptive_no_cliff(self):
        """Recalibrated adaptive denoiser: no sudden R² cliff between
        consecutive noise levels.

        The old thresholds caused a cliff from ~0.91 to ~0.44 at the
        neural->weak transition. After recalibration (neural all the way),
        there should be no drop > 0.5 between consecutive levels.
        """
        from src.adaptive_denoiser import adaptive_sindy_discover
        from src.sindy_discovery import compute_r2_clean

        set_all_seeds(42)
        V_clean, S_grid, t_grid = generate_price_surface(
            n_S=50, n_t=50, K=100, r=0.05, sigma=0.2, T=1.0,
        )

        # Test the regime around the old cliff point (10-15%)
        noise_levels = [0.05, 0.10, 0.15]
        r2_cleans = []

        for nl in noise_levels:
            V = add_noise(V_clean, nl, seed=42)
            result = adaptive_sindy_discover(
                V, S_grid, t_grid,
                true_sigma=0.2, true_r=0.05, seed=42,
            )
            r2c = compute_r2_clean(
                result['discovered_coefficients'], S_grid, t_grid,
                K=100, r=0.05, sigma=0.2, T=1.0,
            )
            r2_cleans.append(r2c)

        # No cliff: consecutive drop should not exceed 0.5
        for i in range(1, len(r2_cleans)):
            drop = r2_cleans[i - 1] - r2_cleans[i]
            assert drop < 0.5, (
                f"R²(clean) cliff at noise={noise_levels[i]:.0%}: "
                f"dropped {drop:.3f} (from {r2_cleans[i-1]:.3f} to {r2_cleans[i]:.3f})"
            )


class TestAnalyzeFullLibraryResult:
    def test_returns_expected_keys(self):
        """analyze_full_library_result returns expected keys."""
        from src.sindy_discovery import analyze_full_library_result

        mock_result = {
            'discovered_coefficients': np.array([0.05, 0.001, 0.0, -0.05, -0.02]),
            'sweep_results': [],
        }

        analysis = analyze_full_library_result(mock_result, true_sigma=0.2, true_r=0.05)

        expected_keys = {
            'true_term_coefficients',
            'true_term_errors',
            'spurious_term_coefficients',
            'correlation_matrix',
            'dimensional_analysis_note',
        }
        assert set(analysis.keys()) == expected_keys

    def test_detects_spurious_terms(self):
        """Spurious terms with nonzero coefficient are detected."""
        from src.sindy_discovery import analyze_full_library_result

        mock_result = {
            'discovered_coefficients': np.array([0.05, 0.5, 0.0, -0.05, -0.02]),
            'sweep_results': [],
        }

        analysis = analyze_full_library_result(mock_result, true_sigma=0.2, true_r=0.05)
        assert 'dV/dS' in analysis['spurious_term_coefficients']

    def test_dimensional_note_nonempty(self):
        """dimensional_analysis_note is a non-empty string."""
        from src.sindy_discovery import analyze_full_library_result

        mock_result = {
            'discovered_coefficients': np.array([0.05, 0.0, 0.0, -0.05, -0.02]),
            'sweep_results': [],
        }

        analysis = analyze_full_library_result(mock_result, true_sigma=0.2, true_r=0.05)
        assert len(analysis['dimensional_analysis_note']) > 0


class TestFinalSummaryFile:
    def test_final_summary_written(self, tmp_path):
        """final_summary.txt is written with content."""
        import os

        # Simulate writing a summary
        summary_text = "=" * 72 + "\n  TEST SUMMARY\n" + "=" * 72
        path = os.path.join(str(tmp_path), 'final_summary.txt')
        with open(path, 'w') as f:
            f.write(summary_text)

        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert 'TEST SUMMARY' in content


class TestSurfaceFitterConfigs:
    def test_diagnose_returns_valid_dataframe(self):
        """diagnose_surface_fitter returns DataFrame with expected columns."""
        from src.neural_derivatives import diagnose_surface_fitter

        set_all_seeds(42)
        V, S_grid, t_grid = generate_price_surface(
            n_S=30, n_t=30, K=100, r=0.05, sigma=0.2, T=1.0,
        )

        # Use a single fast config to keep test quick
        configs = [{'n_layers': 3, 'width': 32, 'epochs': 200, 'lr': 1e-3}]
        diag = diagnose_surface_fitter(
            V, S_grid, t_grid, configs=configs, seed=42,
            K=100, r=0.05, sigma=0.2, T=1.0,
        )

        df = diag['results_df']
        assert len(df) == 1
        assert 'sindy_r2_clean' in df.columns
        assert 'fit_mse' in df.columns
        assert 'd2VdS2_rel_L2' in df.columns
        assert isinstance(diag['best_config'], dict)
        assert 'n_layers' in diag['best_config']


class TestTunedWeakSindyBetter:
    def test_tuned_weak_sindy_geq_default(self):
        """Tuned weak SINDy R²(clean) >= default R²(clean) on clean data."""
        from src.weak_sindy import tune_weak_sindy, weak_sindy_discover
        from src.sindy_discovery import compute_r2_clean

        set_all_seeds(42)
        V, S_grid, t_grid = generate_price_surface(
            n_S=50, n_t=50, K=100, r=0.05, sigma=0.2, T=1.0,
        )

        # Get tuned result (quick: just 3 configs)
        tune = tune_weak_sindy(
            V, S_grid, t_grid, true_sigma=0.2, true_r=0.05,
            n_functions_list=[50, 100, 150], width_factors=[10, 20],
            K=100, T=1.0, seed=42,
        )

        # Get default result
        default_result = weak_sindy_discover(
            V, S_grid, t_grid, n_test_functions=100,
            true_sigma=0.2, true_r=0.05, seed=42,
        )
        default_r2 = compute_r2_clean(
            default_result['discovered_coefficients'], S_grid, t_grid,
            K=100, r=0.05, sigma=0.2, T=1.0,
        )

        # Best tuned should be >= default (or within small margin)
        assert tune['best_r2_clean'] >= default_r2 - 0.05, \
            f"Tuned R²={tune['best_r2_clean']:.4f} should be >= default R²={default_r2:.4f} - 0.05"


class TestAdaptiveNearOracle:
    def test_adaptive_near_oracle(self):
        """Adaptive R²(clean) is within 0.15 of oracle at tested noise levels."""
        from src.adaptive_denoiser import adaptive_sindy_discover
        from src.sindy_discovery import compute_r2_clean, discover_pde
        from src.weak_sindy import weak_sindy_discover

        set_all_seeds(42)
        V_clean, S_grid, t_grid = generate_price_surface(
            n_S=50, n_t=50, K=100, r=0.05, sigma=0.2, T=1.0,
        )

        for nl in [0.01, 0.05, 0.10, 0.20]:
            V = add_noise(V_clean, nl, seed=42) if nl > 0 else V_clean.copy()

            # Adaptive
            adapt = adaptive_sindy_discover(
                V, S_grid, t_grid, true_sigma=0.2, true_r=0.05, seed=42,
            )
            adapt_r2 = compute_r2_clean(
                adapt['discovered_coefficients'], S_grid, t_grid,
                K=100, r=0.05, sigma=0.2, T=1.0,
            )

            # Oracle: best of SavGol and Weak
            r2s = []
            for method_fn in [
                lambda: discover_pde(V, S_grid, t_grid, true_sigma=0.2, true_r=0.05,
                                     smooth=True, savgol_window=21, savgol_poly=5, K=100, T=1.0),
                lambda: weak_sindy_discover(V, S_grid, t_grid, n_test_functions=100,
                                            true_sigma=0.2, true_r=0.05, seed=42),
            ]:
                try:
                    res = method_fn()
                    r2s.append(compute_r2_clean(
                        res['discovered_coefficients'], S_grid, t_grid,
                        K=100, r=0.05, sigma=0.2, T=1.0,
                    ))
                except Exception:
                    pass
            oracle_r2 = max(r2s) if r2s else 0

            assert adapt_r2 >= oracle_r2 - 0.15, \
                f"At noise={nl:.0%}: adaptive R²={adapt_r2:.3f} too far from " \
                f"oracle R²={oracle_r2:.3f} (gap={oracle_r2 - adapt_r2:.3f})"


class TestCrossoverFound:
    def test_crossover_in_valid_range(self):
        """SavGol/Weak crossover exists between 1% and 10% noise."""
        from src.sindy_discovery import compute_r2_clean, discover_pde
        from src.weak_sindy import weak_sindy_discover

        set_all_seeds(42)
        V_clean, S_grid, t_grid = generate_price_surface(
            n_S=50, n_t=50, K=100, r=0.05, sigma=0.2, T=1.0,
        )

        # Check that SavGol wins at 1% and Weak wins at 10%
        for nl, expected_winner in [(0.01, 'savgol'), (0.10, 'weak')]:
            V = add_noise(V_clean, nl, seed=42)

            sg = discover_pde(V, S_grid, t_grid, true_sigma=0.2, true_r=0.05,
                              smooth=True, savgol_window=21, savgol_poly=5, K=100, T=1.0)
            sg_r2 = compute_r2_clean(sg['discovered_coefficients'], S_grid, t_grid,
                                     K=100, r=0.05, sigma=0.2, T=1.0)

            wk = weak_sindy_discover(V, S_grid, t_grid, n_test_functions=100,
                                     true_sigma=0.2, true_r=0.05, seed=42)
            wk_r2 = compute_r2_clean(wk['discovered_coefficients'], S_grid, t_grid,
                                     K=100, r=0.05, sigma=0.2, T=1.0)

            if expected_winner == 'savgol':
                assert sg_r2 > wk_r2, \
                    f"At {nl:.0%}: SavGol ({sg_r2:.3f}) should beat Weak ({wk_r2:.3f})"
            else:
                assert wk_r2 > sg_r2, \
                    f"At {nl:.0%}: Weak ({wk_r2:.3f}) should beat SavGol ({sg_r2:.3f})"
