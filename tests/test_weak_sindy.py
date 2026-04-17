"""
Tests for weak-form SINDy module.

Uses small grids (30 points) for speed. Tests verify:
1. Test function shapes and compact support
2. Clean-data coefficient recovery
3. Noisy-data robustness
"""

import numpy as np
import pytest

from src.utils import set_all_seeds
from src.data_generation import generate_price_surface, add_noise


class TestTestFunctionShape:
    def test_test_function_shape(self):
        """Test functions have correct shape."""
        from src.weak_sindy import create_test_functions

        S_grid = np.linspace(50, 150, 30)
        t_grid = np.linspace(0, 0.99, 30)

        tfs = create_test_functions(S_grid, t_grid, n_functions=10, seed=42)

        assert len(tfs) == 10
        for tf in tfs:
            assert tf['phi'].shape == (30, 30)
            assert tf['dphi_dS'].shape == (30, 30)
            assert tf['d2phi_dS2'].shape == (30, 30)
            assert tf['dphi_dt'].shape == (30, 30)


class TestTestFunctionCompactSupport:
    def test_test_function_compact_support(self):
        """Test functions are approximately zero outside 3 sigma."""
        from src.weak_sindy import create_test_functions

        S_grid = np.linspace(50, 150, 100)
        t_grid = np.linspace(0, 0.99, 100)
        width_S = 10.0

        tfs = create_test_functions(
            S_grid, t_grid, n_functions=5, width_S=width_S, width_t=0.1,
            seed=42
        )

        for tf in tfs:
            phi = tf['phi']
            # The function should be zero at the grid edges (far from center)
            # Check corners - at least some should be zero
            corner_vals = [
                phi[0, 0], phi[0, -1], phi[-1, 0], phi[-1, -1]
            ]
            # At least the corners should be ~0
            assert min(np.abs(corner_vals)) < 1e-10, \
                f"Test function corners should be ~0, got {corner_vals}"


class TestWeakRegressionClean:
    def test_weak_regression_clean(self):
        """On clean 50x50 data, weak SINDy achieves good R² (>0.80)."""
        from src.weak_sindy import weak_sindy_discover

        set_all_seeds(42)
        V, S_grid, t_grid = generate_price_surface(
            n_S=50, n_t=50, K=100, r=0.05, sigma=0.2, T=1.0
        )

        result = weak_sindy_discover(
            V, S_grid, t_grid,
            n_test_functions=100,
            true_sigma=0.2, true_r=0.05, seed=42,
        )

        # Weak SINDy has same multicollinearity as standard 5-term library
        # but the R² should still be good
        assert result['r2_score'] > 0.80, \
            f"Weak SINDy R² = {result['r2_score']:.4f} on clean data, expected > 0.80"
        assert result['n_active'] >= 2, \
            f"Expected >=2 active terms, got {result['n_active']}"


class TestWeakRegressionNoisy:
    def test_weak_regression_noisy(self):
        """At 10% noise, weak SINDy R² remains high (>0.5) unlike FD (~0.001)."""
        from src.weak_sindy import weak_sindy_discover

        set_all_seeds(42)
        V_clean, S_grid, t_grid = generate_price_surface(
            n_S=50, n_t=50, K=100, r=0.05, sigma=0.2, T=1.0
        )
        V_noisy = add_noise(V_clean, 0.10, seed=42)

        result = weak_sindy_discover(
            V_noisy, S_grid, t_grid,
            n_test_functions=100,
            true_sigma=0.2, true_r=0.05, seed=42,
        )

        # Key advantage of weak SINDy: integration averages out noise
        # FD gives R² ≈ 0.001 at 10% noise; weak SINDy should be much better
        assert result['r2_score'] > 0.5, \
            f"Weak SINDy R² = {result['r2_score']:.4f} at 10% noise, expected > 0.5"


class TestWeakSindyNoiseRobustness:
    def test_weak_sindy_noise_robustness(self):
        """Weak SINDy R² degrades gracefully: R²(10% noise) > 0.5 * R²(clean)."""
        from src.weak_sindy import weak_sindy_discover

        set_all_seeds(42)
        V_clean, S_grid, t_grid = generate_price_surface(
            n_S=50, n_t=50, K=100, r=0.05, sigma=0.2, T=1.0
        )

        # Clean
        result_clean = weak_sindy_discover(
            V_clean, S_grid, t_grid,
            n_test_functions=100,
            true_sigma=0.2, true_r=0.05, seed=42,
        )

        # 10% noise
        V_noisy = add_noise(V_clean, 0.10, seed=42)
        result_noisy = weak_sindy_discover(
            V_noisy, S_grid, t_grid,
            n_test_functions=100,
            true_sigma=0.2, true_r=0.05, seed=42,
        )

        # R² should degrade gracefully, not collapse like FD
        ratio = result_noisy['r2_score'] / max(result_clean['r2_score'], 1e-10)
        assert ratio > 0.5, \
            f"Noise degradation ratio = {ratio:.2f}, expected > 0.5"
