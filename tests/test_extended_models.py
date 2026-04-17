"""Tests for Merton and Heston experiments."""
import numpy as np
import pytest
from src.data_generation import merton_call_price, generate_merton_surface, bs_call_price
from src.extended_models import run_merton_experiment, run_heston_variance_slicing


class TestMertonPricing:
    def test_merton_reduces_to_bs(self):
        """With lam=0, Merton should equal BS."""
        S = np.array([80, 100, 120])
        price_merton = merton_call_price(S, 100, 0.05, 0.2, 0.5, lam=0.0)
        price_bs = bs_call_price(S, 100, 0.05, 0.2, 0.5)
        np.testing.assert_allclose(price_merton, price_bs, rtol=1e-6)

    def test_merton_nonnegative(self):
        S = np.linspace(50, 150, 50)
        prices = merton_call_price(S, 100, 0.05, 0.2, 0.5)
        assert np.all(prices >= 0)

    def test_merton_surface_shape(self):
        V, S, t = generate_merton_surface(n_S=30, n_t=30)
        assert V.shape == (30, 30)
        assert len(S) == 30
        assert len(t) == 30


class TestMertonExperiment:
    def test_merton_r2_lower_than_bs(self):
        result = run_merton_experiment()
        # R^2 should be < 1.0 because library is misspecified
        assert result['r2'] < 1.0
        # But should still be reasonable
        assert result['r2'] > 0.5

    def test_has_residual_grid(self):
        result = run_merton_experiment()
        assert result['residual_grid'] is not None
        assert result['residual_grid'].ndim == 2


class TestHestonSlicing:
    def test_heston_linearity(self):
        result = run_heston_variance_slicing()
        # Discovered diffusion coeff should track v linearly
        assert result['linearity_r2'] > 0.95
        # Slope should be near -0.5
        assert abs(result['linear_fit_slope'] - (-0.5)) < 0.1

    def test_heston_n_slices(self):
        result = run_heston_variance_slicing(v_list=[0.01, 0.04, 0.16])
        assert len(result['per_slice_results']) == 3
