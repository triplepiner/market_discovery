"""Tests for baseline PDE discovery methods."""
import numpy as np
import pytest
from src.baselines import dense_regression, lasso_regression, ridge_threshold, run_all_baselines
from src.sindy_discovery import compute_derivatives, build_candidate_library
from src.data_generation import generate_price_surface


@pytest.fixture(scope='module')
def clean_library():
    """Build a clean BS library for testing."""
    V, S_grid, t_grid = generate_price_surface(
        S_min=50, S_max=150, n_S=50, n_t=50,
        K=100, r=0.05, sigma=0.2, T=1.0, option_type='call',
    )
    derivs = compute_derivatives(V, S_grid, t_grid, trim=5)
    library = build_candidate_library(derivs['V'], derivs['dVdS'], derivs['d2VdS2'], derivs['S_mesh'])
    target = derivs['dVdt'].ravel()
    return library, target


class TestDenseRegression:
    def test_returns_5_coefficients(self, clean_library):
        lib, target = clean_library
        result = dense_regression(lib, target)
        assert len(result['coefficients']) == 5

    def test_all_nonzero(self, clean_library):
        lib, target = clean_library
        result = dense_regression(lib, target)
        # Dense regression should have all or most nonzero
        assert result['n_active'] >= 3

    def test_high_r2(self, clean_library):
        lib, target = clean_library
        result = dense_regression(lib, target)
        assert result['r2'] > 0.99


class TestLassoRegression:
    def test_sparse_solution(self, clean_library):
        lib, target = clean_library
        result = lasso_regression(lib, target)
        assert result['n_active'] <= 5
        assert result['r2'] > 0.9

    def test_has_lasso_path(self, clean_library):
        lib, target = clean_library
        result = lasso_regression(lib, target)
        assert 'lasso_path' in result
        assert 'alphas' in result['lasso_path']
        assert 'coefs' in result['lasso_path']


class TestRidgeThreshold:
    def test_runs_without_error(self, clean_library):
        lib, target = clean_library
        result = ridge_threshold(lib, target)
        assert 'coefficients' in result
        assert result['r2'] > 0.9


class TestRunAllBaselines:
    def test_runs_end_to_end(self):
        V, S_grid, t_grid = generate_price_surface(
            S_min=50, S_max=150, n_S=40, n_t=40,
            K=100, r=0.05, sigma=0.2, T=1.0,
        )
        results = run_all_baselines(
            V, S_grid, t_grid, true_sigma=0.2, true_r=0.05,
            K=100, T=1.0,
        )
        assert 'dense' in results
        assert 'lasso' in results
        assert 'ridge_threshold' in results
