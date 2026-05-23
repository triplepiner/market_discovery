"""Tests for advanced baseline PDE discovery methods (Elastic Net, PySR)."""
import numpy as np
import pytest

from src.baselines import elastic_net_regression, pysr_symbolic_regression
from src.data_generation import generate_price_surface


@pytest.fixture(scope='module')
def small_bs_surface():
    """Small (20x20) BS call surface for fast tests."""
    V, S_grid, t_grid = generate_price_surface(
        S_min=50, S_max=150, n_S=20, n_t=20,
        K=100, r=0.05, sigma=0.2, T=1.0, option_type='call',
    )
    return V, S_grid, t_grid


class TestElasticNet:
    def test_elastic_net_runs_without_error(self, small_bs_surface):
        V, S_grid, t_grid = small_bs_surface
        result = elastic_net_regression(V, S_grid, t_grid, smooth=False, seed=42)
        assert 'coefficients' in result
        assert 'r2_score' in result
        assert 'best_alpha' in result
        assert 'best_l1_ratio' in result
        assert 'active_terms' in result
        assert 'term_names' in result
        assert 'n_active' in result
        assert len(result['coefficients']) == 5
        assert result['method'] == 'elastic_net'

    def test_elastic_net_returns_sparse_coefficients(self, small_bs_surface):
        V, S_grid, t_grid = small_bs_surface
        result = elastic_net_regression(V, S_grid, t_grid, smooth=False, seed=42)
        # Clean BS data: true PDE has only 3 nonzero canonical terms
        # (V, S*dV/dS, S^2 d2V/dS2). With sparsity, n_active should be modest.
        assert result['n_active'] < 5, (
            f"Expected sparse solution but got n_active={result['n_active']}"
        )
        # Best alpha and l1_ratio should be finite
        assert np.isfinite(result['best_alpha'])
        assert np.isfinite(result['best_l1_ratio'])


class TestPySR:
    def test_pysr_handles_missing_gracefully(self, small_bs_surface):
        V, S_grid, t_grid = small_bs_surface
        result = pysr_symbolic_regression(
            V, S_grid, t_grid,
            smooth=False,
            n_iterations=3, populations=4, max_size=10,
            timeout_minutes=2, seed=42,
        )
        assert isinstance(result, dict)
        assert 'status' in result
        assert result['method'] == 'pysr'
        if result['status'] == 'skipped':
            assert 'reason' in result
            assert isinstance(result['reason'], str)
        elif result['status'] == 'completed':
            assert 'symbolic_expression' in result
            assert 'r2_score' in result
            assert 'n_terms' in result
            assert isinstance(result['symbolic_expression'], str)
            assert np.isfinite(result['r2_score'])
            assert result['n_terms'] >= 1
        else:
            pytest.fail(f"Unexpected status: {result['status']}")
