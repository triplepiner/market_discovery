"""
Tests for neural derivative estimation module.

Uses small grids (20-30 points) and minimal epochs to stay fast (<30s total).
"""

import numpy as np
import pytest

from src.utils import set_all_seeds
from src.data_generation import generate_price_surface, add_noise, bs_gamma


class TestSurfaceFitterOutputShape:
    def test_surface_fitter_output_shape(self):
        """SurfaceFitter output matches input grid shape."""
        from src.neural_derivatives import SurfaceFitter
        import torch

        set_all_seeds(42)
        n_S, n_t = 20, 20
        model = SurfaceFitter(50.0, 150.0, 0.0, 0.99, V_mean=10.0, V_std=5.0)

        S = torch.linspace(50, 150, n_S, dtype=torch.float64)
        t = torch.linspace(0, 0.99, n_t, dtype=torch.float64)
        S_mesh, t_mesh = torch.meshgrid(S, t, indexing='ij')

        with torch.no_grad():
            V = model(S_mesh.reshape(-1), t_mesh.reshape(-1))

        assert V.shape == (n_S * n_t,)


class TestSurfaceFitterLearnsClean:
    def test_surface_fitter_learns_clean(self):
        """On a clean 20x20 BS surface, after 500 epochs, MSE < 1% of var(V)."""
        from src.neural_derivatives import fit_surface

        set_all_seeds(42)
        V, S_grid, t_grid = generate_price_surface(
            n_S=20, n_t=20, K=100, r=0.05, sigma=0.2, T=1.0
        )

        model, info = fit_surface(V, S_grid, t_grid, epochs=500, seed=42)
        assert info['final_mse'] < 0.01 * np.var(V)


class TestNeuralDerivativesShape:
    def test_neural_derivatives_shape(self):
        """All derivative arrays match grid shape."""
        from src.neural_derivatives import fit_surface, compute_neural_derivatives

        set_all_seeds(42)
        V, S_grid, t_grid = generate_price_surface(n_S=20, n_t=20)
        model, _ = fit_surface(V, S_grid, t_grid, epochs=300, seed=42)
        derivs = compute_neural_derivatives(model, S_grid, t_grid)

        assert derivs['V_smooth'].shape == (20, 20)
        assert derivs['dVdt'].shape == (20, 20)
        assert derivs['dVdS'].shape == (20, 20)
        assert derivs['d2VdS2'].shape == (20, 20)


class TestNeuralVsAnalyticalClean:
    def test_neural_vs_analytical_clean(self):
        """On clean data, neural d2V/dS2 has rel L2 < 35% vs analytical Gamma."""
        from src.neural_derivatives import fit_surface, compute_neural_derivatives

        set_all_seeds(42)
        V, S_grid, t_grid = generate_price_surface(
            n_S=30, n_t=30, K=100, r=0.05, sigma=0.2, T=1.0
        )

        model, _ = fit_surface(V, S_grid, t_grid, epochs=1500, seed=42)
        derivs = compute_neural_derivatives(model, S_grid, t_grid)

        # Analytical Gamma on interior
        trim = 5
        s = slice(trim, -trim)
        S_tr = S_grid[s]
        t_tr = t_grid[s]
        S_mesh, t_mesh = np.meshgrid(S_tr, t_tr, indexing='ij')
        tau_mesh = 1.0 - t_mesh
        gamma_ana = bs_gamma(S_mesh, 100, 0.05, 0.2, tau_mesh)
        gamma_neural = derivs['d2VdS2'][s, s]

        rel_l2 = np.linalg.norm(gamma_neural - gamma_ana) / (np.linalg.norm(gamma_ana) + 1e-15)
        assert rel_l2 < 0.35, f"Neural Gamma rel L2 = {rel_l2:.4f}, expected < 0.35"


class TestNeuralSindyClean:
    def test_neural_sindy_clean(self):
        """Neural SINDy on clean 30x30 data recovers 2+ active terms with good R2."""
        from src.neural_derivatives import sindy_with_neural_derivatives

        set_all_seeds(42)
        V, S_grid, t_grid = generate_price_surface(
            n_S=30, n_t=30, K=100, r=0.05, sigma=0.2, T=1.0
        )

        result = sindy_with_neural_derivatives(
            V, S_grid, t_grid,
            true_sigma=0.2, true_r=0.05,
            fit_epochs=1500, seed=42,
        )

        assert result['n_active'] >= 2
        assert result['r2_score'] > 0.75


class TestNeuralSindyNoisy:
    def test_neural_sindy_noisy(self):
        """At 5% noise on 30x30, neural SINDy R2 > 0.3 (better than FD ~0.001)."""
        from src.neural_derivatives import sindy_with_neural_derivatives

        set_all_seeds(42)
        V_clean, S_grid, t_grid = generate_price_surface(
            n_S=30, n_t=30, K=100, r=0.05, sigma=0.2, T=1.0
        )
        V_noisy = add_noise(V_clean, 0.05, seed=42)

        result = sindy_with_neural_derivatives(
            V_noisy, S_grid, t_grid,
            true_sigma=0.2, true_r=0.05,
            fit_epochs=800, seed=42,
        )

        # Neural SINDy should be much better than FD at 5% noise
        assert result['r2_score'] > 0.3, \
            f"Neural SINDy R² = {result['r2_score']:.4f} at 5% noise, expected > 0.3"
