"""
Tests for spectral (FFT-based) derivative estimation.
"""

import numpy as np
import pytest

from src.utils import set_all_seeds
from src.data_generation import generate_price_surface


class TestSpectralFirstDerivativeSine:
    def test_spectral_first_derivative_sine(self):
        """d/dx sin(x) ≈ cos(x) within 1e-3 on a 100-point grid.

        Tests the underlying spectral algorithm on an intrinsically periodic
        signal (no mirror padding needed) so spectral accuracy is achievable.
        """
        from src.spectral_derivatives import compute_spectral_derivatives_periodic

        n_S, n_t = 100, 8
        # endpoint=False so the grid is exactly one FFT period
        S_grid = np.linspace(0.0, 2.0 * np.pi, n_S, endpoint=False)
        t_grid = np.linspace(0.0, 1.0, n_t, endpoint=False)
        S_mesh, _ = np.meshgrid(S_grid, t_grid, indexing='ij')
        V = np.sin(S_mesh)
        dS = float(S_grid[1] - S_grid[0])
        dt = float(t_grid[1] - t_grid[0])

        derivs = compute_spectral_derivatives_periodic(V, dS, dt)
        truth = np.cos(S_mesh)
        max_err = float(np.max(np.abs(derivs['dV_dS'] - truth)))
        assert max_err < 1e-3, (
            f"Spectral d/dx sin(x): max err = {max_err:.6f}, expected < 1e-3"
        )


class TestSpectralSecondDerivativeSine:
    def test_spectral_second_derivative_sine(self):
        """d²/dx² sin(x) ≈ -sin(x) within 1e-3 on a 100-point grid."""
        from src.spectral_derivatives import compute_spectral_derivatives_periodic

        n_S, n_t = 100, 8
        S_grid = np.linspace(0.0, 2.0 * np.pi, n_S, endpoint=False)
        t_grid = np.linspace(0.0, 1.0, n_t, endpoint=False)
        S_mesh, _ = np.meshgrid(S_grid, t_grid, indexing='ij')
        V = np.sin(S_mesh)
        dS = float(S_grid[1] - S_grid[0])
        dt = float(t_grid[1] - t_grid[0])

        derivs = compute_spectral_derivatives_periodic(V, dS, dt)
        truth = -np.sin(S_mesh)
        max_err = float(np.max(np.abs(derivs['d2V_dS2'] - truth)))
        assert max_err < 1e-3, (
            f"Spectral d²/dx² sin(x): max err = {max_err:.6f}, expected < 1e-3"
        )


class TestSpectralDerivativeShape:
    def test_spectral_derivative_shape(self):
        """compute_spectral_derivatives outputs match input shape."""
        from src.spectral_derivatives import compute_spectral_derivatives

        n_S, n_t = 30, 25
        S_grid = np.linspace(50.0, 150.0, n_S)
        t_grid = np.linspace(0.0, 0.99, n_t)
        V = np.random.RandomState(0).randn(n_S, n_t)

        derivs = compute_spectral_derivatives(V, S_grid, t_grid)
        assert derivs['dV_dS'].shape == (n_S, n_t)
        assert derivs['dV_dt'].shape == (n_S, n_t)
        assert derivs['d2V_dS2'].shape == (n_S, n_t)


class TestSpectralSindyCleanReasonable:
    def test_spectral_sindy_clean_reasonable(self):
        """sindy_with_spectral_derivatives on clean BS data has R² > 0.7."""
        from src.spectral_derivatives import sindy_with_spectral_derivatives

        set_all_seeds(42)
        V, S_grid, t_grid = generate_price_surface(
            n_S=40, n_t=40, K=100, r=0.05, sigma=0.2, T=1.0
        )
        result = sindy_with_spectral_derivatives(
            V, S_grid, t_grid, seed=42,
            K=100, r=0.05, sigma=0.2, T=1.0,
            true_r=0.05, true_sigma=0.2,
        )
        assert result['r2_clean'] > 0.7, (
            f"R²(clean)={result['r2_clean']:.4f} not above 0.7"
        )
