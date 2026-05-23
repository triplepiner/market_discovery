"""
Tests for GP-based derivative estimation.

Uses small grids (20x20) and small subsample sizes to stay fast.
"""

import numpy as np
import pytest

from src.utils import set_all_seeds
from src.data_generation import generate_price_surface


class TestGPFitCleanDataLowMSE:
    def test_gp_fit_clean_data_low_mse(self):
        """GP fit on clean BS data has MSE < 0.01."""
        from src.gp_derivatives import fit_gp_surface, compute_gp_derivatives

        set_all_seeds(42)
        V, S_grid, t_grid = generate_price_surface(
            n_S=20, n_t=20, K=100, r=0.05, sigma=0.2, T=1.0
        )
        gp, _ = fit_gp_surface(V, S_grid, t_grid, n_subsample=200, seed=42)
        derivs = compute_gp_derivatives(gp, S_grid, t_grid)
        mse = float(np.mean((derivs['V_smooth'] - V) ** 2))
        assert mse < 0.01, f"GP fit MSE={mse:.6f} not below 0.01"


class TestGPDerivativeShape:
    def test_gp_derivative_shape(self):
        """compute_gp_derivatives returns arrays matching input grid shape."""
        from src.gp_derivatives import fit_gp_surface, compute_gp_derivatives

        set_all_seeds(42)
        n_S, n_t = 20, 20
        V, S_grid, t_grid = generate_price_surface(
            n_S=n_S, n_t=n_t, K=100, r=0.05, sigma=0.2, T=1.0
        )
        gp, _ = fit_gp_surface(V, S_grid, t_grid, n_subsample=150, seed=42)
        derivs = compute_gp_derivatives(gp, S_grid, t_grid)
        assert derivs['dV_dS'].shape == (n_S, n_t)
        assert derivs['dV_dt'].shape == (n_S, n_t)
        assert derivs['d2V_dS2'].shape == (n_S, n_t)
        assert derivs['V_smooth'].shape == (n_S, n_t)


class TestGPRBFDerivative1D:
    def test_gp_rbf_derivative_1d(self):
        """On a 20x20 grid sampling f(x,t)=sin(x), GP-derivative w.r.t. S
        is close to cos(x) within 0.1 (mean abs error)."""
        from src.gp_derivatives import fit_gp_surface, compute_gp_derivatives

        set_all_seeds(42)
        n_S, n_t = 20, 20
        S_grid = np.linspace(0.0, 2.0 * np.pi, n_S)
        t_grid = np.linspace(0.0, 1.0, n_t)
        S_mesh, _ = np.meshgrid(S_grid, t_grid, indexing='ij')
        V = np.sin(S_mesh)  # constant in t

        gp, _ = fit_gp_surface(V, S_grid, t_grid,
                               n_subsample=min(300, n_S * n_t), seed=42)
        derivs = compute_gp_derivatives(gp, S_grid, t_grid)

        # Compare interior points to avoid GP edge bias
        interior = slice(2, -2)
        cos_truth = np.cos(S_mesh)[interior, interior]
        gp_dVdS = derivs['dV_dS'][interior, interior]
        mae = float(np.mean(np.abs(gp_dVdS - cos_truth)))
        assert mae < 0.1, f"GP dV/dS MAE={mae:.4f} not below 0.1"


class TestGPSindyCleanReasonable:
    def test_gp_sindy_clean_reasonable(self):
        """sindy_with_gp_derivatives on clean BS data has R²(clean) > 0.7."""
        from src.gp_derivatives import sindy_with_gp_derivatives

        set_all_seeds(42)
        V, S_grid, t_grid = generate_price_surface(
            n_S=30, n_t=30, K=100, r=0.05, sigma=0.2, T=1.0
        )
        result = sindy_with_gp_derivatives(
            V, S_grid, t_grid, n_subsample=400, seed=42,
            K=100, r=0.05, sigma=0.2, T=1.0,
            true_r=0.05, true_sigma=0.2,
        )
        # Either R²(clean) or R²(noisy) being >0.7 is acceptable indication
        # the fit recovers a sensible PDE; the spec asks for >0.7.
        assert result['r2_clean'] > 0.7, (
            f"R²(clean)={result['r2_clean']:.4f} not above 0.7"
        )
