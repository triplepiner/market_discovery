"""
Tests for the PINN validation module (src.pinn_validation).

Uses coarse grids and few epochs to keep test runtime manageable while
still verifying that the PINN architecture, training loop, and autograd
machinery work correctly.
"""

import sys
import os
import numpy as np
import torch
import pytest

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pinn_validation import BSPINN, train_pinn, compute_pde_residual, TERM_NAMES
from src.data_generation import generate_price_surface


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coarse_surface():
    """Generate a coarse 20x20 BS call price surface for fast tests."""
    V, S_grid, t_grid = generate_price_surface(
        n_S=20, n_t=20, K=100, r=0.05, sigma=0.2, T=1.0,
    )
    return V, S_grid, t_grid


@pytest.fixture(scope="module")
def medium_surface():
    """Generate a 30x30 BS call price surface for the pricing accuracy test."""
    V, S_grid, t_grid = generate_price_surface(
        n_S=30, n_t=30, K=100, r=0.05, sigma=0.2, T=1.0,
    )
    return V, S_grid, t_grid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBSPINNForward:
    """Tests for the BSPINN network architecture."""

    def test_pinn_forward_shape(self):
        """Forward pass with scalar-like tensors should return shape (1, 1)."""
        model = BSPINN(S_min=50, S_max=150, t_min=0, t_max=0.99)
        S = torch.tensor([[100.0]], dtype=torch.float64)
        t = torch.tensor([[0.5]], dtype=torch.float64)
        output = model(S, t)
        assert output.shape == (1, 1), (
            f"Expected output shape (1, 1), got {output.shape}"
        )


class TestPINNTraining:
    """Tests for the PINN training loop."""

    def test_pinn_trains_without_error(self, coarse_surface):
        """train_pinn with 20x20 grid and 500 epochs completes without exception."""
        V, S_grid, t_grid = coarse_surface
        discovered_coefficients = [0.05, 0, 0, -0.05, -0.02]

        # This should not raise
        result = train_pinn(
            V_surface=V,
            S_grid=S_grid,
            t_grid=t_grid,
            discovered_coefficients=discovered_coefficients,
            K=100,
            r=0.05,
            sigma=0.2,
            T=1.0,
            n_epochs=500,
        )
        assert 'test_metrics' in result
        assert 'loss_history' in result
        assert 'model' in result

    def test_pinn_loss_decreases(self, coarse_surface):
        """After 500 epochs, total loss is lower than the initial loss."""
        V, S_grid, t_grid = coarse_surface
        discovered_coefficients = [0.05, 0, 0, -0.05, -0.02]

        result = train_pinn(
            V_surface=V,
            S_grid=S_grid,
            t_grid=t_grid,
            discovered_coefficients=discovered_coefficients,
            K=100,
            r=0.05,
            sigma=0.2,
            T=1.0,
            n_epochs=500,
        )

        history = result['loss_history']
        total_loss = history['total_loss']

        initial_loss = total_loss[0]
        final_loss = total_loss[-1]

        assert final_loss < initial_loss, (
            f"Final loss ({final_loss:.6e}) should be lower than "
            f"initial loss ({initial_loss:.6e})"
        )

    def test_pinn_pricing_rough_accuracy(self, medium_surface):
        """After 3000 epochs on 30x30 grid, relative L2 error < 15%."""
        V, S_grid, t_grid = medium_surface
        discovered_coefficients = [0.05, 0, 0, -0.05, -0.02]

        result = train_pinn(
            V_surface=V,
            S_grid=S_grid,
            t_grid=t_grid,
            discovered_coefficients=discovered_coefficients,
            K=100,
            r=0.05,
            sigma=0.2,
            T=1.0,
            n_epochs=3000,
        )

        rel_l2 = result['test_metrics']['relative_l2_error']
        assert rel_l2 < 0.15, (
            f"Relative L2 error = {rel_l2:.4f}, expected < 0.15"
        )


class TestAutograd:
    """Test automatic differentiation on a known function."""

    def test_autograd_derivatives(self):
        """
        For f(S, t) = S^2 * t, verify:
            df/dS = 2*S*t
            df/dt = S^2
        using torch.autograd.grad.
        """
        S = torch.tensor([[3.0]], dtype=torch.float64, requires_grad=True)
        t = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)

        f = S ** 2 * t  # f(3, 2) = 9 * 2 = 18

        # df/dS
        dfdS = torch.autograd.grad(
            f, S,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            retain_graph=True,
        )[0]

        # df/dt
        dfdt = torch.autograd.grad(
            f, t,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Expected: dfdS = 2*S*t = 2*3*2 = 12
        expected_dfdS = 2.0 * 3.0 * 2.0
        assert torch.isclose(dfdS, torch.tensor([[expected_dfdS]], dtype=torch.float64), atol=1e-10), (
            f"dfdS = {dfdS.item():.6f}, expected {expected_dfdS}"
        )

        # Expected: dfdt = S^2 = 9
        expected_dfdt = 3.0 ** 2
        assert torch.isclose(dfdt, torch.tensor([[expected_dfdt]], dtype=torch.float64), atol=1e-10), (
            f"dfdt = {dfdt.item():.6f}, expected {expected_dfdt}"
        )
