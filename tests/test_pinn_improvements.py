"""
Tests for PINN improvements #2 (hard-constraint), #8 (log-price), and #16 (warmup).
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pinn_validation import (
    HardConstraintPINN,
    LogPricePINN,
    _make_warmup_lr_lambda,
    train_hard_constraint_pinn,
    train_log_price_pinn,
)


# ---------------------------------------------------------------------------
# Improvement #2 -- Hard-constraint PINN
# ---------------------------------------------------------------------------

class TestHardConstraintPINN:

    def test_hard_constraint_exact_at_maturity(self):
        """At t=T, V(S, T) should equal payoff(S) exactly (architecture trick)."""
        torch.manual_seed(0)
        model = HardConstraintPINN(
            S_min=10.0, S_max=200.0, t_min=0.0, t_max=0.99,
            T=1.0, K=100.0, option_type='call', width=16,
        )
        S = torch.linspace(10.0, 200.0, 50, dtype=torch.float64).unsqueeze(-1)
        t = torch.full_like(S, 1.0)  # t = T
        with torch.no_grad():
            V = model(S, t).squeeze().numpy()
        expected = np.maximum(S.squeeze().numpy() - 100.0, 0.0)
        max_err = float(np.max(np.abs(V - expected)))
        assert max_err < 1e-5, f"Hard constraint violated at maturity: max_err={max_err:.3e}"

    def test_hard_constraint_exact_at_maturity_put(self):
        """Same check for puts."""
        torch.manual_seed(0)
        model = HardConstraintPINN(
            S_min=10.0, S_max=200.0, t_min=0.0, t_max=0.99,
            T=1.0, K=100.0, option_type='put', width=16,
        )
        S = torch.linspace(10.0, 200.0, 50, dtype=torch.float64).unsqueeze(-1)
        t = torch.full_like(S, 1.0)
        with torch.no_grad():
            V = model(S, t).squeeze().numpy()
        expected = np.maximum(100.0 - S.squeeze().numpy(), 0.0)
        max_err = float(np.max(np.abs(V - expected)))
        assert max_err < 1e-5

    def test_hard_constraint_pinn_trains(self):
        """Short training run completes without errors and returns expected keys."""
        result = train_hard_constraint_pinn(
            option_type='call', n_epochs=200, n_S=15, n_t=15,
            width=32, n_collocation=200, device='cpu',
        )
        assert result['model'] is not None
        assert 'test_metrics' in result
        assert 'relative_l2_error' in result['test_metrics']
        assert 'boundary_error' in result
        assert len(result['train_loss_history']) == 200

    def test_hard_constraint_loss_decreases(self):
        """Final loss < initial loss."""
        result = train_hard_constraint_pinn(
            option_type='call', n_epochs=200, n_S=15, n_t=15,
            width=32, n_collocation=200, device='cpu',
        )
        train_hist = result['train_loss_history']
        assert train_hist[-1] < train_hist[0], (
            f"final={train_hist[-1]:.4e} >= initial={train_hist[0]:.4e}"
        )

    def test_hard_constraint_boundary_error_small(self):
        """Boundary error at maturity should be effectively zero by construction."""
        result = train_hard_constraint_pinn(
            option_type='call', n_epochs=50, n_S=10, n_t=10,
            width=16, n_collocation=100, device='cpu',
        )
        assert result['boundary_error'] < 1e-6


# ---------------------------------------------------------------------------
# Improvement #8 -- Log-price PINN
# ---------------------------------------------------------------------------

class TestLogPricePINN:

    def test_log_price_pinn_forward_shape(self):
        """Forward returns shape matching input."""
        torch.manual_seed(0)
        model = LogPricePINN(
            S_min=10.0, S_max=200.0, t_min=0.0, t_max=0.99,
            T=1.0, K=100.0, option_type='call', width=16,
        )
        S = torch.linspace(10.0, 200.0, 7, dtype=torch.float64).unsqueeze(-1)
        t = torch.linspace(0.0, 0.99, 7, dtype=torch.float64).unsqueeze(-1)
        out = model(S, t)
        assert out.shape == (7, 1)

    def test_log_price_pinn_trains(self):
        """Short training completes."""
        result = train_log_price_pinn(
            option_type='call', n_epochs=200, n_S=15, n_t=15,
            width=32, n_collocation=200, device='cpu',
        )
        assert result['model'] is not None
        assert len(result['train_loss_history']) == 200
        assert np.isfinite(result['test_metrics']['relative_l2_error'])

    def test_log_price_pinn_with_hard_constraint(self):
        """Combined log-price + hard-constraint also trains."""
        result = train_log_price_pinn(
            option_type='call', n_epochs=100, n_S=10, n_t=10,
            width=16, n_collocation=100, device='cpu',
            use_hard_constraint=True,
        )
        assert result['model'] is not None
        assert result['boundary_error'] < 1e-6


# ---------------------------------------------------------------------------
# Improvement #16 -- Warmup schedule
# ---------------------------------------------------------------------------

class TestWarmupSchedule:

    def test_warmup_lr_schedule(self):
        """At epoch 0, lr ≈ initial_lr; at warmup_epochs, lr == target_lr."""
        target_lr = 1e-3
        initial_lr = 1e-4
        warmup_epochs = 1000
        lr_lambda = _make_warmup_lr_lambda(
            warmup_epochs=warmup_epochs,
            target_lr=target_lr,
            initial_lr=initial_lr,
        )
        # Compose with an optimizer the same way LambdaLR does (base_lr * lambda).
        lr_at_0 = target_lr * lr_lambda(0)
        lr_at_warmup = target_lr * lr_lambda(warmup_epochs)
        lr_after = target_lr * lr_lambda(warmup_epochs + 500)

        assert abs(lr_at_0 - initial_lr) < 1e-12, (
            f"lr at epoch 0: got {lr_at_0}, expected {initial_lr}"
        )
        assert abs(lr_at_warmup - target_lr) < 1e-12, (
            f"lr at warmup_epochs: got {lr_at_warmup}, expected {target_lr}"
        )
        assert abs(lr_after - target_lr) < 1e-12

    def test_warmup_integration_with_training(self):
        """Hard-constraint PINN with warmup runs successfully."""
        result = train_hard_constraint_pinn(
            option_type='call', n_epochs=100, n_S=10, n_t=10,
            width=16, n_collocation=100, device='cpu',
            use_warmup=True, warmup_epochs=50, warmup_initial_lr=1e-5,
        )
        assert result['model'] is not None
        assert result['used_warmup'] is True
