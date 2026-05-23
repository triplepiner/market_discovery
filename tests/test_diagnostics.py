"""
Tests for src.diagnostics -- data leakage detection, overfitting analysis,
and training convergence monitoring.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.diagnostics import (
    check_data_leakage,
    check_overfitting,
    check_training_convergence,
)
from src.data_generation import generate_price_surface
from src.sindy_discovery import discover_pde
from src.visualization import plot_residual_heatmap


# ---------------------------------------------------------------------------
# 1. Data leakage check with clean (non-overlapping) splits
# ---------------------------------------------------------------------------
class TestDataLeakageCheckClean:
    def test_data_leakage_check_clean(self):
        """
        Non-overlapping splits that cover all 100 points should pass
        without raising an AssertionError.
        """
        n_total = 100
        all_indices = np.arange(n_total)
        np.random.seed(0)
        np.random.shuffle(all_indices)

        train_indices = all_indices[:60]
        val_indices = all_indices[60:80]
        test_indices = all_indices[80:]

        # Should complete without raising
        check_data_leakage(train_indices, val_indices, test_indices, n_total)


# ---------------------------------------------------------------------------
# 2. Data leakage check with overlapping splits
# ---------------------------------------------------------------------------
class TestDataLeakageCheckOverlap:
    def test_data_leakage_check_overlap(self):
        """
        Overlapping splits should raise an AssertionError because indices
        appear in more than one set.
        """
        n_total = 100
        train_indices = np.arange(0, 70)   # 0..69
        val_indices = np.arange(60, 85)     # 60..84  -- overlaps with train
        test_indices = np.arange(85, 100)   # 85..99

        with pytest.raises(AssertionError):
            check_data_leakage(train_indices, val_indices, test_indices, n_total)


# ---------------------------------------------------------------------------
# 3. Overfitting detection
# ---------------------------------------------------------------------------
class TestOverfittingDetection:
    def test_overfitting_detection(self):
        """
        When training loss monotonically decreases but validation loss
        increases in the second half, check_overfitting must flag
        is_overfitting=True.
        """
        n_epochs = 100
        # Training loss: steadily decreasing
        train_losses = np.linspace(1.0, 0.01, n_epochs)
        # Validation loss: decreases then increases (U-shaped)
        val_first_half = np.linspace(1.0, 0.1, n_epochs // 2)
        val_second_half = np.linspace(0.1, 0.8, n_epochs // 2)
        val_losses = np.concatenate([val_first_half, val_second_half])

        result = check_overfitting(train_losses, val_losses)

        assert result['is_overfitting'] is True, (
            f"Expected is_overfitting=True but got {result['is_overfitting']}. "
            f"val_increasing={result['val_increasing']}, "
            f"overfit_ratio={result['overfit_ratio']:.2f}"
        )
        assert 'best_epoch' in result
        assert 'recommendation' in result
        assert result['best_epoch'] < n_epochs - 1, (
            "Best epoch should not be the last epoch when overfitting"
        )


# ---------------------------------------------------------------------------
# 4. Convergence detection
# ---------------------------------------------------------------------------
class TestConvergenceDetection:
    def test_convergence_detection(self):
        """
        A loss curve that decays quickly and then plateaus for many epochs
        should be flagged as converged=True (given enough total epochs).
        """
        n_epochs = 10000
        # Exponential decay to a plateau
        epochs = np.arange(n_epochs, dtype=np.float64)
        loss_history = 1.0 * np.exp(-epochs / 500.0) + 0.001

        # Add tiny noise to make it realistic
        rng = np.random.RandomState(42)
        loss_history += rng.normal(0, 1e-5, size=n_epochs)
        loss_history = np.maximum(loss_history, 0.0)

        result = check_training_convergence(loss_history, min_epochs=5000)

        assert result['converged'] is True, (
            f"Expected converged=True but got {result['converged']}. "
            f"tail_relative_change={result['tail_relative_change']:.6e}, "
            f"n_epochs={result['n_epochs']}"
        )
        assert result['n_epochs'] == n_epochs
        assert result['final_loss'] > 0
        assert result['tail_relative_change'] < 0.01, (
            f"Tail relative change {result['tail_relative_change']:.6e} "
            f"should be < 0.01 for a converged curve"
        )


# ---------------------------------------------------------------------------
# 5. Residual heatmap plotting (Improvement #15)
# ---------------------------------------------------------------------------
class TestResidualHeatmap:
    def test_residual_heatmap_creates_png(self):
        """
        Generate clean BS data, run discover_pde, call plot_residual_heatmap,
        and verify the PNG file exists with non-zero size.
        """
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 1.0

        V, S_grid, t_grid = generate_price_surface(
            S_min=50, S_max=150, n_S=60,
            t_min=0.0, n_t=60,
            K=K, r=r, sigma=sigma, T=T, option_type='call',
        )

        sindy_results = discover_pde(
            V, S_grid, t_grid,
            true_sigma=sigma, true_r=r, K=K, T=T, option_type='call',
        )

        output_filename = 'test_residual_clean.png'
        path = plot_residual_heatmap(
            V, S_grid, t_grid,
            sindy_results['discovered_coefficients'],
            sindy_results['term_names'],
            output_filename,
            K=K,
            title='Test PDE Residual Heatmap',
        )

        assert path is not None, "plot_residual_heatmap returned None"
        assert os.path.exists(path), f"Expected PNG file at {path} not found"
        assert os.path.getsize(path) > 0, f"PNG file at {path} is empty"
