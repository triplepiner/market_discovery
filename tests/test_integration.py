"""
Integration test: run the full pipeline (data generation -> SINDy discovery
-> PINN validation) on a tiny grid as a smoke test.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_generation import generate_price_surface
from src.sindy_discovery import discover_pde, TERM_NAMES
from src.pinn_validation import train_pinn


@pytest.mark.slow
class TestFullPipelineMini:
    """
    End-to-end smoke test on a 20x20 grid with 1000 PINN epochs.

    This verifies that all pipeline stages run without error and produce
    structurally correct outputs. It does NOT assert tight numerical
    accuracy (the grid and epoch count are far too small for that).
    """

    def test_full_pipeline_mini(self):
        # -----------------------------------------------------------------
        # Step 1: Generate a call price surface on a tiny grid
        # -----------------------------------------------------------------
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 1.0

        V, S_grid, t_grid = generate_price_surface(
            n_S=20, n_t=20, K=K, r=r, sigma=sigma, T=T, option_type='call'
        )

        assert V.shape == (20, 20), f"Expected shape (20,20), got {V.shape}"
        assert S_grid.shape == (20,)
        assert t_grid.shape == (20,)

        # -----------------------------------------------------------------
        # Step 2: Discover the PDE via SINDy
        # -----------------------------------------------------------------
        sindy_result = discover_pde(
            V, S_grid, t_grid,
            true_sigma=sigma, true_r=r,
            K=K, T=T, option_type='call',
        )

        # The Black-Scholes PDE has 3 true active terms: V, S*dV/dS, S^2*d2V/dS2.
        # On a tiny 20x20 grid (10x10 after trimming), collinearity between the
        # raw and S-weighted derivative terms prevents STLSQ from achieving full
        # sparsity, so the solver may keep extra terms. We verify the 3 expected
        # terms are among the active set and at least 3 terms are selected.
        n_active = sindy_result['n_active']
        assert n_active >= 3, (
            f"Expected at least 3 active SINDy terms for Black-Scholes, got {n_active}. "
            f"Active terms: {sindy_result['active_terms']}"
        )

        # The 3 physically-required terms must be present
        expected_terms = {'V', 'S*dV/dS', 'S2*d2V/dS2'}
        active_set = set(sindy_result['active_terms'])
        missing_terms = expected_terms - active_set
        assert not missing_terms, (
            f"Missing expected Black-Scholes terms: {missing_terms}. "
            f"Active terms: {sindy_result['active_terms']}"
        )

        discovered_coeffs = sindy_result['discovered_coefficients']
        assert len(discovered_coeffs) == 5, (
            f"Coefficient vector length should be 5, got {len(discovered_coeffs)}"
        )

        # R^2 should be very high on clean data
        assert sindy_result['r2_score'] > 0.99, (
            f"SINDy R^2 = {sindy_result['r2_score']:.4f}, expected > 0.99"
        )

        # -----------------------------------------------------------------
        # Step 3: Train a PINN with discovered coefficients (short run)
        # -----------------------------------------------------------------
        pinn_result = train_pinn(
            V_surface=V,
            S_grid=S_grid,
            t_grid=t_grid,
            discovered_coefficients=discovered_coeffs,
            term_names=TERM_NAMES,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            option_type='call',
            n_epochs=1000,
            lr=1e-3,
        )

        # -----------------------------------------------------------------
        # Step 4: Verify PINN results dict has expected keys
        # -----------------------------------------------------------------
        expected_keys = {
            'test_metrics',
            'loss_history',
            'model',
            'sanity_checks',
            'discovered_coefficients',
            'term_names',
            'option_type',
            'training_params',
            'full_grid_metrics',
            'V_predicted',
            'V_analytical',
        }
        actual_keys = set(pinn_result.keys())
        missing = expected_keys - actual_keys
        assert not missing, f"PINN result dict is missing keys: {missing}"

        # Test metrics sub-dict
        test_metrics = pinn_result['test_metrics']
        for metric_key in ('relative_l2_error', 'mae', 'max_error', 'r2'):
            assert metric_key in test_metrics, (
                f"test_metrics missing key '{metric_key}'"
            )

        # Loss history should have entries for each epoch that ran
        loss_history = pinn_result['loss_history']
        assert len(loss_history['total_loss']) > 0, "Loss history is empty"
        assert len(loss_history['total_loss']) <= 1000, (
            "Loss history longer than n_epochs"
        )

        # V_predicted shape should match the input surface
        assert pinn_result['V_predicted'].shape == V.shape, (
            f"V_predicted shape {pinn_result['V_predicted'].shape} != "
            f"input shape {V.shape}"
        )


class TestNewModulesImport:
    """Verify all new modules import without error."""
    def test_import_baselines(self):
        from src.baselines import run_all_baselines

    def test_import_extended_models(self):
        from src.extended_models import run_merton_experiment, run_heston_variance_slicing

    def test_import_ablation(self):
        from src.ablation import run_all_ablation_experiments

    def test_import_real_data(self):
        from src.real_data import run_real_data_experiment
