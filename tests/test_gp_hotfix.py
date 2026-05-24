"""
Tests for GP hotfixes (Fix #2 sparse-surface handling, Fix #3 Dupire
over-smoothing).

Each test is constrained to under ~30 seconds on CPU.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils import set_all_seeds
from src.data_generation import generate_price_surface, bs_call_price


class TestAutoSubsampleReduces:
    def test_auto_subsample_reduces(self):
        """Requesting n_subsample=1000 on a ~200-point grid auto-reduces to
        ~70% of total => <= 140 points."""
        from src.gp_derivatives import fit_gp_surface

        set_all_seeds(42)
        n_S, n_t = 14, 14  # 196 total points
        V, S_grid, t_grid = generate_price_surface(
            n_S=n_S, n_t=n_t, K=100, r=0.05, sigma=0.2, T=1.0
        )
        gp, subsample_idx = fit_gp_surface(
            V, S_grid, t_grid, n_subsample=1000, seed=42
        )
        total = n_S * n_t
        assert len(subsample_idx) <= int(total * 0.7), (
            f"Got {len(subsample_idx)} samples; expected <= {int(total * 0.7)}"
        )


class TestMaternKernelRuns:
    def test_matern_kernel_runs(self):
        """fit_gp_surface(kernel='matern') runs without error and produces
        usable derivatives."""
        from src.gp_derivatives import fit_gp_surface, compute_gp_derivatives

        set_all_seeds(42)
        V, S_grid, t_grid = generate_price_surface(
            n_S=20, n_t=20, K=100, r=0.05, sigma=0.2, T=1.0
        )
        info = fit_gp_surface(
            V, S_grid, t_grid, n_subsample=150, seed=42,
            kernel='matern', return_info=True,
        )
        assert info['kernel_used'] == 'matern'
        assert info['length_scales'].shape == (2,)

        derivs = compute_gp_derivatives(info['gp'], S_grid, t_grid)
        assert derivs['V_smooth'].shape == V.shape
        # Derivatives are finite
        assert np.all(np.isfinite(derivs['V_smooth']))
        assert np.all(np.isfinite(derivs['dV_dS']))


class TestDensityAwareForcesSavgol:
    def test_density_aware_forces_savgol(self):
        """select_derivative_strategy with n_data_points=200 and 5% noise
        forces savgol instead of gp."""
        from src.adaptive_denoiser import select_derivative_strategy

        # Without density-aware override at 5% noise, would pick 'gp'.
        strat_no_density, _ = select_derivative_strategy(0.05)
        assert strat_no_density == 'gp', (
            f"sanity: 5% noise without density -> {strat_no_density}"
        )

        strat, params = select_derivative_strategy(0.05, n_data_points=200)
        assert strat == 'savgol', (
            f"Sparse-surface override should yield savgol, got '{strat}'"
        )
        assert params.get('savgol_window') is not None


class TestConstrainedLsRuns:
    def test_constrained_ls_runs(self):
        """fit_gp_surface_constrained runs without error and respects bounds."""
        from src.gp_derivatives import (
            fit_gp_surface_constrained,
            compute_gp_derivatives,
        )

        set_all_seeds(42)
        V, S_grid, t_grid = generate_price_surface(
            n_S=20, n_t=20, K=100, r=0.05, sigma=0.2, T=1.0
        )
        gp, idx = fit_gp_surface_constrained(
            V, S_grid, t_grid, n_subsample=150, seed=42,
            ls_bounds_S=(1.0, 20.0),
            ls_init_S=5.0,
        )
        derivs = compute_gp_derivatives(gp, S_grid, t_grid)
        assert derivs['V_smooth'].shape == V.shape
        assert np.all(np.isfinite(derivs['d2V_dS2']))

        # Stratified mode also runs
        gp2, idx2 = fit_gp_surface_constrained(
            V, S_grid, t_grid, n_subsample=120, seed=42,
            stratified=True,
        )
        assert len(idx2) > 0


class TestDupireSyntheticAtLeastOneWorks:
    def test_dupire_synthetic_at_least_one_works(self):
        """compare_dupire_approaches_synthetic returns at least one approach
        with R^2 > 0.5 and sigma within 30% of 0.20."""
        from src.real_data_publication import compare_dupire_approaches_synthetic

        # Small-ish grid to stay under 30s on CPU.
        df = compare_dupire_approaches_synthetic(
            K_min=70.0, K_max=130.0, n_K=30,
            tau_min=0.05, tau_max=1.5, n_tau=30,
            S0=100.0, r=0.05, sigma=0.20,
            seed=42, n_subsample=300,
        )
        assert isinstance(df, pd.DataFrame)
        for col in ('approach', 'r2_score', 'sigma_recovered',
                    'sigma_rel_error'):
            assert col in df.columns

        # At least one approach achieves R^2 > 0.5 AND sigma within 30%.
        ok = df[
            (df['r2_score'] > 0.5)
            & (df['sigma_rel_error'] < 0.30)
        ]
        assert len(ok) >= 1, (
            f"No approach hit R^2 > 0.5 and sigma within 30%.\n"
            f"{df.to_string(index=False)}"
        )
