"""Tests for the misspecification diagnostic taxonomy (src/misspec_taxonomy.py)."""
import numpy as np
import pandas as pd
import pytest

from src.misspec_taxonomy import (
    jump_intensity_sweep,
    jump_size_sweep,
    stochvol_sweep,
    heston_call_price_scalar,
    generate_heston_surface,
)
from src.data_generation import bs_call_price


# A trimmed parameter list keeps unit tests fast while still exercising the
# code path.  The full taxonomy run (run_misspec_taxonomy) uses the defaults.
_FAST_LAMS    = (0.05, 0.20)
_FAST_MU_JS   = (-0.10, 0.10)
_FAST_SIGMAVS = (0.0, 0.3)


class TestJumpIntensitySweep:
    def test_jump_sweep_runs(self):
        """SINDy completes for all six lambda values without error and
        returns a DataFrame with the expected schema and row count."""
        df = jump_intensity_sweep()
        # The default sweep has six lambda values.
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6
        for col in [
            'experiment', 'parameter_name', 'parameter_value',
            'coef_V', 'coef_dVdS', 'coef_d2VdS2',
            'coef_S_dVdS', 'coef_S2_d2VdS2', 'R2',
        ]:
            assert col in df.columns, f"missing column {col}"
        assert (df['experiment'] == 'jump_intensity').all()
        # R^2 finite and bounded
        assert df['R2'].notna().all()
        assert (df['R2'] <= 1.0 + 1e-9).all()


class TestJumpCoefficientScales:
    def test_jump_coefficient_scales(self):
        """The spurious d^2V/dS^2 coefficient at lambda=0.20 must be
        strictly larger in absolute value than at lambda=0.05.  This is
        the key falsifiable claim of Experiment A."""
        # Use only the two values we need — 2 surfaces is ~0.1s.
        df = jump_intensity_sweep(lambdas=_FAST_LAMS)
        c05 = float(df.loc[df['parameter_value'] == 0.05, 'coef_d2VdS2'].iloc[0])
        c20 = float(df.loc[df['parameter_value'] == 0.20, 'coef_d2VdS2'].iloc[0])
        assert abs(c20) > abs(c05), (
            f"|d2V/dS2 coef| at lambda=0.20 ({c20}) should exceed "
            f"the value at lambda=0.05 ({c05})."
        )


class TestStochvolSweep:
    def test_stochvol_sweep_runs(self):
        """Heston-based sweep completes for all sigma_v values and returns
        a DataFrame with one row per sigma_v."""
        df = stochvol_sweep(sigma_vs=_FAST_SIGMAVS, n_S=30, n_t=30)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(_FAST_SIGMAVS)
        assert (df['experiment'] == 'stochvol').all()
        # R^2 finite
        assert df['R2'].notna().all()
        # All coefficient columns present and finite
        for col in ['coef_V', 'coef_dVdS', 'coef_d2VdS2',
                    'coef_S_dVdS', 'coef_S2_d2VdS2']:
            assert np.isfinite(df[col].values).all(), f"{col} not finite"


class TestHestonPricerSanity:
    def test_heston_reduces_to_bs(self):
        """sigma_v=0 with theta=v0 collapses to constant-variance BS."""
        S, K, r, tau, v0 = 100.0, 100.0, 0.05, 0.5, 0.04
        bs = float(bs_call_price(np.array([S]), K, r, np.sqrt(v0), tau)[0])
        hes = heston_call_price_scalar(
            S, K, r, tau,
            v0=v0, kappa=2.0, theta=v0, sigma_v=0.0, rho=-0.5,
        )
        np.testing.assert_allclose(hes, bs, rtol=1e-6)

    def test_heston_surface_shape_and_finiteness(self):
        V, S_grid, t_grid = generate_heston_surface(
            n_S=12, n_t=12, sigma_v=0.3
        )
        assert V.shape == (12, 12)
        assert len(S_grid) == 12
        assert len(t_grid) == 12
        assert np.isfinite(V).all()
        # Prices should be (approximately) non-negative.  Tolerance is
        # generous because the semi-analytic Heston integral can produce
        # tiny negative values for deep-OTM very-short-maturity grid cells
        # due to numerical quadrature error in the characteristic function.
        # The downstream SINDy fit uses interior trimmed cells where this
        # numerical artifact is washed out.
        assert (V >= -0.1).all()


class TestJumpSizeSweepRuns:
    def test_jump_size_sweep_runs_smoke(self):
        """Fast smoke test that Experiment B returns a DataFrame for at
        least two mu_J values."""
        df = jump_size_sweep(mu_Js=_FAST_MU_JS)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(_FAST_MU_JS)
        assert (df['experiment'] == 'jump_size').all()
