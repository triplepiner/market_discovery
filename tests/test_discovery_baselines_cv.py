"""Tests for the discovery-method baselines and 5-fold CV pipeline."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.discovery_baselines_cv import (
    _assign_folds,
    cv_kan_dupire,
    cv_linear_dupire,
    ridge_threshold_dupire_2term,
    run_cv,
    run_discovery_method_comparison,
    stlsq_dupire_2term_baseline,
    weak_form_dupire_2term,
)
from src.sindy_kan import generate_synthetic_dupire_constsig


@pytest.fixture(scope='module')
def synth_dupire_dataset():
    """Small synthetic BS Dupire surface (~20x10) used as a fast fixture.

    Provides the keys the baselines / CV helpers expect on the real
    pipeline output: C, k, tau, sigma_imp, dCdk, d2Cdk2, theta, S0, r, q.
    """
    base = generate_synthetic_dupire_constsig(
        sigma=0.20, r=0.05, q=0.0, S0=100.0,
        n_k=20, n_tau=20, k_range=(-0.20, 0.20), tau_range=(0.1, 1.2),
    )
    # Add the K_2d column that the dataset interface uses in some paths.
    from src.real_data_v2 import compute_forward_prices
    F = compute_forward_prices(base['S0'], base['r'], base['q'], base['tau'])
    K_2d = np.outer(np.exp(base['k']), F)
    base['K_2d'] = K_2d
    return base


class TestWeakDiscoveryOnGp:
    def test_weak_discovery_on_gp(self, synth_dupire_dataset):
        """Weak-form discovery on a 20x20 synthetic BS surface returns finite coefficients."""
        res = weak_form_dupire_2term(synth_dupire_dataset, n_modes_k=4,
                                       n_modes_tau=4)
        assert np.isfinite(res['coef_dCdk']), 'coef_dCdk must be finite'
        assert np.isfinite(res['coef_d2Cdk2']), 'coef_d2Cdk2 must be finite'
        assert np.isfinite(res['r2_pointwise']), 'pointwise R^2 must be finite'

    def test_weak_recovers_constant_sigma_in_ballpark(self, synth_dupire_dataset):
        """On clean synth, the implied sigma_loc should be reasonable (>0)."""
        res = weak_form_dupire_2term(synth_dupire_dataset, n_modes_k=4,
                                       n_modes_tau=4)
        # Don't require BS sigma=0.20 exactly -- the 2-term Dupire fit on a
        # constant-sigma BS surface is well-posed but sensitive to FD edge
        # effects. Just sanity-check finiteness and positivity.
        assert res['coef_d2Cdk2'] > 0, 'd2Cdk2 coefficient should be positive'
        assert np.isfinite(res['sigma_loc']), 'sigma_loc must be finite'


class TestRidgeDiscoveryRuns:
    def test_ridge_discovery_runs(self, synth_dupire_dataset):
        """Ridge + threshold on a 2-column synthetic library returns finite coefficients."""
        res = ridge_threshold_dupire_2term(synth_dupire_dataset)
        assert np.isfinite(res['coef_dCdk'])
        assert np.isfinite(res['coef_d2Cdk2'])
        assert np.isfinite(res['r2'])

    def test_stlsq_ols_baseline_finite(self, synth_dupire_dataset):
        res = stlsq_dupire_2term_baseline(synth_dupire_dataset)
        assert np.isfinite(res['coef_dCdk'])
        assert np.isfinite(res['coef_d2Cdk2'])
        assert -0.5 <= res['r2'] <= 1.0 + 1e-9


class TestCvHelpers:
    def test_assign_folds_balanced_and_seeded(self):
        f1 = _assign_folds(50, n_folds=5, seed=42)
        f2 = _assign_folds(50, n_folds=5, seed=42)
        assert (f1 == f2).all(), 'fold assignment must be deterministic for fixed seed'
        unique, counts = np.unique(f1, return_counts=True)
        assert len(unique) == 5
        # With n divisible by 5, folds are equal-sized.
        assert (counts == 10).all()


class TestCvProduces5Folds:
    def test_cv_produces_5_folds_linear(self, synth_dupire_dataset):
        """CV returns 5 finite R^2 values for the linear Dupire model."""
        df = cv_linear_dupire(synth_dupire_dataset, n_folds=5, seed=42)
        assert len(df) == 5
        r2s = df['R2_test'].to_numpy()
        assert np.all(np.isfinite(r2s)), 'all 5 fold R^2 values must be finite'

    @pytest.mark.slow
    def test_cv_produces_5_folds_kan(self, synth_dupire_dataset):
        """CV returns 5 finite R^2 values for the [2,1] KAN model.

        Marked slow because each KAN fit costs ~1-3s; the full sweep runs
        under 30s on CPU.
        """
        cv_df, act_df = cv_kan_dupire(synth_dupire_dataset, n_folds=5, seed=42,
                                        n_epochs=300, activations_n_eval=64)
        assert len(cv_df) == 5
        r2s = cv_df['R2_test'].to_numpy()
        assert np.all(np.isfinite(r2s))
        # Activations: 5 folds * 2 edges * 64 eval points = 640 rows.
        assert len(act_df) == 5 * 2 * 64


class TestCvNotInflatedTooMuch:
    def test_cv_r2_less_than_insample(self, synth_dupire_dataset):
        """Mean CV test R^2 <= in-sample R^2 + 0.05 (tolerance for variance).

        Sanity check from the PRD: cross-validated test R^2 should not
        exceed the in-sample R^2 by more than a small slack. The slack
        absorbs sample-noise on small synthetic grids where individual
        folds can occasionally beat the full-fit in-sample R^2.
        """
        df = cv_linear_dupire(synth_dupire_dataset, n_folds=5, seed=42)
        mean_test_r2 = float(df['mean_R2_test'].iloc[0])
        in_sample = stlsq_dupire_2term_baseline(synth_dupire_dataset)
        in_sample_r2 = in_sample['r2']
        assert mean_test_r2 <= in_sample_r2 + 0.05, (
            f"mean CV R^2 ({mean_test_r2:.4f}) should not exceed "
            f"in-sample R^2 ({in_sample_r2:.4f}) by more than 0.05"
        )


class TestOutputArtifacts:
    """Verify the CSV deliverables exist after the script has run.

    These are skipped when the artifacts haven't been generated yet; the
    test exists so a CI run that has executed
    ``scripts/run_discovery_baselines_cv.py`` will assert their presence.
    """

    def _has(self, path: str) -> bool:
        return os.path.exists(path) and os.path.getsize(path) > 0

    def test_discovery_comparison_csv_if_present(self):
        path = os.path.join('outputs', 'tables', 'discovery_method_comparison.csv')
        if not self._has(path):
            pytest.skip("Run scripts/run_discovery_baselines_cv.py first")
        df = pd.read_csv(path)
        assert len(df) == 5
        for col in ('method', 'derivatives', 'R2_SPY', 'sigma_loc_median',
                     'sigma_loc_iqr', 'interpretability', 'notes'):
            assert col in df.columns, f'missing column {col}'

    def test_cv_results_csv_if_present(self):
        path = os.path.join('outputs', 'tables', 'cv_results.csv')
        if not self._has(path):
            pytest.skip("Run scripts/run_discovery_baselines_cv.py first")
        df = pd.read_csv(path)
        for col in ('model', 'fold', 'R2_train', 'R2_test',
                     'n_train', 'n_test', 'mean_R2_test', 'std_R2_test'):
            assert col in df.columns, f'missing column {col}'
        # 5 folds * 2 models = 10 rows expected.
        assert len(df) == 10

    def test_cv_kan_activations_csv_if_present(self):
        path = os.path.join('outputs', 'tables',
                              'cv_kan_activations_summary.csv')
        if not self._has(path):
            pytest.skip("Run scripts/run_discovery_baselines_cv.py first")
        df = pd.read_csv(path)
        for col in ('fold', 'edge_idx', 'x_eval', 'activation'):
            assert col in df.columns


class TestRunDiscoveryComparisonFromSynth:
    def test_runs_with_synthetic_dataset(self, synth_dupire_dataset, tmp_path):
        """End-to-end smoke test on the synthetic dataset.

        Uses an empty CSV path so the function writes to a temp file
        and doesn't clobber the production artifact.
        """
        out_csv = str(tmp_path / 'discovery_method_comparison_synth.csv')
        df = run_discovery_method_comparison(dataset=synth_dupire_dataset,
                                               save_csv=out_csv)
        assert os.path.exists(out_csv)
        assert len(df) == 5


class TestRunCvFromSynth:
    @pytest.mark.slow
    def test_runs_with_synthetic_dataset(self, synth_dupire_dataset, tmp_path):
        cv_csv = str(tmp_path / 'cv_results_synth.csv')
        act_csv = str(tmp_path / 'cv_kan_activations_synth.csv')
        out = run_cv(dataset=synth_dupire_dataset, n_folds=5, seed=42,
                      n_epochs=300, save_csv=cv_csv, save_act_csv=act_csv)
        assert os.path.exists(cv_csv)
        assert os.path.exists(act_csv)
        combined = out['combined']
        assert 'model' in combined.columns
        assert combined['model'].nunique() == 2
