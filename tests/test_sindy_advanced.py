"""
Tests for advanced SINDy variants: ensemble, PCA, time-varying, CV-threshold
selection, and bootstrap confidence intervals.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generation import generate_price_surface
from src.sindy_discovery import (
    ensemble_sindy,
    pca_sindy,
    time_varying_sindy,
    cv_threshold_select,
    bootstrap_confidence_intervals,
    TERM_NAMES,
)


def generate_bs_data(S_min=50, S_max=150, n_S=30, t_min=0.0, t_max=0.99,
                     n_t=30, K=100, r=0.05, sigma=0.2, T=1.0):
    """Thin wrapper around generate_price_surface for the test signature."""
    return generate_price_surface(
        S_min=S_min, S_max=S_max, n_S=n_S,
        t_min=t_min, t_max=t_max, n_t=n_t,
        K=K, r=r, sigma=sigma, T=T, option_type='call',
    )


@pytest.fixture(scope="module")
def small_surface():
    V, S, t = generate_bs_data(n_S=30, n_t=30)
    return V, S, t


# ---------------------------------------------------------------------------
# Ensemble SINDy (#5)
# ---------------------------------------------------------------------------

def test_ensemble_inclusion_probabilities_valid(small_surface):
    V, S, t = small_surface
    result = ensemble_sindy(V, S, t, threshold=0.05, n_bootstraps=20, seed=42)
    probs = result['inclusion_probabilities']
    assert probs.shape == (5,)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.all(result['ci_low'] <= result['ci_high'] + 1e-12)
    assert result['n_bootstraps'] == 20
    assert len(result['term_names']) == 5


def test_ensemble_true_terms_high_prob_clean_data(small_surface):
    V, S, t = small_surface
    # Use a threshold below true_r=0.05 so the V term has a chance of being
    # retained.  Even on clean data with default thresholds, multicollinearity
    # between dV/dS and S*dV/dS can suppress the true terms.
    result = ensemble_sindy(V, S, t, threshold=0.02, n_bootstraps=20, seed=42)
    probs = result['inclusion_probabilities']
    # True BS terms are indices 0 (V), 3 (S*dV/dS), 4 (S^2 d2V/dS2)
    assert probs[0] > 0.7, f"V inclusion prob too low: {probs[0]:.2f}"
    assert probs[3] > 0.7, f"S*dV/dS inclusion prob too low: {probs[3]:.2f}"
    assert probs[4] > 0.7, f"S^2*d2V/dS2 inclusion prob too low: {probs[4]:.2f}"


# ---------------------------------------------------------------------------
# PCA SINDy (#6)
# ---------------------------------------------------------------------------

def test_pca_sindy_returns_valid_coefficients(small_surface):
    V, S, t = small_surface
    result = pca_sindy(V, S, t, threshold=0.05, secondary_threshold=0.05)
    coeffs = result['discovered_coefficients']
    assert coeffs.shape == (5,)
    assert np.all(np.isfinite(coeffs))
    assert result['n_active'] == int(np.sum(np.abs(coeffs) > 0))
    assert len(result['term_names']) == 5


# ---------------------------------------------------------------------------
# Time-varying SINDy (#11)
# ---------------------------------------------------------------------------

def test_time_varying_clean_bs_autonomous(small_surface):
    V, S, t = small_surface
    result = time_varying_sindy(
        V, S, t, window_size=15, stride=3, threshold=0.05
    )
    assert result['coefficients_per_window'].ndim == 2
    assert result['coefficients_per_window'].shape[1] == 5
    assert len(result['window_centers']) > 0
    # Clean BS data is autonomous: coefficient variance should be small.
    assert result['is_autonomous'] is True


# ---------------------------------------------------------------------------
# CV threshold selection (#13)
# ---------------------------------------------------------------------------

def test_cv_threshold_returns_positive(small_surface):
    V, S, t = small_surface
    best, scores = cv_threshold_select(
        V, S, t,
        candidate_thresholds=np.logspace(-2, 0, 5),
        n_folds=5, seed=42,
    )
    assert best > 0
    assert isinstance(scores, dict)
    assert len(scores) == 5


# ---------------------------------------------------------------------------
# Bootstrap CIs (#14)
# ---------------------------------------------------------------------------

def test_bootstrap_ci_validity(small_surface):
    V, S, t = small_surface
    df = bootstrap_confidence_intervals(
        V, S, t, threshold=0.05, n_bootstraps=20, seed=42
    )
    assert list(df.columns) == [
        'term', 'point_estimate', 'ci_low', 'ci_high', 'ci_contains_zero'
    ]
    assert len(df) == 5
    assert (df['ci_low'] <= df['ci_high'] + 1e-12).all()
    assert set(df['term']) == set(TERM_NAMES)
