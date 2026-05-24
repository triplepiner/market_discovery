"""
Tests for SINDy library standardization and real-data diagnostic reporting.

Covers:
- discover_pde(standardize=True) preserves R^2 vs standardize=False on
  clean synthetic Black-Scholes data.
- The back-transformation from standardized to physical coefficients is
  correct (i.e. yields identical numeric coefficients within tight tol).
- The active-term selection is identical on clean data.
- diagnose_real_data_quality returns a dict with the expected schema and
  runs without crash on a mock surface.
"""

import os
import sys
import numpy as np
import pytest

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generation import generate_price_surface
from src.sindy_discovery import discover_pde, TERM_NAMES
from src.real_data_analysis import (
    diagnose_real_data_quality, compute_term_contributions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clean_surface_50():
    """Clean 50x50 Black-Scholes call surface (sigma=0.2, r=0.05)."""
    V, S_grid, t_grid = generate_price_surface(n_S=50, n_t=50)
    return V, S_grid, t_grid


@pytest.fixture(scope="module")
def both_runs(clean_surface_50):
    """Run discover_pde with standardize=False and standardize=True."""
    V, S, t = clean_surface_50
    no_std = discover_pde(V, S, t, true_sigma=0.2, true_r=0.05,
                          standardize=False)
    with_std = discover_pde(V, S, t, true_sigma=0.2, true_r=0.05,
                            standardize=True)
    return no_std, with_std


# ---------------------------------------------------------------------------
# Tests on standardization
# ---------------------------------------------------------------------------

def test_standardize_preserves_r2(both_runs):
    """R^2 must be identical (within 1e-10) on clean data."""
    no_std, with_std = both_runs
    assert abs(no_std['r2_score'] - with_std['r2_score']) < 1e-10, (
        f"R^2 mismatch: no_std={no_std['r2_score']:.16f}, "
        f"with_std={with_std['r2_score']:.16f}"
    )


def test_back_transformation_correct(both_runs):
    """Physical coefficients must be identical (within 1e-8) on clean data."""
    no_std, with_std = both_runs
    c1 = np.asarray(no_std['discovered_coefficients'], dtype=float)
    c2 = np.asarray(with_std['discovered_coefficients'], dtype=float)
    diff = np.max(np.abs(c1 - c2))
    assert diff < 1e-8, (
        f"Physical coefficient mismatch: max |diff|={diff:.3e}\n"
        f"no_std   = {c1}\nwith_std = {c2}"
    )


def test_active_terms_identical(both_runs):
    """Active terms detected must be the same on clean synthetic data."""
    no_std, with_std = both_runs
    assert set(no_std['active_terms']) == set(with_std['active_terms']), (
        f"Active terms differ: no_std={no_std['active_terms']} "
        f"vs with_std={with_std['active_terms']}"
    )
    # And the standardization_used flag must be propagated correctly.
    assert no_std['standardization_used'] is False
    assert with_std['standardization_used'] is True


# ---------------------------------------------------------------------------
# Tests on diagnose_real_data_quality
# ---------------------------------------------------------------------------

def _mock_surface_and_option_data():
    """Build a small mock surface + option_data dict for diagnose tests."""
    V, S_grid, t_grid = generate_price_surface(n_S=20, n_t=20)
    # surface_data shape: (n_K, n_tau); we just reuse S, t as K, tau.
    surface_data = {
        'V_surface': V,
        'K_grid': S_grid,
        'tau_grid': t_grid,
        'S0': float(np.median(S_grid)),
        'r': 0.05,
        'iv_surface': np.full_like(V, 0.2),
        'n_valid_points': V.size,
    }
    option_data = {
        'S0': float(np.median(S_grid)),
        'r': 0.05,
        'implied_vols': np.full(V.size, 0.20),
        'ticker': 'MOCK',
    }
    return surface_data, option_data


def test_diagnose_returns_expected_keys(capsys):
    """diagnose_real_data_quality runs without crash and returns the
    documented set of keys."""
    surface_data, option_data = _mock_surface_and_option_data()
    report = diagnose_real_data_quality(option_data, surface_data, 'MOCK')

    required_keys = {
        'ticker', 'surface_shape', 'V_stats', 'K_range', 'K_spacing',
        'tau_range', 'tau_spacing', 'derivative_stats', 'bs_expected',
        'library_col_max', 'library_col_std', 'condition_number',
        'correlation_matrix', 'corr_diag_max', 'corr_offdiag_max',
    }
    missing = required_keys - set(report.keys())
    assert not missing, f"Missing keys: {missing}"

    # Sanity-check a few values
    assert report['ticker'] == 'MOCK'
    assert report['surface_shape'] == (20, 20)
    for key in ('min', 'max', 'mean', 'std'):
        assert key in report['V_stats']
    for d in ('dV/dt', 'dV/dK', 'd2V/dK2'):
        assert d in report['derivative_stats']
        for k in ('min', 'max', 'mean', 'std', 'abs_max', 'abs_mean'):
            assert k in report['derivative_stats'][d]
    assert report['condition_number'] > 0
    assert 0.0 <= report['corr_offdiag_max'] <= 1.0 + 1e-12

    # A report was printed to stdout
    captured = capsys.readouterr()
    assert 'DATA-QUALITY DIAGNOSTIC' in captured.out
    assert 'MOCK' in captured.out


# ---------------------------------------------------------------------------
# Tests on compute_term_contributions and standardized cond reporting
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def contrib_df_synthetic(clean_surface_50):
    """Term contributions from the standardize=True run on clean BS data."""
    V, S, t = clean_surface_50
    res = discover_pde(V, S, t, true_sigma=0.2, true_r=0.05, standardize=True,
                       trim=5)
    df = compute_term_contributions(res, V, S, t, trim=5, smooth=False)
    return df, res


def test_contribution_fractions_sum_to_one(contrib_df_synthetic):
    """fraction_of_total across terms must sum to ~1.0 within 0.01."""
    df, _ = contrib_df_synthetic
    total = float(df['fraction_of_total'].sum())
    assert abs(total - 1.0) < 0.01, (
        f"Fractions sum to {total:.6f}, expected ~1.0"
    )


def test_contribution_positive(contrib_df_synthetic):
    """All mean_abs_contribution values must be >= 0."""
    df, _ = contrib_df_synthetic
    assert (df['mean_abs_contribution'] >= 0).all(), (
        f"Negative contribution found:\n{df}"
    )


def test_bs_synthetic_dominant_terms(contrib_df_synthetic):
    """
    On clean synthetic BS data, the top-3 contributing terms must include
    V, S*dV/dS, S^2*d2V/dS^2 (the three true BS terms), even though spurious
    bare-derivative terms may still appear in the raw coefficient vector.
    """
    df, _ = contrib_df_synthetic
    top3 = set(
        df.sort_values('mean_abs_contribution', ascending=False)
          .head(3)['term'].tolist()
    )
    bs_terms = {'V', 'S*dV/dS', 'S2*d2V/dS2'}
    missing = bs_terms - top3
    assert not missing, (
        f"Top-3 contributors {top3} miss BS terms {missing}.\nFull table:\n{df}"
    )


def test_standardized_cond_is_smaller(clean_surface_50):
    """
    When standardize=True on synthetic data, the standardized condition
    number must be at least 10x smaller than the raw condition number.
    This demonstrates that standardization actually reconditions the
    library matrix (and that we report both numbers correctly).
    """
    V, S, t = clean_surface_50
    res = discover_pde(V, S, t, true_sigma=0.2, true_r=0.05, standardize=True)
    cond_raw = res['condition_number_raw']
    cond_std = res['condition_number_standardized']
    assert cond_std is not None, "condition_number_standardized missing"
    assert cond_raw > 0 and cond_std > 0
    assert cond_std < cond_raw / 10.0, (
        f"Standardization did not reduce cond by >=10x: "
        f"raw={cond_raw:.3e}, std={cond_std:.3e}"
    )
