"""
Tests for the Dupire PDE discovery module (src.dupire_discovery).

Validates that the (K, tau) pipeline correctly recovers the Dupire PDE
structure from analytical BS call surfaces and survives the tiny / sparse
input cases that real option chains produce.
"""

import os
import sys

import numpy as np
import pytest

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generation import bs_call_price
from src.dupire_discovery import (
    build_dupire_library,
    discover_dupire,
    dupire_sanity_check,
    DUPIRE_TERM_NAMES,
)


@pytest.fixture(scope="module")
def small_bs_surface():
    """A modest analytical BS call surface for fast tests."""
    K_grid = np.linspace(70.0, 130.0, 40, dtype=np.float64)
    tau_grid = np.linspace(0.05, 1.5, 40, dtype=np.float64)
    S0, r, sigma = 100.0, 0.05, 0.20

    C = np.zeros((len(K_grid), len(tau_grid)), dtype=np.float64)
    for j, tau in enumerate(tau_grid):
        C[:, j] = bs_call_price(S0, K_grid, r, sigma, float(tau))
    return C, K_grid, tau_grid


def test_library_shape(small_bs_surface):
    """build_dupire_library returns the right shapes and names."""
    C, K_grid, tau_grid = small_bs_surface

    target, library, term_names = build_dupire_library(
        C, K_grid, tau_grid, smooth=True,
    )

    n_expected = C.size
    assert library.shape == (n_expected, 5)
    assert target.shape == (n_expected,)
    assert term_names == list(DUPIRE_TERM_NAMES)
    assert library.dtype == np.float64
    assert target.dtype == np.float64
    # Library should be finite -- smoothing/derivatives must not produce NaNs.
    assert np.all(np.isfinite(library))
    assert np.all(np.isfinite(target))


def test_sanity_check_passes():
    """dupire_sanity_check should recover sigma ~ 0.20 with R^2 > 0.99."""
    result = dupire_sanity_check(
        K_min=70, K_max=130, n_K=80,
        tau_min=0.05, tau_max=1.5, n_tau=80,
        S0=100, r=0.05, sigma=0.20, seed=42,
    )

    assert result['r2_score'] > 0.99, (
        f"Expected R^2 > 0.99, got {result['r2_score']:.6f}"
    )
    assert np.isfinite(result['sigma_discovered']), (
        f"sigma_discovered must be finite; got {result['sigma_discovered']}"
    )
    sigma_err = abs(result['sigma_discovered'] - 0.20) / 0.20
    assert sigma_err < 0.05, (
        f"sigma error {sigma_err:.4f} > 5%; "
        f"sigma_discovered={result['sigma_discovered']:.6f}"
    )


def test_sign_convention(small_bs_surface):
    """Call prices increase with tau, so the target dC/dT must be positive."""
    C, K_grid, tau_grid = small_bs_surface

    # 1. The raw surface itself: C(K, tau_high) > C(K, tau_low) for any K.
    diffs = np.diff(C, axis=1)
    assert (diffs >= -1e-10).all(), (
        "Call surface should be non-decreasing in tau."
    )

    # 2. The pipeline target dC/dT should be strictly positive on average.
    target, _, _ = build_dupire_library(
        C, K_grid, tau_grid, smooth=True,
    )
    assert target.mean() > 0, (
        f"Expected positive mean dC/dT, got {target.mean():.6e}"
    )
    # Vast majority of grid points should also be positive (allow a small
    # boundary fraction for finite-difference edges).
    positive_frac = float((target > 0).mean())
    assert positive_frac > 0.95, (
        f"Only {positive_frac:.3f} of dC/dT samples positive; "
        "sign convention likely flipped."
    )


def test_handles_sparse_data():
    """A 5x5 mock surface should not crash discover_dupire."""
    K_grid = np.linspace(80.0, 120.0, 5, dtype=np.float64)
    tau_grid = np.linspace(0.1, 1.0, 5, dtype=np.float64)
    S0, r, sigma = 100.0, 0.05, 0.25

    C = np.zeros((len(K_grid), len(tau_grid)), dtype=np.float64)
    for j, tau in enumerate(tau_grid):
        C[:, j] = bs_call_price(S0, K_grid, r, sigma, float(tau))

    # With smoothing disabled the savgol window would otherwise exceed
    # the array, but the module also clamps it -- either path must work.
    result = discover_dupire(C, K_grid, tau_grid, smooth=False)

    assert 'discovered_coefficients' in result
    assert len(result['discovered_coefficients']) == 5
    assert result['term_names'] == list(DUPIRE_TERM_NAMES)
    # Just sanity: a finite (possibly poor) R^2 and no exception thrown.
    assert np.isfinite(result['r2_score'])
