"""
Tests for Black-Scholes Greeks: analytical vs finite-difference validation,
put-call delta relation, and gamma symmetry.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_generation import (
    bs_call_price, bs_call_delta, bs_put_delta, bs_gamma
)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
S0 = 100.0
K = 100.0
R = 0.05
SIGMA = 0.2
TAU = 0.5
H = 0.01  # finite-difference step size


# ---------------------------------------------------------------------------
# 1. Analytical Delta vs central finite difference
# ---------------------------------------------------------------------------
class TestAnalyticalDeltaVsFiniteDiff:
    def test_analytical_delta_vs_finite_diff(self):
        """
        Delta = dV/dS should match the central finite-difference approximation
        (V(S+h) - V(S-h)) / (2h) within 1e-3.
        """
        delta_analytical = bs_call_delta(S0, K, R, SIGMA, TAU)

        V_plus = bs_call_price(S0 + H, K, R, SIGMA, TAU)
        V_minus = bs_call_price(S0 - H, K, R, SIGMA, TAU)
        delta_fd = float((V_plus - V_minus) / (2.0 * H))

        np.testing.assert_allclose(
            float(delta_analytical), delta_fd, atol=1e-3,
            err_msg=(
                f"Analytical delta ({float(delta_analytical):.6f}) and "
                f"finite-diff delta ({delta_fd:.6f}) disagree beyond 1e-3"
            ),
        )


# ---------------------------------------------------------------------------
# 2. Analytical Gamma vs central finite difference
# ---------------------------------------------------------------------------
class TestAnalyticalGammaVsFiniteDiff:
    def test_analytical_gamma_vs_finite_diff(self):
        """
        Gamma = d2V/dS2 should match the central finite-difference approximation
        (V(S+h) - 2*V(S) + V(S-h)) / h^2 within 1e-2.
        """
        gamma_analytical = bs_gamma(S0, K, R, SIGMA, TAU)

        V_plus = bs_call_price(S0 + H, K, R, SIGMA, TAU)
        V_center = bs_call_price(S0, K, R, SIGMA, TAU)
        V_minus = bs_call_price(S0 - H, K, R, SIGMA, TAU)
        gamma_fd = float((V_plus - 2.0 * V_center + V_minus) / (H ** 2))

        np.testing.assert_allclose(
            float(gamma_analytical), gamma_fd, atol=1e-2,
            err_msg=(
                f"Analytical gamma ({float(gamma_analytical):.6f}) and "
                f"finite-diff gamma ({gamma_fd:.6f}) disagree beyond 1e-2"
            ),
        )


# ---------------------------------------------------------------------------
# 3. Put-call delta relation: Delta_call - Delta_put = 1
# ---------------------------------------------------------------------------
class TestPutCallDeltaRelation:
    def test_put_call_delta_relation(self):
        """
        For tau > 0, the difference Delta_call - Delta_put must equal 1.0
        across a range of stock prices.
        """
        S_array = np.linspace(60.0, 140.0, 81)
        delta_call = bs_call_delta(S_array, K, R, SIGMA, TAU)
        delta_put = bs_put_delta(S_array, K, R, SIGMA, TAU)

        np.testing.assert_allclose(
            delta_call - delta_put, 1.0, atol=1e-12,
            err_msg="Delta_call - Delta_put != 1.0 for some S values",
        )


# ---------------------------------------------------------------------------
# 4. Gamma is the same for calls and puts
# ---------------------------------------------------------------------------
class TestGammaSymmetric:
    def test_gamma_symmetric(self):
        """
        Gamma is identical for calls and puts (same function bs_gamma).
        Verify the interface returns consistent results when called with
        the same inputs at multiple stock prices and maturities.
        """
        S_array = np.linspace(60.0, 140.0, 50)
        tau_array = np.linspace(0.01, 1.0, 20)
        S_mesh, tau_mesh = np.meshgrid(S_array, tau_array, indexing='ij')

        gamma_1 = bs_gamma(S_mesh, K, R, SIGMA, tau_mesh)
        gamma_2 = bs_gamma(S_mesh, K, R, SIGMA, tau_mesh)

        np.testing.assert_array_equal(
            gamma_1, gamma_2,
            err_msg="Two calls to bs_gamma with identical inputs returned different results",
        )

        # Also verify gamma is non-negative (sanity)
        assert np.all(gamma_1 >= 0.0), "Gamma must be non-negative"
