"""
Tests for src.data_generation -- Black-Scholes analytical pricing,
Greeks, surface generation, and noise injection.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_generation import (
    bs_call_price, bs_put_price, bs_call_delta, bs_put_delta,
    bs_gamma, generate_price_surface, add_noise
)

# ---------------------------------------------------------------------------
# Default parameters used across most tests
# ---------------------------------------------------------------------------
K = 100.0
R = 0.05
SIGMA = 0.2
T = 1.0


# ---------------------------------------------------------------------------
# 1. Call price at (near) maturity equals intrinsic value
# ---------------------------------------------------------------------------
class TestCallPriceAtMaturity:
    def test_call_price_at_maturity(self):
        S = np.linspace(50, 150, 201)
        tau = 0.001  # near maturity
        C = bs_call_price(S, K, R, SIGMA, tau)
        intrinsic = np.maximum(S - K, 0.0)
        np.testing.assert_allclose(C, intrinsic, atol=0.5)


# ---------------------------------------------------------------------------
# 2. Put-call parity: C - P = S - K * exp(-r * tau)
# ---------------------------------------------------------------------------
class TestPutCallParity:
    def test_put_call_parity(self):
        V_call, S_grid, t_grid = generate_price_surface(
            n_S=50, n_t=50, K=K, r=R, sigma=SIGMA, T=T, option_type='call'
        )
        V_put, _, _ = generate_price_surface(
            n_S=50, n_t=50, K=K, r=R, sigma=SIGMA, T=T, option_type='put'
        )
        S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')
        tau_mesh = T - t_mesh
        lhs = V_call - V_put
        rhs = S_mesh - K * np.exp(-R * tau_mesh)
        np.testing.assert_allclose(lhs, rhs, atol=1e-8)


# ---------------------------------------------------------------------------
# 3. Call price bounds: 0 <= C <= S  and  C >= max(S - K*exp(-r*tau), 0)
# ---------------------------------------------------------------------------
class TestCallPriceBounds:
    def test_call_price_bounds(self):
        S = np.linspace(50, 150, 100)
        tau = np.linspace(0.01, T, 50)
        S_mesh, tau_mesh = np.meshgrid(S, tau, indexing='ij')
        C = bs_call_price(S_mesh, K, R, SIGMA, tau_mesh)

        assert np.all(C >= 0.0), "Call prices must be non-negative"
        assert np.all(C <= S_mesh + 1e-12), "Call price must not exceed stock price"

        lower = np.maximum(S_mesh - K * np.exp(-R * tau_mesh), 0.0)
        assert np.all(C >= lower - 1e-12), "Call price violates lower bound"


# ---------------------------------------------------------------------------
# 4. Put price bounds: 0 <= P <= K*exp(-r*tau) and P >= max(K*exp(-r*tau)-S, 0)
# ---------------------------------------------------------------------------
class TestPutPriceBounds:
    def test_put_price_bounds(self):
        S = np.linspace(50, 150, 100)
        tau = np.linspace(0.01, T, 50)
        S_mesh, tau_mesh = np.meshgrid(S, tau, indexing='ij')
        P = bs_put_price(S_mesh, K, R, SIGMA, tau_mesh)
        Kexp = K * np.exp(-R * tau_mesh)

        assert np.all(P >= 0.0), "Put prices must be non-negative"
        assert np.all(P <= Kexp + 1e-12), "Put price must not exceed discounted strike"

        lower = np.maximum(Kexp - S_mesh, 0.0)
        assert np.all(P >= lower - 1e-12), "Put price violates lower bound"


# ---------------------------------------------------------------------------
# 5. Call delta in [0, 1]
# ---------------------------------------------------------------------------
class TestCallDeltaRange:
    def test_call_delta_range(self):
        S = np.linspace(50, 150, 100)
        tau = np.linspace(0.01, T, 50)
        S_mesh, tau_mesh = np.meshgrid(S, tau, indexing='ij')
        delta = bs_call_delta(S_mesh, K, R, SIGMA, tau_mesh)

        assert np.all(delta >= -1e-12), "Call delta must be >= 0"
        assert np.all(delta <= 1.0 + 1e-12), "Call delta must be <= 1"


# ---------------------------------------------------------------------------
# 6. Put delta in [-1, 0]
# ---------------------------------------------------------------------------
class TestPutDeltaRange:
    def test_put_delta_range(self):
        S = np.linspace(50, 150, 100)
        tau = np.linspace(0.01, T, 50)
        S_mesh, tau_mesh = np.meshgrid(S, tau, indexing='ij')
        delta = bs_put_delta(S_mesh, K, R, SIGMA, tau_mesh)

        assert np.all(delta >= -1.0 - 1e-12), "Put delta must be >= -1"
        assert np.all(delta <= 0.0 + 1e-12), "Put delta must be <= 0"


# ---------------------------------------------------------------------------
# 7. Gamma is non-negative everywhere
# ---------------------------------------------------------------------------
class TestGammaNonnegative:
    def test_gamma_nonnegative(self):
        S = np.linspace(50, 150, 100)
        tau = np.linspace(0.01, T, 50)
        S_mesh, tau_mesh = np.meshgrid(S, tau, indexing='ij')
        gamma = bs_gamma(S_mesh, K, R, SIGMA, tau_mesh)

        assert np.all(gamma >= -1e-12), "Gamma must be non-negative"


# ---------------------------------------------------------------------------
# 8. Surface shape is (n_S, n_t)
# ---------------------------------------------------------------------------
class TestSurfaceShape:
    @pytest.mark.parametrize("n_S,n_t", [(50, 50), (100, 100), (30, 70)])
    def test_surface_shape(self, n_S, n_t):
        V, S_grid, t_grid = generate_price_surface(
            n_S=n_S, n_t=n_t, K=K, r=R, sigma=SIGMA, T=T
        )
        assert V.shape == (n_S, n_t)
        assert S_grid.shape == (n_S,)
        assert t_grid.shape == (n_t,)


# ---------------------------------------------------------------------------
# 9. Noise preserves shape
# ---------------------------------------------------------------------------
class TestNoisePreservesShape:
    def test_noise_preserves_shape(self):
        V, _, _ = generate_price_surface(
            n_S=50, n_t=50, K=K, r=R, sigma=SIGMA, T=T
        )
        V_noisy = add_noise(V, noise_pct=0.01, seed=42)
        assert V_noisy.shape == V.shape


# ---------------------------------------------------------------------------
# 10. Noise magnitude: std(noisy - clean) ~ noise_pct * std(clean) within 10%
# ---------------------------------------------------------------------------
class TestNoiseMagnitude:
    def test_noise_magnitude(self):
        V, _, _ = generate_price_surface(
            n_S=200, n_t=200, K=K, r=R, sigma=SIGMA, T=T
        )
        noise_pct = 0.05
        V_noisy = add_noise(V, noise_pct=noise_pct, seed=123)

        diff = V_noisy - V
        actual_std = np.std(diff)
        expected_std = noise_pct * np.std(V)

        relative_error = abs(actual_std - expected_std) / expected_std
        assert relative_error < 0.10, (
            f"Noise std relative error {relative_error:.3f} exceeds 10% tolerance. "
            f"actual_std={actual_std:.6f}, expected_std={expected_std:.6f}"
        )


# ---------------------------------------------------------------------------
# 11. Invalid inputs raise ValueError
# ---------------------------------------------------------------------------
class TestInvalidInputs:
    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            bs_call_price(100.0, K, R, 0.0, 0.5)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            bs_call_price(100.0, K, R, -0.1, 0.5)

    def test_S_zero_raises(self):
        with pytest.raises(ValueError, match="Stock price"):
            bs_call_price(0.0, K, R, SIGMA, 0.5)

    def test_S_negative_raises(self):
        with pytest.raises(ValueError, match="Stock price"):
            bs_call_price(-10.0, K, R, SIGMA, 0.5)

    def test_tau_negative_raises(self):
        with pytest.raises(ValueError, match="[Tt]ime"):
            bs_call_price(100.0, K, R, SIGMA, -0.1)

    def test_sigma_zero_put_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            bs_put_price(100.0, K, R, 0.0, 0.5)

    def test_S_negative_delta_raises(self):
        with pytest.raises(ValueError, match="Stock price"):
            bs_call_delta(-5.0, K, R, SIGMA, 0.5)

    def test_sigma_negative_gamma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            bs_gamma(100.0, K, R, -0.2, 0.5)

    def test_sigma_zero_surface_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            generate_price_surface(sigma=0.0)
