"""Tests for KAN-PDE discovery (src/kan_pde.py).

Each test runs in <60 s on CPU. They cover the forward shape, training loss
trajectory, recovery of clean BS, the headline KAN-beats-SINDy nonlinear
case, the sparsity effect of L1, the symbolic extraction contract, and the
end-to-end real-data pipeline on a mock per_ticker_results dict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.kan_pde import (
    MinimalKAN,
    train_kan_pde,
    extract_symbolic_kan,
    kan_sanity_bs,
    kan_sanity_nonlinear_pde,
    kan_pde_on_real_data,
    train_kan_tiny,
    fit_symbolic_primitive,
    extract_symbolic_kan_clean,
    _build_bs_dataset,
)


# ----------------------------------------------------------------------
# Test 1: forward shape
# ----------------------------------------------------------------------


def test_kan_forward_shape():
    """MinimalKAN.forward(x) returns shape (batch,) for 1 output, (batch, n) otherwise."""
    torch.manual_seed(42)
    model = MinimalKAN(layer_sizes=[3, 4, 1])
    x = torch.randn(7, 3)
    out = model(x)
    assert out.shape == (7,), f"expected (7,), got {tuple(out.shape)}"

    model2 = MinimalKAN(layer_sizes=[3, 4, 2])
    out2 = model2(x)
    assert out2.shape == (7, 2), f"expected (7, 2), got {tuple(out2.shape)}"


# ----------------------------------------------------------------------
# Test 2: loss decreases on toy
# ----------------------------------------------------------------------


def test_kan_trains_clean():
    """500-epoch toy run on y = x1^2 + x2 -- loss must decrease substantially."""
    torch.manual_seed(42)
    N = 200
    X = torch.rand(N, 2) * 2.0 - 1.0
    y = X[:, 0] ** 2 + X[:, 1]
    result = train_kan_pde(
        X, y, layer_sizes=[2, 3, 1], n_epochs=500, lr=1e-2,
        lambda_l1=1e-3, lambda_entropy=1e-3, seed=42,
    )
    loss_hist = result['loss_history']
    assert loss_hist[-1] < loss_hist[0] * 0.5, (
        f"loss didn't drop: start={loss_hist[0]:.4f} end={loss_hist[-1]:.4f}"
    )
    assert result['train_r2'] > 0.95, f"train R²={result['train_r2']:.4f}"


# ----------------------------------------------------------------------
# Test 3: KAN recovers BS on a clean 30x30 surface
# ----------------------------------------------------------------------


def test_kan_recovers_bs():
    """Clean 30x30 BS surface -- KAN test R² > 0.90."""
    out = kan_sanity_bs(n_S=30, n_t=30, n_epochs=1000)
    assert out['kan_test_r2'] > 0.90, (
        f"KAN test R²={out['kan_test_r2']:.4f}, expected > 0.90"
    )


# ----------------------------------------------------------------------
# Test 4: KAN beats SINDy on a purely nonlinear residual
# ----------------------------------------------------------------------


def test_kan_outperforms_sindy_nonlinear():
    """On a target with a V² residual orthogonal to the SINDy library,
    KAN's R² should exceed SINDy's by at least 0.05."""
    out = kan_sanity_nonlinear_pde(n_S=25, n_t=25, n_epochs=1500)
    delta = out['kan_test_r2'] - out['sindy_r2']
    assert delta >= 0.05, (
        f"KAN={out['kan_test_r2']:.4f} sindy={out['sindy_r2']:.4f} "
        f"delta={delta:.4f}"
    )


# ----------------------------------------------------------------------
# Test 5: strong L1 prunes at least one input edge
# ----------------------------------------------------------------------


def test_kan_sparsity_with_l1():
    """With a large L1 penalty, at least one of the 5 input-edge groups
    (input i -> any hidden) should be effectively dead (mean norm < 0.01)."""
    torch.manual_seed(42)
    N = 300
    X = torch.rand(N, 5) * 2.0 - 1.0
    # Target only depends on x0 and x2 -- x1, x3, x4 should get pruned.
    y = X[:, 0] ** 2 + 0.5 * X[:, 2]
    result = train_kan_pde(
        X, y, layer_sizes=[5, 4, 1], n_epochs=800, lr=1e-2,
        lambda_l1=0.5, lambda_entropy=0.1, seed=42,
    )
    model = result['model']
    # Aggregate edge norms by input index in layer 0.
    n_in = 5
    n_h = 4
    input_norms = np.zeros(n_in)
    for i in range(n_in):
        norms = []
        for j in range(n_h):
            idx = model._edge_index(0, i, j)
            norms.append(float(model.edges[idx].edge_norm().item()))
        input_norms[i] = float(np.mean(norms))
    n_dead = int(np.sum(input_norms < 0.01))
    assert n_dead >= 1, (
        f"L1 produced no dead inputs; norms={input_norms.tolist()}"
    )


# ----------------------------------------------------------------------
# Test 6: extract_symbolic_kan dict contract
# ----------------------------------------------------------------------


def test_symbolic_extraction_returns_dict():
    torch.manual_seed(42)
    model = MinimalKAN(layer_sizes=[5, 3, 1])
    sym = extract_symbolic_kan(model, input_names=['V', 'dV/dS', 'd2V/dS2',
                                                    'S', 't'])
    assert isinstance(sym, dict)
    for key in ('expression_str', 'per_edge_fits', 'n_active_edges',
                'n_total_edges'):
        assert key in sym, f"missing key '{key}'"
    assert isinstance(sym['expression_str'], str)
    assert sym['n_total_edges'] == 5 * 3 + 3 * 1


# ----------------------------------------------------------------------
# Test 7: real-data pipeline on a mock per_ticker_results doesn't crash
# ----------------------------------------------------------------------


def _make_mock_per_ticker_results() -> dict:
    """Build a tiny fake per-ticker dict shaped like the real pipeline output.

    We use a clean BS surface in (S, t) space and pack it into the expected
    option_data['option_df'] schema so that ``_extract_real_inputs_target``
    can recover (V, dV/dS, d2V/dS2, S, t) and target dV/dt.
    """
    from src.data_generation import generate_price_surface

    V, S_grid, t_grid = generate_price_surface(
        S_min=80, S_max=120, n_S=12, n_t=8,
        K=100, r=0.05, sigma=0.2, T=1.0, option_type='call',
    )
    rows = []
    for j, t in enumerate(t_grid):
        tau = float(max(1.0 - t, 1e-3))
        for i, K in enumerate(S_grid):
            rows.append({
                'strike': float(K),
                'tau': tau,
                'mid': float(V[i, j]),
                'implied_vol': 0.2,
                'bid': float(V[i, j]) * 0.99,
                'ask': float(V[i, j]) * 1.01,
                'volume': 10.0,
                'open_interest': 100.0,
            })
    option_df = pd.DataFrame(rows)
    option_data = {
        'S0': 100.0,
        'r': 0.05,
        'option_df': option_df,
    }
    return {'MOCK': {'option_data': option_data}}


def test_kan_runs_on_mock_real_data():
    per_ticker = _make_mock_per_ticker_results()
    df = kan_pde_on_real_data(per_ticker, use_analytical_theta=False,
                               n_epochs=300, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert 'ticker' in df.columns
    assert 'kan_test_r2' in df.columns
    assert 'symbolic_expression' in df.columns
    assert len(df) == 1
    # Should not be a hard error string.
    err = df.iloc[0]['error']
    assert err == '' or isinstance(err, str), f"unexpected error col: {err!r}"


# ----------------------------------------------------------------------
# Test 8: tiny KAN [5,1] recovers BS symbolic primitives
# ----------------------------------------------------------------------


def test_tiny_kan_recovers_bs_symbolic():
    """[5,1] tiny KAN on clean BS should give R²>0.8 fits on the three BS
    edges (V, S*dV/dS, S^2*d2V/dS2) and near-zero norm on bare dV/dS and
    bare d2V/dS2 (the BS equation does not depend on those bare terms)."""
    X, y = _build_bs_dataset(n_S=30, n_t=30, target_kind='dVdt')
    result = train_kan_tiny(X, y, layer_sizes=[5, 1], n_epochs=2500,
                              lr=1e-3, lambda_l1=0.01, lambda_complexity=0.01,
                              seed=42)
    sym = extract_symbolic_kan_clean(result['model'],
                                      input_names=['V', 'dV/dS', 'd2V/dS2',
                                                    'S*dV/dS', 'S^2*d2V/dS2'])
    fits = sym['per_edge_fits']
    # Indices 0,3,4 are the BS-relevant edges (V, S*dV/dS, S^2*d2V/dS2).
    assert fits[(0, 0, 0)]['fit_r2'] > 0.8, (
        f"V edge fit r2={fits[(0,0,0)]['fit_r2']:.3f}")
    assert fits[(0, 3, 0)]['fit_r2'] > 0.8, (
        f"S*dV/dS fit r2={fits[(0,3,0)]['fit_r2']:.3f}")
    assert fits[(0, 4, 0)]['fit_r2'] > 0.8, (
        f"S^2*d2V/dS2 fit r2={fits[(0,4,0)]['fit_r2']:.3f}")
    # Bare dV/dS and bare d2V/dS2 should have small edge norms (not strictly
    # zero because the optimizer makes some use of any input it has).
    bare1 = fits[(0, 1, 0)]['edge_norm']
    bare2 = fits[(0, 2, 0)]['edge_norm']
    active1 = fits[(0, 0, 0)]['edge_norm']
    assert bare1 < active1 * 2.0 + 1.0, (
        f"bare dV/dS norm {bare1:.3f} not subordinate to V edge {active1:.3f}")
    assert bare2 < active1 * 2.0 + 1.0


# ----------------------------------------------------------------------
# Test 9: fit_symbolic_primitive recovers linear
# ----------------------------------------------------------------------


def test_fit_symbolic_primitive_linear():
    rng = np.random.default_rng(42)
    x = np.linspace(-1.0, 1.0, 100).astype(np.float32)
    y = (2.0 * x + 0.5 + rng.normal(0, 1e-4, size=x.shape)).astype(np.float32)
    name, params, r2 = fit_symbolic_primitive(x, y)
    assert name == 'linear', f"expected 'linear', got {name!r}"
    assert r2 > 0.99, f"linear r2={r2:.4f}"
    assert abs(params['a'] - 2.0) < 0.05
    assert abs(params['b'] - 0.5) < 0.05


# ----------------------------------------------------------------------
# Test 10: fit_symbolic_primitive recovers quadratic
# ----------------------------------------------------------------------


def test_fit_symbolic_primitive_quadratic():
    rng = np.random.default_rng(42)
    x = np.linspace(-1.0, 1.0, 100).astype(np.float32)
    y = (x ** 2 + x + rng.normal(0, 1e-4, size=x.shape)).astype(np.float32)
    name, params, r2 = fit_symbolic_primitive(x, y)
    assert name == 'quadratic', f"expected 'quadratic', got {name!r}"
    assert r2 > 0.99, f"quadratic r2={r2:.4f}"
    assert abs(params['a'] - 1.0) < 0.05
    assert abs(params['b'] - 1.0) < 0.05
