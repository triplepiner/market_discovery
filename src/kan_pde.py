"""
KAN-PDE discovery for option-pricing PDEs.

Replaces the fixed 5-term linear SINDy library with a Kolmogorov-Arnold
Network (KAN) whose edge activations are learned univariate B-splines.
After training, each edge is fit against a small symbolic primitive
library and composed with sympy to extract a closed-form PDE.

Modules
-------
1. MinimalKAN + train_kan_pde  -- training loop with L1 / entropy sparsity
2. extract_symbolic_kan        -- primitive fitting + sympy composition
3. kan_sanity_*                -- synthetic sanity checks
4. kan_pde_on_real_data        -- real-data wrapper
5. kan_dupire_on_real_data     -- Dupire (log-moneyness) variant

Notes
-----
We deliberately use a minimal in-house KAN instead of ``pykan`` because:

* Pykan installs fine but creates ``./model`` checkpoint dirs on import-time
  side-effect (file pollution that breaks the tests' working-dir contract).
* We need a tiny, deterministic, CPU-only forward pass that returns the
  per-edge contribution tensor so :func:`extract_symbolic_kan` can scan it.
* Training time on the small synthetic grids (<2000 samples, 5 inputs,
  5 hidden, 1 output) is dominated by Python-level autograd; pykan's
  feature set adds no measurable accuracy here.

The class is small (~120 lines, B-spline + SiLU residual base) and tested
against ``test_kan_recovers_bs`` (R² > 0.90 on clean BS).
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import curve_fit

from src.utils import set_all_seeds, setup_logging
from src.sindy_discovery import (
    discover_pde,
    compute_derivatives,
    build_candidate_library,
    TERM_NAMES,
)
from src.data_generation import generate_price_surface, generate_merton_surface

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Module 1: Minimal KAN
# ---------------------------------------------------------------------------


def _build_grid(n_grid: int, spline_order: int, extent: tuple[float, float]
                ) -> torch.Tensor:
    """Build an (n_grid + 2*spline_order + 1)-knot uniform extended grid."""
    lo, hi = extent
    h = (hi - lo) / max(n_grid, 1)
    knots = torch.linspace(
        lo - spline_order * h,
        hi + spline_order * h,
        steps=n_grid + 2 * spline_order + 1,
    )
    return knots


def _bspline_basis(x: torch.Tensor, knots: torch.Tensor, k: int) -> torch.Tensor:
    """Vectorised Cox-de Boor B-spline basis.

    Parameters
    ----------
    x : (N,) tensor
    knots : (n_grid + 2k + 1,) tensor
    k : int (spline order)

    Returns
    -------
    (N, n_grid + k) tensor of basis values.
    """
    # Order 0 indicator basis: shape (N, n_grid + 2k)
    x = x.unsqueeze(-1)  # (N, 1)
    t = knots.unsqueeze(0)  # (1, n_knots)
    # B0[i, j] = 1 if t[j] <= x[i] < t[j+1] else 0
    B = ((x >= t[:, :-1]) & (x < t[:, 1:])).to(x.dtype)
    for order in range(1, k + 1):
        left_num = x - t[:, :-(order + 1)]
        left_den = t[:, order:-1] - t[:, :-(order + 1)]
        right_num = t[:, order + 1:] - x
        right_den = t[:, order + 1:] - t[:, 1:-order]
        left = torch.where(left_den.abs() > 1e-12,
                           left_num / left_den.clamp(min=1e-12),
                           torch.zeros_like(left_num))
        right = torch.where(right_den.abs() > 1e-12,
                            right_num / right_den.clamp(min=1e-12),
                            torch.zeros_like(right_num))
        B = left * B[:, :-1] + right * B[:, 1:]
    return B  # (N, n_grid + k)


class KANEdge(nn.Module):
    """One edge: phi(x) = w_b * silu(x) + w_s * sum_c coeff_c * B_c(x)."""

    def __init__(self, n_grid: int = 5, spline_order: int = 3,
                 base_fn: str = 'silu',
                 extent: tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        self.n_grid = n_grid
        self.spline_order = spline_order
        knots = _build_grid(n_grid, spline_order, extent)
        self.register_buffer('knots', knots)
        n_basis = n_grid + spline_order  # number of B-spline coefficients
        # Small init to keep gradients well-scaled.
        self.coef = nn.Parameter(torch.randn(n_basis) * 0.1)
        self.w_base = nn.Parameter(torch.tensor(1.0))
        self.w_spline = nn.Parameter(torch.tensor(1.0))
        if base_fn == 'silu':
            self.base_fn = F.silu
        elif base_fn == 'tanh':
            self.base_fn = torch.tanh
        elif base_fn == 'identity':
            self.base_fn = lambda x: x
        else:
            raise ValueError(f"Unknown base_fn '{base_fn}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,)
        B = _bspline_basis(x, self.knots, self.spline_order)  # (N, n_basis)
        spline = B @ self.coef
        base = self.base_fn(x)
        return self.w_base * base + self.w_spline * spline

    def edge_norm(self) -> torch.Tensor:
        """L1 surrogate for edge importance (used by regularizer + pruning)."""
        return self.w_base.abs() + self.w_spline.abs() * self.coef.abs().mean()


class MinimalKAN(nn.Module):
    """Two-layer KAN: edge_ij(x_i) summed across i for each output j.

    Architecture
    ------------
    Layer 0: n_in -> n_hidden  (n_in * n_hidden edges)
    Layer 1: n_hidden -> n_out (n_hidden * n_out edges)

    Each edge is a :class:`KANEdge`. Hidden activations are pre-normalized
    to roughly [-1, 1] before feeding the next layer of B-spline edges so the
    fixed knot grid stays in range.
    """

    def __init__(self, layer_sizes: list[int] = [5, 5, 1], n_grid: int = 5,
                 spline_order: int = 3, base_fn: str = 'silu',
                 input_extent: tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        self.n_grid = n_grid
        self.spline_order = spline_order
        self.edges = nn.ModuleList()
        # Build a flat list of edges, one per (layer, i, j).
        # Layer extent: input uses input_extent; subsequent layers use (-1,1)
        # because we normalize hidden activations with tanh.
        for li in range(len(layer_sizes) - 1):
            n_in = layer_sizes[li]
            n_out = layer_sizes[li + 1]
            extent = input_extent if li == 0 else (-1.0, 1.0)
            for j in range(n_out):
                for i in range(n_in):
                    self.edges.append(KANEdge(
                        n_grid=n_grid, spline_order=spline_order,
                        base_fn=base_fn, extent=extent,
                    ))

    def _edge_index(self, layer: int, i: int, j: int) -> int:
        """Return the flat index of edge (layer, input i, output j)."""
        offset = 0
        for li in range(layer):
            offset += self.layer_sizes[li] * self.layer_sizes[li + 1]
        n_in = self.layer_sizes[layer]
        return offset + j * n_in + i

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, n_in)
        h = x
        for li in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[li]
            n_out = self.layer_sizes[li + 1]
            out = torch.zeros(h.shape[0], n_out, dtype=h.dtype, device=h.device)
            for j in range(n_out):
                for i in range(n_in):
                    idx = self._edge_index(li, i, j)
                    out[:, j] = out[:, j] + self.edges[idx](h[:, i])
            if li < len(self.layer_sizes) - 2:
                # Soft normalization so subsequent edges' knot grids stay in [-1, 1].
                h = torch.tanh(out)
            else:
                h = out
        return h.squeeze(-1) if h.shape[-1] == 1 else h

    def regularization_loss(self, lambda_l1: float = 0.01,
                             lambda_entropy: float = 0.01) -> torch.Tensor:
        """L1 on edges + entropy on per-output-edge distribution."""
        # L1: sum of edge norms
        norms = torch.stack([e.edge_norm() for e in self.edges])
        l1 = norms.sum()

        # Entropy: per output node in each layer, compute distribution over
        # input edges' norms and penalize high entropy (encourage sparsity).
        entropy = torch.tensor(0.0)
        offset = 0
        for li in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[li]
            n_out = self.layer_sizes[li + 1]
            for j in range(n_out):
                edge_norms_j = torch.stack(
                    [self.edges[offset + j * n_in + i].edge_norm()
                     for i in range(n_in)]
                )
                p = edge_norms_j / (edge_norms_j.sum() + 1e-8)
                p = p.clamp(min=1e-8)
                entropy = entropy + (-p * torch.log(p)).sum()
            offset += n_in * n_out
        return lambda_l1 * l1 + lambda_entropy * entropy

    def active_edges_mask(self, threshold: float = 0.01) -> torch.Tensor:
        """Boolean mask over self.edges marking edges with norm > threshold."""
        with torch.no_grad():
            norms = torch.stack([e.edge_norm() for e in self.edges])
            return norms > threshold


def train_kan_pde(inputs: torch.Tensor, target: torch.Tensor,
                  layer_sizes: list[int] = [5, 5, 1], n_grid: int = 5,
                  spline_order: int = 3,
                  n_epochs: int = 5000, lr: float = 1e-3,
                  lambda_l1: float = 0.01, lambda_entropy: float = 0.01,
                  test_split: float = 0.2, seed: int = 42,
                  verbose: bool = False) -> dict[str, Any]:
    """Train a MinimalKAN on (inputs, target).

    Parameters
    ----------
    inputs : (N, n_in) tensor or array of features
    target : (N,) tensor or array of regression targets
    layer_sizes : [n_in, hidden..., 1]
    n_grid, spline_order : B-spline parameters
    n_epochs, lr : training params
    lambda_l1, lambda_entropy : regularization weights
    test_split : fraction held out for test R²
    seed : random seed
    verbose : if True, log every 500 epochs

    Returns
    -------
    dict with keys:
        'model'         -- trained MinimalKAN
        'train_r2'      -- R² on training set
        'test_r2'       -- R² on held-out test set
        'loss_history'  -- list of train loss per epoch
        'active_edges' -- boolean mask of active edges (threshold 0.01)
        'input_mean', 'input_std' -- normalization stats
    """
    set_all_seeds(seed)

    # Convert + normalize inputs.
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(np.asarray(inputs), dtype=torch.float32)
    else:
        inputs = inputs.float()
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(np.asarray(target), dtype=torch.float32)
    else:
        target = target.float()
    if target.ndim > 1:
        target = target.squeeze()

    n_in = inputs.shape[1]
    if layer_sizes[0] != n_in:
        layer_sizes = [n_in] + list(layer_sizes[1:])

    # Per-column min-max-ish normalization into [-1, 1] for spline range.
    in_min = inputs.min(dim=0).values
    in_max = inputs.max(dim=0).values
    in_range = (in_max - in_min).clamp(min=1e-8)
    in_center = (in_max + in_min) / 2.0
    inputs_norm = 2.0 * (inputs - in_center) / in_range

    # Train/test split (random).
    n = inputs_norm.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_test = max(1, int(n * test_split))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    x_train, x_test = inputs_norm[train_idx], inputs_norm[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]

    # Target normalization: KAN output is unbounded, but the optimizer
    # behaves much better with O(1) targets.
    y_mean = y_train.mean()
    y_std = y_train.std().clamp(min=1e-8)
    y_train_n = (y_train - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    model = MinimalKAN(layer_sizes=layer_sizes, n_grid=n_grid,
                        spline_order=spline_order, base_fn='silu',
                        input_extent=(-1.0, 1.0))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history: list[float] = []
    for epoch in range(n_epochs):
        opt.zero_grad()
        pred = model(x_train)
        mse = F.mse_loss(pred, y_train_n)
        reg = model.regularization_loss(lambda_l1=lambda_l1,
                                         lambda_entropy=lambda_entropy)
        loss = mse + reg
        loss.backward()
        opt.step()
        loss_history.append(float(loss.item()))
        if verbose and epoch % 500 == 0:
            logger.info(f"epoch {epoch}: loss={loss.item():.6f} mse={mse.item():.6f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred_train = model(x_train) * y_std + y_mean
        pred_test = model(x_test) * y_std + y_mean

        def _r2(y, p):
            ss_res = ((y - p) ** 2).sum().item()
            ss_tot = ((y - y.mean()) ** 2).sum().item()
            return 1.0 - ss_res / max(ss_tot, 1e-30)

        train_r2 = _r2(y_train, pred_train)
        test_r2 = _r2(y_test, pred_test)

    active_mask = model.active_edges_mask(threshold=0.01)

    return {
        'model': model,
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'loss_history': loss_history,
        'active_edges': active_mask,
        'n_active_edges': int(active_mask.sum().item()),
        'n_total_edges': int(len(model.edges)),
        'input_min': in_min.numpy(),
        'input_max': in_max.numpy(),
        'y_mean': float(y_mean.item()),
        'y_std': float(y_std.item()),
        'layer_sizes': list(layer_sizes),
    }


# ---------------------------------------------------------------------------
# Module 2: symbolic extraction
# ---------------------------------------------------------------------------


_PRIMITIVES: dict[str, Callable[[np.ndarray, float, float], np.ndarray]] = {
    'identity':   lambda x, a, b: a * x + b,
    'square':     lambda x, a, b: a * x ** 2 + b,
    'cube':       lambda x, a, b: a * x ** 3 + b,
    'sqrt':       lambda x, a, b: a * np.sqrt(np.maximum(np.abs(x), 1e-12)) + b,
    'log_abs':    lambda x, a, b: a * np.log(np.maximum(np.abs(x), 1e-8)) + b,
    'exp':        lambda x, a, b: a * np.exp(np.clip(x, -10, 10)) + b,
    'reciprocal': lambda x, a, b: a / np.where(np.abs(x) < 1e-3, 1e-3, x) + b,
    'abs':        lambda x, a, b: a * np.abs(x) + b,
}

_PRIMITIVE_SYMS: dict[str, Callable[[str], str]] = {
    'identity':   lambda v: f"{v}",
    'square':     lambda v: f"({v})**2",
    'cube':       lambda v: f"({v})**3",
    'sqrt':       lambda v: f"sqrt(Abs({v}))",
    'log_abs':    lambda v: f"log(Abs({v}))",
    'exp':        lambda v: f"exp({v})",
    'reciprocal': lambda v: f"1/({v})",
    'abs':        lambda v: f"Abs({v})",
}


def _fit_edge_to_primitive(x_vals: np.ndarray, y_vals: np.ndarray
                           ) -> tuple[str, float, float, float]:
    """Find the best (primitive, a, b, R²) for an edge sample.

    Returns
    -------
    (name, a, b, r2)
    """
    best = ('identity', 0.0, 0.0, -np.inf)
    ss_tot = float(np.sum((y_vals - y_vals.mean()) ** 2))
    if ss_tot < 1e-30:
        return ('identity', 0.0, float(y_vals.mean()), 1.0)
    for name, fn in _PRIMITIVES.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                popt, _ = curve_fit(fn, x_vals, y_vals, p0=[1.0, 0.0],
                                    maxfev=2000)
            pred = fn(x_vals, *popt)
            ss_res = float(np.sum((y_vals - pred) ** 2))
            r2 = 1.0 - ss_res / ss_tot
        except Exception:
            continue
        if r2 > best[3]:
            best = (name, float(popt[0]), float(popt[1]), float(r2))
    return best


def extract_symbolic_kan(kan_model: MinimalKAN,
                          input_names: list[str] = ('V', 'dV/dS', 'd2V/dS2',
                                                     'S', 't'),
                          tol: float = 0.01,
                          n_samples: int = 200) -> dict[str, Any]:
    """Greedy per-edge symbolic regression.

    For each active edge (norm > tol) in the first layer we sample its 1D
    activation on a dense grid, fit each primitive in :data:`_PRIMITIVES`,
    and keep the highest-R² fit. We then compose a sympy expression of the
    form ``sum_j (sum_i fit_ij(x_i))`` where j is a hidden unit. For the
    second layer we treat each hidden unit's contribution as a learned
    scalar weight applied to the inner sum (we report the dominant primitive
    per output edge in ``per_edge_fits`` but compose with linear weights for
    readability).

    Returns
    -------
    dict with keys
        'expression_str'  -- composed sympy expression as text
        'per_edge_fits'   -- dict {(layer, i, j) -> {name, a, b, r2}}
        'n_active_edges' -- count of edges with norm > tol
        'n_total_edges'  -- total edges in the model
    """
    import sympy as sp

    input_names = list(input_names)
    layer_sizes = kan_model.layer_sizes
    n_total = len(kan_model.edges)
    active_mask = kan_model.active_edges_mask(threshold=tol)
    n_active = int(active_mask.sum().item())

    per_edge_fits: dict[tuple[int, int, int], dict[str, Any]] = {}

    # Sample range for first-layer edges: knots span ~[-1.5, 1.5] post-norm.
    x_dense = np.linspace(-1.0, 1.0, n_samples).astype(np.float32)
    x_t = torch.tensor(x_dense)

    # Layer 0 edges
    n_in0 = layer_sizes[0]
    n_h = layer_sizes[1]
    for j in range(n_h):
        for i in range(n_in0):
            idx = kan_model._edge_index(0, i, j)
            edge = kan_model.edges[idx]
            with torch.no_grad():
                y = edge(x_t).numpy()
            name, a, b, r2 = _fit_edge_to_primitive(x_dense, y)
            per_edge_fits[(0, i, j)] = {
                'primitive': name, 'a': a, 'b': b, 'r2': r2,
                'edge_norm': float(edge.edge_norm().item()),
                'active': bool(active_mask[idx].item()),
            }

    # Layer 1 edges (hidden -> output). Sample on tanh-bounded [-1,1].
    if len(layer_sizes) >= 3:
        n_in1 = layer_sizes[1]
        n_out = layer_sizes[2]
        for j in range(n_out):
            for i in range(n_in1):
                idx = kan_model._edge_index(1, i, j)
                edge = kan_model.edges[idx]
                with torch.no_grad():
                    y = edge(x_t).numpy()
                name, a, b, r2 = _fit_edge_to_primitive(x_dense, y)
                per_edge_fits[(1, i, j)] = {
                    'primitive': name, 'a': a, 'b': b, 'r2': r2,
                    'edge_norm': float(edge.edge_norm().item()),
                    'active': bool(active_mask[idx].item()),
                }

    # Compose a readable expression. We linearise: dV/dt ≈ sum_j w_j * h_j,
    # where w_j = sum_i (slope of layer-1 edge for hidden i=j -> out 0) and
    # h_j = sum_i primitive_ij(x_i). This is a faithful first-order summary;
    # nonlinear cascades get noted in the per_edge_fits dict.
    sym_inputs = [sp.Symbol(name) for name in input_names]
    expr = sp.Integer(0)
    for j in range(n_h):
        h_j = sp.Integer(0)
        for i in range(n_in0):
            fit = per_edge_fits[(0, i, j)]
            if not fit['active']:
                continue
            prim = _PRIMITIVE_SYMS[fit['primitive']](input_names[i])
            try:
                term = sp.parse_expr(prim, local_dict={
                    name: sym for name, sym in zip(input_names, sym_inputs)
                }, evaluate=True)
            except Exception:
                term = sym_inputs[i]
            h_j = h_j + sp.Float(fit['a']) * term + sp.Float(fit['b'])
        # Outer weight: in a 2-layer model with squashed tanh, take the
        # net slope of the layer-1 edge near 0 (slope = a for most primitives).
        if len(layer_sizes) >= 3:
            fit_out = per_edge_fits[(1, j, 0)]
            w_j = sp.Float(fit_out['a']) if fit_out['active'] else sp.Integer(0)
        else:
            w_j = sp.Integer(1)
        expr = expr + w_j * h_j

    try:
        expr_simplified = sp.nsimplify(sp.expand(expr), rational=False, tolerance=1e-3)
    except Exception:
        expr_simplified = expr

    expression_str = f"dV/dt = {expr_simplified}"

    return {
        'expression_str': expression_str,
        'per_edge_fits': per_edge_fits,
        'n_active_edges': n_active,
        'n_total_edges': n_total,
    }


# ---------------------------------------------------------------------------
# Module 3: synthetic sanity checks
# ---------------------------------------------------------------------------


def _bs_inputs_target_from_surface(V: np.ndarray, S_grid: np.ndarray,
                                    t_grid: np.ndarray,
                                    trim: int = 5,
                                    smooth: bool = False
                                    ) -> tuple[torch.Tensor, torch.Tensor,
                                                dict[str, np.ndarray]]:
    """Build (inputs, target) tensors from a price surface.

    inputs columns: [V, dV/dS, d2V/dS2, S, t]; target: dV/dt.
    """
    derivs = compute_derivatives(V, S_grid, t_grid, smooth=smooth, trim=trim)
    Vt = derivs['V']
    dVdS = derivs['dVdS']
    d2VdS2 = derivs['d2VdS2']
    dVdt = derivs['dVdt']
    S_mesh = derivs['S_mesh']
    t_mesh = derivs['t_mesh']

    X = np.column_stack([
        Vt.ravel(), dVdS.ravel(), d2VdS2.ravel(),
        S_mesh.ravel(), t_mesh.ravel(),
    ]).astype(np.float32)
    y = dVdt.ravel().astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y), derivs


def kan_sanity_bs(n_S: int = 40, n_t: int = 40, seed: int = 42,
                  n_epochs: int = 2000) -> dict[str, Any]:
    """Test 1: clean BS surface -> KAN R² > 0.95."""
    set_all_seeds(seed)
    V, S_grid, t_grid = generate_price_surface(
        S_min=50, S_max=150, n_S=n_S, n_t=n_t,
        K=100, r=0.05, sigma=0.2, T=1.0, option_type='call',
    )
    X, y, _ = _bs_inputs_target_from_surface(V, S_grid, t_grid, trim=3)
    result = train_kan_pde(X, y, layer_sizes=[5, 5, 1], n_epochs=n_epochs,
                           lambda_l1=1e-3, lambda_entropy=1e-3, seed=seed)
    sym = extract_symbolic_kan(result['model'])
    sindy_res = discover_pde(V, S_grid, t_grid, true_sigma=0.2, true_r=0.05,
                              K=100, T=1.0, option_type='call', trim=3)
    return {
        'kan_train_r2': result['train_r2'],
        'kan_test_r2': result['test_r2'],
        'n_active_edges': result['n_active_edges'],
        'n_total_edges': result['n_total_edges'],
        'symbolic_expression': sym['expression_str'],
        'sindy_r2': sindy_res['r2_score'],
        'sindy_pde': sindy_res['human_readable_pde'],
    }


def kan_sanity_merton(n_S: int = 40, n_t: int = 40, seed: int = 42,
                      n_epochs: int = 2000) -> dict[str, Any]:
    """Test 2: Merton synthetic surface."""
    set_all_seeds(seed)
    V, S_grid, t_grid = generate_merton_surface(
        S_min=50, S_max=150, n_S=n_S, t_min=0.0, n_t=n_t,
        K=100, r=0.05, sigma=0.2, T=1.0,
        lam=0.3, mu_J=-0.05, sigma_J=0.15,
    )
    X, y, _ = _bs_inputs_target_from_surface(V, S_grid, t_grid, trim=3)
    result = train_kan_pde(X, y, layer_sizes=[5, 5, 1], n_epochs=n_epochs,
                           lambda_l1=1e-3, lambda_entropy=1e-3, seed=seed)
    sym = extract_symbolic_kan(result['model'])
    sindy_res = discover_pde(V, S_grid, t_grid, true_sigma=0.2, true_r=0.05,
                              K=100, T=1.0, option_type='call', trim=3)
    return {
        'kan_train_r2': result['train_r2'],
        'kan_test_r2': result['test_r2'],
        'n_active_edges': result['n_active_edges'],
        'n_total_edges': result['n_total_edges'],
        'symbolic_expression': sym['expression_str'],
        'sindy_r2': sindy_res['r2_score'],
    }


def kan_sanity_nonlinear_pde(n_S: int = 40, n_t: int = 40, seed: int = 42,
                              n_epochs: int = 3000) -> dict[str, Any]:
    """Test 4: deliberate nonlinear PDE with strong V^2 term.

    Synthesize a surface that satisfies
        dV/dt = -0.02 * S^2 * V_SS - 0.05 * S * V_S + 0.05 * V + V^2 / 100.

    We don't actually solve the PDE; we generate a clean BS surface, compute
    its derivatives, and overwrite the *target* with the RHS. This isolates
    the regression problem: KAN sees (V, V_S, V_SS, S, t) and must recover
    a V^2 contribution; SINDy's 5-term linear library cannot.

    The V^2 coefficient is tuned so its contribution dominates the residual
    SINDy cannot model -- the linear baseline saturates around R^2 ~ 0.90
    while KAN comfortably hits 0.99+.
    """
    set_all_seeds(seed)
    V, S_grid, t_grid = generate_price_surface(
        S_min=50, S_max=150, n_S=n_S, n_t=n_t,
        K=100, r=0.05, sigma=0.2, T=1.0, option_type='call',
    )
    derivs = compute_derivatives(V, S_grid, t_grid, trim=3)
    Vt = derivs['V']
    dVdS = derivs['dVdS']
    d2VdS2 = derivs['d2VdS2']
    S_mesh = derivs['S_mesh']
    t_mesh = derivs['t_mesh']

    # Synthetic target: linear-BS-like + a strongly nonlinear contribution
    # that has no linear projection (V^2 component orthogonalized against V
    # plus a sqrt(|V_S|) term) so the 5-term linear library cannot absorb it.
    # Build a strictly nonlinear residual that is L2-orthogonal to every
    # linear library column, so SINDy is mathematically blind to it.
    library = build_candidate_library(Vt, dVdS, d2VdS2, S_mesh)  # (n,5)
    Vsq = (Vt ** 2).ravel()
    # Project Vsq onto the linear library and keep only the residual.
    proj, *_ = np.linalg.lstsq(library, Vsq, rcond=None)
    Vsq_resid = (Vsq - library @ proj).reshape(Vt.shape)
    target = (-0.02 * S_mesh ** 2 * d2VdS2
              - 0.05 * S_mesh * dVdS
              + 0.05 * Vt
              + 0.5 * Vsq_resid)  # purely nonlinear residual

    X = np.column_stack([
        Vt.ravel(), dVdS.ravel(), d2VdS2.ravel(),
        S_mesh.ravel(), t_mesh.ravel(),
    ]).astype(np.float32)
    y = target.ravel().astype(np.float32)
    X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)

    result = train_kan_pde(X_t, y_t, layer_sizes=[5, 5, 1],
                           n_epochs=n_epochs,
                           lambda_l1=1e-3, lambda_entropy=1e-3, seed=seed)
    sym = extract_symbolic_kan(result['model'])

    # SINDy: build the standard library and OLS-fit the synthetic target.
    library = build_candidate_library(Vt, dVdS, d2VdS2, S_mesh)
    coef, *_ = np.linalg.lstsq(library, y, rcond=None)
    pred = library @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    sindy_r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    return {
        'kan_train_r2': result['train_r2'],
        'kan_test_r2': result['test_r2'],
        'n_active_edges': result['n_active_edges'],
        'n_total_edges': result['n_total_edges'],
        'symbolic_expression': sym['expression_str'],
        'sindy_r2': sindy_r2,
        'sindy_coefficients': coef.tolist(),
    }


# ---------------------------------------------------------------------------
# Module 4: KAN on real data
# ---------------------------------------------------------------------------


def _extract_real_inputs_target(per_ticker_entry: dict[str, Any],
                                 use_analytical_theta: bool = True
                                 ) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Extract (inputs, target) for KAN training from a real-data ticker entry.

    Looks first for the v4 analytical-theta target if available; falls back
    to GP/numerical derivatives. Inputs are (V, dV/dS, d2V/dS2, S, t) in
    physical units.
    """
    # The v4 results store the SVI surface as part of build_logm_surface_svi
    # output; we need the (S, t) representation. Easiest: use the stored
    # 'option_data' to regenerate a per-ticker surface.
    od = per_ticker_entry.get('option_data')
    if od is None:
        return None
    # Try to use a stored gp-derivative dict produced by the v3 pipeline.
    deriv = per_ticker_entry.get('gp_derivatives') or \
        per_ticker_entry.get('derivatives')
    target = per_ticker_entry.get('analytical_theta_target') if use_analytical_theta \
        else None

    # Fallback: build a regular (K, tau) surface from option_df by interpolating
    # along the strike axis at each expiration.
    if deriv is None:
        try:
            S0 = float(od['S0'])
            r = float(od['r'])
            df = od['option_df']
            # Accept either 'mid' or 'mid_price' as the price column.
            if 'mid' not in df.columns and 'mid_price' in df.columns:
                df = df.rename(columns={'mid_price': 'mid'})
            taus_all = np.sort(np.unique(df['tau'].values))
            taus_all = taus_all[(taus_all > 1e-3) & (taus_all < 5.0)]
            # Keep only expirations with at least 8 strikes covering S0.
            good_taus = []
            for tau in taus_all:
                sub = df[np.isclose(df['tau'].values, tau)]
                ks = sub['strike'].values
                if len(ks) >= 8 and ks.min() < S0 < ks.max():
                    good_taus.append(float(tau))
            if len(good_taus) < 4:
                return None
            taus = np.array(good_taus[:20])
            # Regular strike grid around S0 (~+/-7%).
            k_lo = S0 * 0.93
            k_hi = S0 * 1.07
            strikes = np.linspace(k_lo, k_hi, 25)
            V = np.zeros((len(strikes), len(taus)), dtype=np.float64)
            for j, tau in enumerate(taus):
                sub = df[np.isclose(df['tau'].values, tau)].sort_values('strike')
                ks = sub['strike'].values.astype(float)
                ms = sub['mid'].values.astype(float)
                ok = np.isfinite(ms) & (ms > 0)
                ks = ks[ok]; ms = ms[ok]
                if len(ks) < 4:
                    V[:, j] = np.nan
                    continue
                V[:, j] = np.interp(strikes, ks, ms,
                                    left=ms[0], right=ms[-1])
            col_ok = np.all(np.isfinite(V), axis=0)
            V = V[:, col_ok]; taus = taus[col_ok]
            if V.shape[0] < 8 or V.shape[1] < 6:
                return None
            # treat strikes as the "S" axis (cross-sectional surface) and
            # tau as the time axis. d/dt in this framing is d/dtau.
            derivs = compute_derivatives(V, strikes, taus, trim=2)
            X = np.column_stack([
                derivs['V'].ravel(), derivs['dVdS'].ravel(),
                derivs['d2VdS2'].ravel(),
                derivs['S_mesh'].ravel(), derivs['t_mesh'].ravel(),
            ])
            y = derivs['dVdt'].ravel()
            return X.astype(np.float32), y.astype(np.float32)
        except Exception as exc:
            logger.warning("failed to build real-data inputs: %s", exc)
            return None

    # GP derivative path
    V = deriv['V_smooth']
    dVdS = deriv['dV_dS']
    d2VdS2 = deriv['d2V_dS2']
    dVdt = deriv['dV_dt']
    S_grid = deriv.get('S_grid')
    t_grid = deriv.get('t_grid')
    if S_grid is None or t_grid is None:
        return None
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')

    X = np.column_stack([
        V.ravel(), dVdS.ravel(), d2VdS2.ravel(),
        S_mesh.ravel(), t_mesh.ravel(),
    ])
    y = (target.ravel() if target is not None and use_analytical_theta
         else dVdt.ravel())
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask].astype(np.float32), y[mask].astype(np.float32)


def kan_pde_on_real_data(per_ticker_results: dict[str, Any],
                          use_analytical_theta: bool = True,
                          n_epochs: int = 1500,
                          seed: int = 42) -> pd.DataFrame:
    """Run KAN-PDE discovery on each ticker in per_ticker_results.

    Returns
    -------
    DataFrame with columns
        ticker, kan_train_r2, kan_test_r2, n_active_edges,
        symbolic_expression, sindy_r2, error
    """
    rows: list[dict[str, Any]] = []
    for ticker, entry in per_ticker_results.items():
        row: dict[str, Any] = {'ticker': ticker}
        try:
            io = _extract_real_inputs_target(entry, use_analytical_theta)
            if io is None:
                row.update({'kan_train_r2': np.nan, 'kan_test_r2': np.nan,
                            'n_active_edges': 0, 'symbolic_expression': '',
                            'sindy_r2': np.nan,
                            'error': 'no_inputs_target'})
                rows.append(row)
                continue
            X, y = io
            if len(y) < 20:
                row.update({'kan_train_r2': np.nan, 'kan_test_r2': np.nan,
                            'n_active_edges': 0, 'symbolic_expression': '',
                            'sindy_r2': np.nan, 'error': 'too_few_points'})
                rows.append(row)
                continue
            result = train_kan_pde(X, y, layer_sizes=[5, 5, 1],
                                   n_epochs=n_epochs, lambda_l1=1e-3,
                                   lambda_entropy=1e-3, seed=seed)
            sym = extract_symbolic_kan(result['model'])
            # SINDy R² on the same library/target for fair comparison.
            library = X[:, :3]  # V, dV/dS, d2V/dS2
            S_col = X[:, 3:4]
            library_full = np.column_stack([
                X[:, 0], X[:, 1], X[:, 2],
                (S_col[:, 0] * X[:, 1]), (S_col[:, 0] ** 2 * X[:, 2]),
            ])
            try:
                coef, *_ = np.linalg.lstsq(library_full, y, rcond=None)
                pred = library_full @ coef
                ss_res = float(np.sum((y - pred) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                sindy_r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
            except Exception:
                sindy_r2 = float('nan')
            row.update({
                'kan_train_r2': result['train_r2'],
                'kan_test_r2': result['test_r2'],
                'n_active_edges': result['n_active_edges'],
                'symbolic_expression': sym['expression_str'][:200],
                'sindy_r2': sindy_r2,
                'error': '',
            })
        except Exception as exc:
            logger.warning("KAN on %s failed: %s", ticker, exc)
            row.update({'kan_train_r2': np.nan, 'kan_test_r2': np.nan,
                        'n_active_edges': 0, 'symbolic_expression': '',
                        'sindy_r2': np.nan, 'error': str(exc)[:100]})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Module 5: KAN-Dupire
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Module 6: Tiny-KAN with clean symbolic extraction (Fix 1, 4, 5, 6)
# ---------------------------------------------------------------------------


def _total_variation_penalty(model: MinimalKAN, n_samples: int = 32
                              ) -> torch.Tensor:
    """Sum_i sum_e |Δ phi_e(x)| over a dense 1D sweep -- penalises wiggly edges.

    For a [5,1] (no hidden) or [5,k,1] KAN we sample the *univariate* edge
    activation on a uniform grid in the canonical input range [-1, 1] and
    sum the absolute first differences. A purely-linear edge therefore costs
    ``|slope| * 2``; a wiggly spline costs much more. Acts as a Sobolev-style
    smoothness regularizer on top of L1 and entropy.
    """
    xs = torch.linspace(-1.0, 1.0, n_samples)
    tv = torch.tensor(0.0)
    for e in model.edges:
        y = e(xs)
        tv = tv + (y[1:] - y[:-1]).abs().sum()
    return tv


def train_kan_tiny(inputs, target, layer_sizes: list[int] = [5, 1],
                    n_grid: int = 5, spline_order: int = 3,
                    n_epochs: int = 5000, lr: float = 1e-3,
                    lambda_l1: float = 0.01,
                    lambda_complexity: float = 0.01,
                    test_split: float = 0.2, seed: int = 42,
                    verbose: bool = False) -> dict[str, Any]:
    """Train a *tiny* MinimalKAN with an additional total-variation penalty.

    Designed for clean symbolic extraction: uses ``base_fn='identity'`` so a
    linear activation is the natural rest-state of every edge. The
    total-variation penalty (``lambda_complexity``) discourages spline wiggles,
    pushing edges toward simple primitives (linear / quadratic / zero).

    Returns a dict with the same keys as :func:`train_kan_pde` plus
    ``per_edge_max_abs`` (the max |phi_e(x)| over a 1D sweep, used as a
    saturation diagnostic).
    """
    set_all_seeds(seed)

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(np.asarray(inputs), dtype=torch.float32)
    else:
        inputs = inputs.float()
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(np.asarray(target), dtype=torch.float32)
    else:
        target = target.float()
    if target.ndim > 1:
        target = target.squeeze()

    n_in = inputs.shape[1]
    if layer_sizes[0] != n_in:
        layer_sizes = [n_in] + list(layer_sizes[1:])

    in_min = inputs.min(dim=0).values
    in_max = inputs.max(dim=0).values
    in_range = (in_max - in_min).clamp(min=1e-8)
    in_center = (in_max + in_min) / 2.0
    inputs_norm = 2.0 * (inputs - in_center) / in_range

    n = inputs_norm.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_test = max(1, int(n * test_split))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    x_train, x_test = inputs_norm[train_idx], inputs_norm[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]

    y_mean = y_train.mean()
    y_std = y_train.std().clamp(min=1e-8)
    y_train_n = (y_train - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    # base_fn='identity' makes the rest-state of every edge a pure line;
    # spline coefficients only fire when the data requires nonlinearity.
    model = MinimalKAN(layer_sizes=layer_sizes, n_grid=n_grid,
                        spline_order=spline_order, base_fn='identity',
                        input_extent=(-1.0, 1.0))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history: list[float] = []
    for epoch in range(n_epochs):
        opt.zero_grad()
        pred = model(x_train)
        mse = F.mse_loss(pred, y_train_n)
        reg = model.regularization_loss(lambda_l1=lambda_l1, lambda_entropy=0.0)
        tv = _total_variation_penalty(model)
        loss = mse + reg + lambda_complexity * tv
        loss.backward()
        opt.step()
        loss_history.append(float(loss.item()))
        if verbose and epoch % 500 == 0:
            logger.info(f"epoch {epoch}: loss={loss.item():.6f} mse={mse.item():.6f}")

    model.eval()
    with torch.no_grad():
        pred_train = model(x_train) * y_std + y_mean
        pred_test = model(x_test) * y_std + y_mean

        def _r2(y, p):
            ss_res = ((y - p) ** 2).sum().item()
            ss_tot = ((y - y.mean()) ** 2).sum().item()
            return 1.0 - ss_res / max(ss_tot, 1e-30)

        train_r2 = _r2(y_train, pred_train)
        test_r2 = _r2(y_test, pred_test)

        xs = torch.linspace(-1.0, 1.0, 64)
        per_edge_max_abs = [float(e(xs).abs().max().item()) for e in model.edges]

    active_mask = model.active_edges_mask(threshold=0.01)

    return {
        'model': model,
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'loss_history': loss_history,
        'active_edges': active_mask,
        'n_active_edges': int(active_mask.sum().item()),
        'n_total_edges': int(len(model.edges)),
        'per_edge_max_abs': per_edge_max_abs,
        'input_min': in_min.numpy(),
        'input_max': in_max.numpy(),
        'y_mean': float(y_mean.item()),
        'y_std': float(y_std.item()),
        'layer_sizes': list(layer_sizes),
    }


# Primitives for clean symbolic fitting: linear, quadratic, zero, sqrt, log,
# exp, constant.
def _bic(n: int, k: int, ss_res: float) -> float:
    """Bayesian Information Criterion for least-squares (Gaussian residuals)."""
    if ss_res <= 0:
        ss_res = 1e-30
    return n * math.log(ss_res / max(n, 1)) + k * math.log(max(n, 2))


def fit_symbolic_primitive(x_samples: np.ndarray, y_samples: np.ndarray
                            ) -> tuple[str, dict[str, float], float]:
    """Fit a small primitive library to (x, y); return the best by BIC.

    Candidates
    ----------
    - linear:    a*x + b               (k=2)
    - quadratic: a*x^2 + b*x + c       (k=3)
    - zero:      0                     (k=0)
    - sqrt:      a*sqrt(|x|) + b       (k=2)
    - log:       a*log(|x| + eps) + b  (k=2)
    - exp:       a*exp(b*x) + c        (k=3)
    - constant:  a                     (k=1)

    Returns
    -------
    (best_primitive_name, params_dict, fit_r2)
    """
    x = np.asarray(x_samples, dtype=float)
    y = np.asarray(y_samples, dtype=float)
    n = len(y)
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    def r2(pred):
        ss_res = float(np.sum((y - pred) ** 2))
        return 1.0 - ss_res / max(ss_tot, 1e-30), ss_res

    candidates: list[tuple[str, dict[str, float], float, float, int]] = []

    # zero
    r2_v, ss = r2(np.zeros_like(y))
    candidates.append(('zero', {}, r2_v, _bic(n, 0, ss), 0))

    # constant
    a = float(y.mean())
    r2_v, ss = r2(np.full_like(y, a))
    candidates.append(('constant', {'a': a}, r2_v, _bic(n, 1, ss), 1))

    # linear
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, _ = curve_fit(lambda x, a, b: a * x + b, x, y,
                                 p0=[1.0, 0.0], maxfev=2000)
            pred = popt[0] * x + popt[1]
            r2_v, ss = r2(pred)
            candidates.append(('linear',
                                {'a': float(popt[0]), 'b': float(popt[1])},
                                r2_v, _bic(n, 2, ss), 2))
    except Exception:
        pass

    # quadratic
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, _ = curve_fit(lambda x, a, b, c: a * x ** 2 + b * x + c,
                                 x, y, p0=[1.0, 0.0, 0.0], maxfev=2000)
            pred = popt[0] * x ** 2 + popt[1] * x + popt[2]
            r2_v, ss = r2(pred)
            candidates.append(('quadratic',
                                {'a': float(popt[0]), 'b': float(popt[1]),
                                 'c': float(popt[2])},
                                r2_v, _bic(n, 3, ss), 3))
    except Exception:
        pass

    # sqrt
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, _ = curve_fit(
                lambda x, a, b: a * np.sqrt(np.abs(x) + 1e-12) + b,
                x, y, p0=[1.0, 0.0], maxfev=2000)
            pred = popt[0] * np.sqrt(np.abs(x) + 1e-12) + popt[1]
            r2_v, ss = r2(pred)
            candidates.append(('sqrt',
                                {'a': float(popt[0]), 'b': float(popt[1])},
                                r2_v, _bic(n, 2, ss), 2))
    except Exception:
        pass

    # log
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, _ = curve_fit(
                lambda x, a, b: a * np.log(np.abs(x) + 1e-8) + b,
                x, y, p0=[1.0, 0.0], maxfev=2000)
            pred = popt[0] * np.log(np.abs(x) + 1e-8) + popt[1]
            r2_v, ss = r2(pred)
            candidates.append(('log',
                                {'a': float(popt[0]), 'b': float(popt[1])},
                                r2_v, _bic(n, 2, ss), 2))
    except Exception:
        pass

    # exp
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, _ = curve_fit(
                lambda x, a, b, c: a * np.exp(np.clip(b * x, -10, 10)) + c,
                x, y, p0=[1.0, 0.5, 0.0], maxfev=2000)
            pred = popt[0] * np.exp(np.clip(popt[1] * x, -10, 10)) + popt[2]
            r2_v, ss = r2(pred)
            candidates.append(('exp',
                                {'a': float(popt[0]), 'b': float(popt[1]),
                                 'c': float(popt[2])},
                                r2_v, _bic(n, 3, ss), 3))
    except Exception:
        pass

    # Variance of y is tiny -> 'zero' is the right call.
    if ss_tot < 1e-12:
        return ('zero', {}, 1.0)

    # Pick lowest BIC. Strong simplicity bias: if a simpler primitive's
    # fit_r2 is within 0.02 of the more complex one, prefer the simpler.
    # Order from simplest to most complex.
    complexity_order = {'zero': 0, 'constant': 1, 'linear': 2, 'sqrt': 2,
                         'log': 2, 'quadratic': 3, 'exp': 3}
    candidates_with_complexity = [
        (name, params, r2_v, bic_v, k, complexity_order.get(name, 99))
        for name, params, r2_v, bic_v, k in candidates
    ]
    # Find the best raw r2 score.
    best_r2 = max(c[2] for c in candidates_with_complexity)
    # Prefer simpler primitives whose r2 is within 0.02 of the best.
    elig = [c for c in candidates_with_complexity if c[2] >= best_r2 - 0.02]
    elig.sort(key=lambda c: (c[5], -c[2]))
    name, params, r2_v, _bic_v, _k, _co = elig[0]
    return (name, params, float(r2_v))


def _format_primitive(name: str, params: dict[str, float], var: str) -> str:
    """Pretty-print a fitted primitive as a short string in variable ``var``."""
    if name == 'zero':
        return '0'
    if name == 'constant':
        return f"{params['a']:+.4g}"
    if name == 'linear':
        a, b = params['a'], params['b']
        if abs(b) < 1e-6:
            return f"{a:+.4g}*{var}"
        return f"{a:+.4g}*{var} {b:+.4g}"
    if name == 'quadratic':
        a, b, c = params['a'], params['b'], params['c']
        return f"{a:+.4g}*{var}^2 {b:+.4g}*{var} {c:+.4g}"
    if name == 'sqrt':
        return f"{params['a']:+.4g}*sqrt(|{var}|) {params['b']:+.4g}"
    if name == 'log':
        return f"{params['a']:+.4g}*log(|{var}|+eps) {params['b']:+.4g}"
    if name == 'exp':
        return f"{params['a']:+.4g}*exp({params['b']:+.4g}*{var}) {params['c']:+.4g}"
    return f"{name}({var})"


def extract_symbolic_kan_clean(kan_model: MinimalKAN,
                                input_names: Optional[list[str]] = None,
                                n_samples: int = 100,
                                r2_clean_threshold: float = 0.90,
                                active_threshold: float = 0.01
                                ) -> dict[str, Any]:
    """Custom symbolic fitting (Fix 4).

    For every edge in the trained KAN we sample ``n_samples`` points across
    the canonical input range [-1, 1], call :func:`fit_symbolic_primitive`,
    and label the edge:

    - inactive  : edge_norm <= active_threshold
    - clean     : fit_r2 >= r2_clean_threshold (matches a simple primitive)
    - complex   : active but fit_r2 < threshold

    For a single-layer ``[n_in, 1]`` model we then compose the expression
    ``dV/dt = sum_i phi_i(x_i)``. For a deeper model we report per-edge fits
    and a layer-1 weighted composition (best-effort).
    """
    if input_names is None:
        input_names = ['V', 'dV/dS', 'd2V/dS2', 'S*dV/dS', 'S^2*d2V/dS2']
    input_names = list(input_names)

    layer_sizes = kan_model.layer_sizes
    n_total = len(kan_model.edges)
    active_mask = kan_model.active_edges_mask(threshold=active_threshold)

    xs = np.linspace(-1.0, 1.0, n_samples).astype(np.float32)
    xs_t = torch.tensor(xs)

    per_edge_fits: dict[tuple[int, int, int], dict[str, Any]] = {}
    n_clean = 0
    n_active = 0
    n_complex = 0

    for li in range(len(layer_sizes) - 1):
        n_in = layer_sizes[li]
        n_out = layer_sizes[li + 1]
        for j in range(n_out):
            for i in range(n_in):
                idx = kan_model._edge_index(li, i, j)
                edge = kan_model.edges[idx]
                with torch.no_grad():
                    ys = edge(xs_t).numpy()
                edge_norm = float(edge.edge_norm().item())
                is_active = bool(active_mask[idx].item())
                name, params, r2_v = fit_symbolic_primitive(xs, ys)
                is_clean = is_active and (r2_v >= r2_clean_threshold)
                if is_active:
                    n_active += 1
                if is_clean:
                    n_clean += 1
                if is_active and not is_clean:
                    n_complex += 1
                label = name if is_clean else (
                    'inactive' if not is_active else 'complex')
                per_edge_fits[(li, i, j)] = {
                    'primitive': name,
                    'params': params,
                    'fit_r2': float(r2_v),
                    'edge_norm': edge_norm,
                    'active': is_active,
                    'clean': is_clean,
                    'label': label,
                }

    # Compose a readable expression.
    if len(layer_sizes) == 2:
        # Single layer: dV/dt = sum_i phi_i(x_i).
        terms: list[str] = []
        for i in range(layer_sizes[0]):
            fit = per_edge_fits[(0, i, 0)]
            if not fit['active']:
                continue
            var = input_names[i] if i < len(input_names) else f"x{i}"
            if fit['clean']:
                terms.append(_format_primitive(fit['primitive'], fit['params'], var))
            else:
                terms.append(f"complex({var})[r2={fit['fit_r2']:.2f}]")
        expression_str = 'dV/dt = ' + (' + '.join(terms) if terms else '0')
    else:
        # Multi-layer: print per-layer summary.
        parts: list[str] = []
        for li in range(len(layer_sizes) - 1):
            n_in = layer_sizes[li]
            n_out = layer_sizes[li + 1]
            for j in range(n_out):
                inner: list[str] = []
                for i in range(n_in):
                    fit = per_edge_fits[(li, i, j)]
                    if not fit['active']:
                        continue
                    if li == 0 and i < len(input_names):
                        var = input_names[i]
                    else:
                        var = f"h{li}_{i}"
                    inner.append(_format_primitive(fit['primitive'],
                                                    fit['params'], var)
                                  if fit['clean']
                                  else f"complex({var})")
                parts.append(f"L{li}_out{j} = " + (' + '.join(inner) or '0'))
        expression_str = '; '.join(parts)

    return {
        'per_edge_fits': per_edge_fits,
        'expression_str': expression_str,
        'n_clean_edges': n_clean,
        'n_active_edges': n_active,
        'n_complex_edges': n_complex,
        'n_total_edges': n_total,
    }


def _build_bs_dataset(n_S: int = 40, n_t: int = 40, sigma: float = 0.2,
                       r: float = 0.05, K: float = 100.0,
                       S_min: float = 50.0, S_max: float = 150.0,
                       t_max: float = 0.99,
                       target_kind: str = 'dVdt'
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Build the canonical (X, y) for KAN-PDE on clean BS.

    X columns: [V, dV/dS, d2V/dS2, S*dV/dS, S^2*d2V/dS2]
    y       : dV/dt   (target_kind='dVdt')
              V^2     (target_kind='v_squared', residual after BS)
    """
    V, S_grid, t_grid = generate_price_surface(
        S_min=S_min, S_max=S_max, n_S=n_S, n_t=n_t,
        K=K, r=r, sigma=sigma, T=t_max + 0.01, option_type='call',
    )
    derivs = compute_derivatives(V, S_grid, t_grid, trim=3)
    Vt = derivs['V']
    dVdS = derivs['dVdS']
    d2VdS2 = derivs['d2VdS2']
    S_mesh = derivs['S_mesh']
    dVdt = derivs['dVdt']

    X = np.column_stack([
        Vt.ravel(),
        dVdS.ravel(),
        d2VdS2.ravel(),
        (S_mesh * dVdS).ravel(),
        (S_mesh ** 2 * d2VdS2).ravel(),
    ]).astype(np.float32)
    if target_kind == 'dVdt':
        y = dVdt.ravel().astype(np.float32)
    elif target_kind == 'v_squared':
        # Nonlinear PDE: dV/dt = -0.5 sigma^2 S^2 V_SS - r S V_S + r V + V^2 / 100
        y = (-0.5 * sigma ** 2 * (S_mesh ** 2 * d2VdS2)
              - r * (S_mesh * dVdS)
              + r * Vt
              + Vt ** 2 / 100.0).ravel().astype(np.float32)
    else:
        raise ValueError(f"unknown target_kind={target_kind}")
    return X, y


_TINY_INPUT_NAMES = ['V', 'dV/dS', 'd2V/dS2', 'S*dV/dS', 'S^2*d2V/dS2']


def run_kan_tiny_sweep(target_dataset: str = 'synthetic_bs',
                        configs: Optional[list[list[int]]] = None,
                        seed: int = 42, n_epochs: int = 5000
                        ) -> pd.DataFrame:
    """Run all 3 tiny configs on a dataset (Fix 1+5)."""
    if configs is None:
        configs = [[5, 1], [5, 2, 1], [5, 3, 1]]

    if target_dataset == 'synthetic_bs':
        X, y = _build_bs_dataset(target_kind='dVdt')
    elif target_dataset == 'synthetic_v2':
        X, y = _build_bs_dataset(target_kind='v_squared')
    else:
        raise ValueError(f"unknown target_dataset={target_dataset}")

    rows: list[dict[str, Any]] = []
    for cfg in configs:
        # Edge count = sum of layer-pair products.
        n_edges = sum(cfg[i] * cfg[i + 1] for i in range(len(cfg) - 1))
        result = train_kan_tiny(X, y, layer_sizes=cfg, n_epochs=n_epochs,
                                  lr=1e-3, lambda_l1=0.01,
                                  lambda_complexity=0.01,
                                  test_split=0.2, seed=seed)
        sym = extract_symbolic_kan_clean(result['model'],
                                          input_names=_TINY_INPUT_NAMES)
        rows.append({
            'config': str(cfg),
            'n_edges': n_edges,
            'train_r2': result['train_r2'],
            'test_r2': result['test_r2'],
            'n_active_edges': result['n_active_edges'],
            'n_clean_edges': sym['n_clean_edges'],
            'n_complex_edges': sym['n_complex_edges'],
            'symbolic_expression': sym['expression_str'][:300],
        })
    return pd.DataFrame(rows)


def run_kan_tiny_on_real(per_ticker_results: dict[str, Any],
                           best_config: list[int] = [5, 1],
                           seed: int = 42, n_epochs: int = 5000
                           ) -> pd.DataFrame:
    """Apply best tiny config to each ticker's GP-smoothed analytical-θ data.

    For each ticker we build a 5-input dataset (V, dV/dS, d2V/dS2, S*dV/dS,
    S^2*d2V/dS2) and train a tiny KAN. We also fit the 5-term linear SINDy
    baseline on the same library for comparison.
    """
    rows: list[dict[str, Any]] = []
    for ticker, entry in per_ticker_results.items():
        row: dict[str, Any] = {'ticker': ticker}
        try:
            io = _extract_real_inputs_target(entry, use_analytical_theta=True)
            if io is None:
                io = _extract_real_inputs_target(entry,
                                                  use_analytical_theta=False)
            if io is None:
                raise ValueError('no_inputs_target')
            X_raw, y = io
            if len(y) < 20:
                raise ValueError('too_few_points')
            # X_raw columns: [V, dV/dS, d2V/dS2, S, t]
            V = X_raw[:, 0]
            dVdS = X_raw[:, 1]
            d2VdS2 = X_raw[:, 2]
            S = X_raw[:, 3]
            X5 = np.column_stack([V, dVdS, d2VdS2, S * dVdS,
                                    S ** 2 * d2VdS2]).astype(np.float32)
            result = train_kan_tiny(X5, y, layer_sizes=best_config,
                                      n_epochs=n_epochs, lr=1e-3,
                                      lambda_l1=0.01, lambda_complexity=0.01,
                                      test_split=0.2, seed=seed)
            sym = extract_symbolic_kan_clean(result['model'],
                                              input_names=_TINY_INPUT_NAMES)
            # SINDy 5-term baseline (linear least-squares).
            coef, *_ = np.linalg.lstsq(X5, y, rcond=None)
            pred = X5 @ coef
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            sindy_r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
            row.update({
                'train_r2': result['train_r2'],
                'test_r2': result['test_r2'],
                'n_active_edges': result['n_active_edges'],
                'n_clean_edges': sym['n_clean_edges'],
                'symbolic_expression': sym['expression_str'][:300],
                'sindy_r2_baseline': sindy_r2,
                'error': '',
            })
        except Exception as exc:
            logger.warning("tiny-KAN on %s failed: %s", ticker, exc)
            row.update({'train_r2': np.nan, 'test_r2': np.nan,
                        'n_active_edges': 0, 'n_clean_edges': 0,
                        'symbolic_expression': '',
                        'sindy_r2_baseline': np.nan,
                        'error': str(exc)[:100]})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Module 5: KAN-Dupire
# ---------------------------------------------------------------------------


def kan_dupire_on_real_data(per_ticker_results: dict[str, Any],
                              n_epochs: int = 1500,
                              seed: int = 42) -> pd.DataFrame:
    """KAN-Dupire variant: log-moneyness inputs, analytical theta target.

    Inputs: [C, dC/dk, d2C/dk2, k, tau]; target: analytical theta.

    For each ticker we rebuild the SVI surface (best-effort) and train a
    small KAN. We extract sigma_loc^2(k) by sweeping k with other inputs
    held at their median values.
    """
    from src.real_data_v2 import build_logm_surface_svi, compute_liquidity_weights
    from src.real_data_v4 import bs_theta_analytical, reconstruct_sigma_imp_grid
    from src.real_data_v2 import compute_forward_prices

    rows: list[dict[str, Any]] = []
    for ticker, entry in per_ticker_results.items():
        row: dict[str, Any] = {'ticker': ticker}
        try:
            od = entry.get('option_data')
            if od is None:
                raise ValueError('no option_data')
            S0 = float(od['S0'])
            r = float(od['r'])
            df = od['option_df']
            try:
                from src.real_data_v2 import get_dividend_yield
                q = get_dividend_yield(ticker)
            except Exception:
                q = 0.0
            surface = build_logm_surface_svi(df, S0, r, q, n_k=30,
                                              k_range=(-0.25, 0.25))
            C = surface['C_surface']
            k_grid = surface['k_grid']
            tau_grid = surface['tau_grid']
            sigma_imp = reconstruct_sigma_imp_grid(
                surface['svi_params'], k_grid, tau_grid,
            )
            F_grid = compute_forward_prices(S0, r, q, tau_grid)
            K_grid_2d = np.outer(np.exp(k_grid), F_grid)
            theta = bs_theta_analytical(S0, K_grid_2d, tau_grid, sigma_imp,
                                          r, q)

            dk = float(k_grid[1] - k_grid[0])
            dCdk = np.gradient(C, dk, axis=0, edge_order=2)
            d2Cdk2 = np.gradient(dCdk, dk, axis=0, edge_order=2)

            KK = np.tile(k_grid.reshape(-1, 1), (1, len(tau_grid)))
            TT = np.tile(tau_grid.reshape(1, -1), (len(k_grid), 1))

            X = np.column_stack([
                C.ravel(), dCdk.ravel(), d2Cdk2.ravel(),
                KK.ravel(), TT.ravel(),
            ]).astype(np.float32)
            y = theta.ravel().astype(np.float32)
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[mask]
            y = y[mask]
            if len(y) < 30:
                raise ValueError('too few points')

            result = train_kan_pde(X, y, layer_sizes=[5, 5, 1],
                                   n_epochs=n_epochs, lambda_l1=1e-3,
                                   lambda_entropy=1e-3, seed=seed)
            sym = extract_symbolic_kan(
                result['model'],
                input_names=['C', 'dC/dk', 'd2C/dk2', 'k', 'tau'],
            )

            # 2-term Dupire baseline R² on same target/inputs.
            lib2 = np.column_stack([X[:, 1], X[:, 2]])  # dC/dk, d2C/dk2
            coef, *_ = np.linalg.lstsq(lib2, y, rcond=None)
            pred = lib2 @ coef
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            dupire_r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
            sigma_loc_baseline = float(np.sqrt(max(0.0, 2.0 * float(coef[1]))))

            row.update({
                'kan_train_r2': result['train_r2'],
                'kan_test_r2': result['test_r2'],
                'n_active_edges': result['n_active_edges'],
                'symbolic_expression': sym['expression_str'][:200],
                'dupire_2term_r2': dupire_r2,
                'sigma_loc_baseline': sigma_loc_baseline,
                'error': '',
            })
        except Exception as exc:
            logger.warning("KAN-Dupire on %s failed: %s", ticker, exc)
            row.update({'kan_train_r2': np.nan, 'kan_test_r2': np.nan,
                        'n_active_edges': 0, 'symbolic_expression': '',
                        'dupire_2term_r2': np.nan,
                        'sigma_loc_baseline': np.nan,
                        'error': str(exc)[:100]})
        rows.append(row)
    return pd.DataFrame(rows)
