"""
Compute and compare analytical vs PINN-based Greeks (Delta, Gamma, Theta).

Provides functions to:
- Compute Delta, Gamma, Theta analytically from closed-form Black-Scholes formulas.
- Compute Delta, Gamma, Theta from a trained PINN via automatic differentiation.
- Compare the two sets of Greeks with detailed error metrics across different
  grid regions (full, interior, boundary).
"""

import numpy as np
import torch

from src.data_generation import (
    bs_call_delta,
    bs_put_delta,
    bs_gamma,
    bs_theta_call,
    bs_theta_put,
)
from src.utils import setup_logging

logger = setup_logging(__name__)


def analytical_greeks(S, K, r, sigma, tau, option_type='call'):
    """
    Compute Delta, Gamma, Theta analytically using Black-Scholes formulas.

    Parameters
    ----------
    S : float or ndarray
        Stock price(s). Must be positive.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility. Must be positive.
    tau : float or ndarray
        Time to maturity (T - t). Must be non-negative.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    dict
        {'delta': ndarray, 'gamma': ndarray, 'theta': ndarray}
    """
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)

    if option_type == 'call':
        delta = bs_call_delta(S, K, r, sigma, tau)
        theta = bs_theta_call(S, K, r, sigma, tau)
    elif option_type == 'put':
        delta = bs_put_delta(S, K, r, sigma, tau)
        theta = bs_theta_put(S, K, r, sigma, tau)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    gamma = bs_gamma(S, K, r, sigma, tau)

    logger.info(
        "Analytical Greeks computed: delta range [%.4f, %.4f], "
        "gamma range [%.6f, %.6f], theta range [%.4f, %.4f]",
        np.min(delta), np.max(delta),
        np.min(gamma), np.max(gamma),
        np.min(theta), np.max(theta),
    )

    return {'delta': delta, 'gamma': gamma, 'theta': theta}


def pinn_greeks(model, S_tensor, t_tensor):
    """
    Compute Delta (dV/dS), Gamma (d2V/dS2), Theta (dV/dt) from a trained PINN.

    Uses torch.autograd.grad for automatic differentiation through the network.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PINN that takes (S, t) concatenated along dim=1 and returns V.
    S_tensor : torch.Tensor
        Stock price tensor. Must have requires_grad=True.
    t_tensor : torch.Tensor
        Time tensor. Must have requires_grad=True.

    Returns
    -------
    dict
        {'delta': ndarray, 'gamma': ndarray, 'theta': ndarray}
        All values are numpy arrays on CPU.
    """
    if not S_tensor.requires_grad:
        raise ValueError("S_tensor must have requires_grad=True")
    if not t_tensor.requires_grad:
        raise ValueError("t_tensor must have requires_grad=True")

    model.eval()

    # Forward pass — BSPINN takes separate S, t arguments
    V = model(S_tensor, t_tensor)

    # Delta = dV/dS (with create_graph=True for Gamma)
    dV_dS = torch.autograd.grad(
        outputs=V,
        inputs=S_tensor,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Gamma = d2V/dS2
    d2V_dS2 = torch.autograd.grad(
        outputs=dV_dS,
        inputs=S_tensor,
        grad_outputs=torch.ones_like(dV_dS),
        create_graph=False,
        retain_graph=True,
    )[0]

    # Theta = dV/dt
    dV_dt = torch.autograd.grad(
        outputs=V,
        inputs=t_tensor,
        grad_outputs=torch.ones_like(V),
        create_graph=False,
        retain_graph=False,
    )[0]

    delta_np = dV_dS.detach().cpu().numpy().flatten()
    gamma_np = d2V_dS2.detach().cpu().numpy().flatten()
    theta_np = dV_dt.detach().cpu().numpy().flatten()

    logger.info(
        "PINN Greeks computed: delta range [%.4f, %.4f], "
        "gamma range [%.6f, %.6f], theta range [%.4f, %.4f]",
        np.min(delta_np), np.max(delta_np),
        np.min(gamma_np), np.max(gamma_np),
        np.min(theta_np), np.max(theta_np),
    )

    return {'delta': delta_np, 'gamma': gamma_np, 'theta': theta_np}


def _compute_region_errors(pinn_vals, analytical_vals, mask, region_name):
    """
    Compute error metrics for a specific grid region.

    Parameters
    ----------
    pinn_vals : ndarray
        PINN-predicted Greek values (flattened).
    analytical_vals : ndarray
        Analytical Greek values (flattened).
    mask : ndarray of bool
        Boolean mask selecting the region of interest (flattened).
    region_name : str
        Name of the region for logging.

    Returns
    -------
    dict
        {'mae': float, 'max_abs_error': float, 'relative_l2': float, 'n_points': int}
        Returns NaN metrics and n_points=0 if no points fall in the region.
    """
    if not np.any(mask):
        logger.warning("No points in region '%s'; returning NaN metrics.", region_name)
        return {
            'mae': np.nan,
            'max_abs_error': np.nan,
            'relative_l2': np.nan,
            'n_points': 0,
        }

    p = pinn_vals[mask]
    a = analytical_vals[mask]
    diff = np.abs(p - a)

    mae = float(np.mean(diff))
    max_abs_error = float(np.max(diff))

    # Relative L2 error: ||pinn - analytical||_2 / ||analytical||_2
    l2_num = np.sqrt(np.sum((p - a) ** 2))
    l2_den = np.sqrt(np.sum(a ** 2))
    if l2_den < 1e-15:
        relative_l2 = float(l2_num)  # analytical is essentially zero
    else:
        relative_l2 = float(l2_num / l2_den)

    return {
        'mae': mae,
        'max_abs_error': max_abs_error,
        'relative_l2': relative_l2,
        'n_points': int(np.sum(mask)),
    }


def compare_greeks(pinn_greeks_dict, analytical_greeks_dict,
                   S_grid=None, t_grid=None,
                   S_interior=(70, 130), t_interior=(0, 0.8)):
    """
    Compare PINN-predicted Greeks against analytical Greeks with detailed metrics.

    Computes MAE, max absolute error, and relative L2 error for each Greek
    (Delta, Gamma, Theta) over three regions:
    - Full grid: all points.
    - Interior region: S in [S_interior[0], S_interior[1]] and
      t in [t_interior[0], t_interior[1]].
    - Boundary region: complement of the interior.

    Parameters
    ----------
    pinn_greeks_dict : dict
        {'delta': ndarray, 'gamma': ndarray, 'theta': ndarray} from the PINN.
    analytical_greeks_dict : dict
        {'delta': ndarray, 'gamma': ndarray, 'theta': ndarray} from analytical formulas.
    S_grid : ndarray or None
        1D array of stock prices. Required for interior/boundary split.
    t_grid : ndarray or None
        1D array of calendar times. Required for interior/boundary split.
    S_interior : tuple of float
        (S_min, S_max) defining the interior region in stock price.
    t_interior : tuple of float
        (t_min, t_max) defining the interior region in calendar time.

    Returns
    -------
    dict
        Nested dictionary structured as:
        {
            'delta': {
                'full': {'mae': ..., 'max_abs_error': ..., 'relative_l2': ..., 'n_points': ...},
                'interior': {...},
                'boundary': {...},
            },
            'gamma': {...},
            'theta': {...},
        }
    """
    greek_names = ['delta', 'gamma', 'theta']
    results = {}

    # Build interior/boundary masks if grids are provided
    has_grids = S_grid is not None and t_grid is not None
    if has_grids:
        S_grid = np.asarray(S_grid, dtype=np.float64)
        t_grid = np.asarray(t_grid, dtype=np.float64)
        S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')
        S_flat = S_mesh.flatten()
        t_flat = t_mesh.flatten()

        interior_mask = (
            (S_flat >= S_interior[0]) & (S_flat <= S_interior[1]) &
            (t_flat >= t_interior[0]) & (t_flat <= t_interior[1])
        )
        boundary_mask = ~interior_mask
    else:
        logger.info(
            "S_grid and/or t_grid not provided; only full-grid metrics will be computed."
        )

    for name in greek_names:
        pinn_vals = np.asarray(pinn_greeks_dict[name], dtype=np.float64).flatten()
        analytical_vals = np.asarray(analytical_greeks_dict[name], dtype=np.float64).flatten()

        if pinn_vals.shape != analytical_vals.shape:
            raise ValueError(
                f"Shape mismatch for '{name}': PINN {pinn_vals.shape} vs "
                f"analytical {analytical_vals.shape}"
            )

        full_mask = np.ones(pinn_vals.shape, dtype=bool)

        entry = {}
        entry['full'] = _compute_region_errors(
            pinn_vals, analytical_vals, full_mask, f"{name}/full"
        )

        if has_grids:
            if pinn_vals.size != S_flat.size:
                raise ValueError(
                    f"Grid size mismatch for '{name}': Greek array has {pinn_vals.size} "
                    f"elements but grid has {S_flat.size} points "
                    f"(S_grid: {S_grid.size}, t_grid: {t_grid.size})."
                )
            entry['interior'] = _compute_region_errors(
                pinn_vals, analytical_vals, interior_mask, f"{name}/interior"
            )
            entry['boundary'] = _compute_region_errors(
                pinn_vals, analytical_vals, boundary_mask, f"{name}/boundary"
            )
        else:
            entry['interior'] = {
                'mae': np.nan, 'max_abs_error': np.nan,
                'relative_l2': np.nan, 'n_points': 0,
            }
            entry['boundary'] = {
                'mae': np.nan, 'max_abs_error': np.nan,
                'relative_l2': np.nan, 'n_points': 0,
            }

        results[name] = entry

    # Log summary
    for name in greek_names:
        full = results[name]['full']
        logger.info(
            "%s — Full grid: MAE=%.6f, MaxAE=%.6f, RelL2=%.6f (%d pts)",
            name.capitalize(), full['mae'], full['max_abs_error'],
            full['relative_l2'], full['n_points'],
        )
        if has_grids:
            interior = results[name]['interior']
            boundary = results[name]['boundary']
            logger.info(
                "%s — Interior (S in %s, t in %s): MAE=%.6f, MaxAE=%.6f, "
                "RelL2=%.6f (%d pts)",
                name.capitalize(), S_interior, t_interior,
                interior['mae'], interior['max_abs_error'],
                interior['relative_l2'], interior['n_points'],
            )
            logger.info(
                "%s — Boundary: MAE=%.6f, MaxAE=%.6f, RelL2=%.6f (%d pts)",
                name.capitalize(), boundary['mae'], boundary['max_abs_error'],
                boundary['relative_l2'], boundary['n_points'],
            )

    return results
