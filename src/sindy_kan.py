"""
Unified SINDy-KAN-Dupire framework.

The PRD's central insight: in log-moneyness Dupire space, SINDy has established
that only 2 terms matter -- the drift (dC/dk) and the diffusion (d2C/dk2). So a
[2, 1] KAN with just 2 learned univariate activation functions is the most
interpretable architecture possible -- and the phi_diffusion curve IS the
discovered local-volatility structure.

This module
-----------
1. ``generate_synthetic_dupire_constsig``  -- BS call surface, constant sigma
2. ``generate_synthetic_dupire_smile``     -- BS call surface, smile sigma(k)
3. ``train_kan_dupire_21``                 -- train a [2, 1] KAN on (dC/dk,
                                              d2C/dk2) -> theta
4. ``extract_activations``                 -- per-edge (x, phi(x)) sweeps
5. ``extract_sigma_loc_from_kan``          -- invert phi_diffusion -> sigma_loc
6. ``sindy_kan_dupire_on_ticker``          -- full real-data orchestrator
7. ``sindy_kan_dupire_all_tickers``        -- loop over per_ticker_results
8. ``plot_sindy_kan_activations``          -- 2-panel publication figure

All steps are wrapped in try/except so a single bad ticker can't crash a sweep.
CPU only. Seed = 42 throughout.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.utils import set_all_seeds, setup_logging
from src.kan_pde import MinimalKAN
from src.data_generation import bs_call_price
from src.real_data_v2 import (
    build_logm_surface_svi,
    compute_forward_prices,
    get_dividend_yield,
)
from src.real_data_v4 import (
    bs_theta_analytical,
    reconstruct_sigma_imp_grid,
    _central_dk,
    _r2_from,
)

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Step 1: synthetic data generators
# ---------------------------------------------------------------------------


def _bs_call_safe(S0: float, K: float, r: float, sigma: float,
                  tau: float, q: float) -> float:
    """BS call price with continuous dividend yield ``q``.

    Uses the standard substitution: a call on a stock with yield q equals a
    call on a non-dividend stock with spot ``S0 * exp(-q*tau)``.
    """
    Sd = float(S0) * float(np.exp(-q * tau))
    return float(bs_call_price(Sd, K, r, sigma, tau))


def _make_grid(S0: float, r: float, q: float, n_k: int, n_tau: int,
               k_range: tuple[float, float], tau_range: tuple[float, float]
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (k_grid, tau_grid, F_grid, K_grid_2d) for a regular Dupire grid."""
    k_grid = np.linspace(float(k_range[0]), float(k_range[1]), int(n_k))
    tau_grid = np.linspace(float(tau_range[0]), float(tau_range[1]), int(n_tau))
    F_grid = compute_forward_prices(S0, r, q, tau_grid)  # shape (n_tau,)
    K_grid_2d = np.outer(np.exp(k_grid), F_grid)  # (n_k, n_tau)
    return k_grid, tau_grid, F_grid, K_grid_2d


def generate_synthetic_dupire_constsig(sigma: float = 0.20, r: float = 0.05,
                                        q: float = 0.0, S0: float = 100.0,
                                        n_k: int = 40, n_tau: int = 15,
                                        k_range: tuple[float, float] = (-0.25, 0.25),
                                        tau_range: tuple[float, float] = (0.05, 1.5),
                                        ) -> dict[str, Any]:
    """Generate BS prices on a log-moneyness grid with constant sigma.

    For each (k, tau): K = F(tau) * exp(k); C = BS call price; theta analytic.

    Returns
    -------
    dict
        {'C', 'k', 'tau', 'sigma_imp', 'theta', 'dCdk', 'd2Cdk2',
         'sigma_true', 'r', 'q', 'S0'}.
    """
    try:
        k_grid, tau_grid, F_grid, K_grid_2d = _make_grid(
            S0, r, q, n_k, n_tau, k_range, tau_range)
        C = np.zeros((n_k, n_tau), dtype=np.float64)
        for i in range(n_k):
            for j in range(n_tau):
                C[i, j] = _bs_call_safe(S0, float(K_grid_2d[i, j]), r,
                                         float(sigma), float(tau_grid[j]), q)
        sigma_imp = np.full((n_k, n_tau), float(sigma))
        theta = bs_theta_analytical(S0, K_grid_2d, tau_grid, sigma_imp, r, q)
        dCdk, d2Cdk2 = _central_dk(C, k_grid)
        return {
            'C': C, 'k': k_grid, 'tau': tau_grid,
            'sigma_imp': sigma_imp, 'theta': theta,
            'dCdk': dCdk, 'd2Cdk2': d2Cdk2,
            'sigma_true': float(sigma), 'r': float(r), 'q': float(q),
            'S0': float(S0),
        }
    except Exception as exc:
        logger.warning("generate_synthetic_dupire_constsig failed: %s", exc)
        raise


def generate_synthetic_dupire_smile(sigma_atm: float = 0.20,
                                     smile_curvature: float = 0.10,
                                     r: float = 0.05, q: float = 0.0,
                                     S0: float = 100.0,
                                     n_k: int = 40, n_tau: int = 15,
                                     k_range: tuple[float, float] = (-0.25, 0.25),
                                     tau_range: tuple[float, float] = (0.05, 1.5),
                                     ) -> dict[str, Any]:
    """BS call surface with smile sigma(k) = sigma_atm + smile_curvature*k^2."""
    try:
        k_grid, tau_grid, F_grid, K_grid_2d = _make_grid(
            S0, r, q, n_k, n_tau, k_range, tau_range)
        sigma_imp = np.zeros((n_k, n_tau), dtype=np.float64)
        for i, k in enumerate(k_grid):
            sigma_k = float(sigma_atm) + float(smile_curvature) * (float(k) ** 2)
            sigma_imp[i, :] = max(sigma_k, 1e-3)
        C = np.zeros((n_k, n_tau), dtype=np.float64)
        for i in range(n_k):
            for j in range(n_tau):
                C[i, j] = _bs_call_safe(S0, float(K_grid_2d[i, j]), r,
                                         float(sigma_imp[i, j]),
                                         float(tau_grid[j]), q)
        theta = bs_theta_analytical(S0, K_grid_2d, tau_grid, sigma_imp, r, q)
        dCdk, d2Cdk2 = _central_dk(C, k_grid)
        return {
            'C': C, 'k': k_grid, 'tau': tau_grid,
            'sigma_imp': sigma_imp, 'theta': theta,
            'dCdk': dCdk, 'd2Cdk2': d2Cdk2,
            'sigma_atm': float(sigma_atm),
            'smile_curvature': float(smile_curvature),
            'r': float(r), 'q': float(q), 'S0': float(S0),
        }
    except Exception as exc:
        logger.warning("generate_synthetic_dupire_smile failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Step 2: [2, 1] KAN-Dupire trainer
# ---------------------------------------------------------------------------


def _standardize_to_unit(x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-column min-max standardize to [-1, 1]. Returns (x_std, params)."""
    x = np.asarray(x, dtype=np.float64)
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    x_range = np.maximum(x_max - x_min, 1e-12)
    x_center = (x_max + x_min) / 2.0
    x_std = 2.0 * (x - x_center) / x_range
    params = {'min': x_min, 'max': x_max, 'center': x_center, 'range': x_range}
    return x_std.astype(np.float32), params


def _standardize_y(y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    y = np.asarray(y, dtype=np.float64)
    y_mean = float(y.mean())
    y_std = float(y.std()) if float(y.std()) > 1e-12 else 1.0
    y_n = (y - y_mean) / y_std
    return y_n.astype(np.float32), {'mean': y_mean, 'std': y_std}


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return _r2_from(np.asarray(y_true, dtype=np.float64),
                    np.asarray(y_pred, dtype=np.float64))


def train_kan_dupire_21(dCdk: np.ndarray, d2Cdk2: np.ndarray,
                          theta_target: np.ndarray,
                          n_grid: int = 8, n_epochs: int = 5000,
                          lr: float = 1e-3,
                          lambda_l1: float = 0.005,
                          lambda_complexity: float = 0.005,
                          test_split: float = 0.2,
                          seed: int = 42,
                          verbose: bool = False) -> dict[str, Any]:
    """Train a [2, 1] KAN: inputs (dC/dk, d2C/dk2), output theta = dC/dtau.

    Inputs are standardized per column to [-1, 1]; the target is standardized
    to zero mean and unit std. The returned dict carries the standardization
    parameters so callers can invert.

    Returns
    -------
    dict
        ``model``, ``train_r2``, ``test_r2``, ``x_std_params``,
        ``y_std_params``, ``n_train``, ``n_test``, ``loss_history``.
    """
    set_all_seeds(seed)
    torch.manual_seed(seed)

    try:
        # Flatten the grids; mask NaNs.
        a = np.asarray(dCdk, dtype=np.float64).ravel()
        b = np.asarray(d2Cdk2, dtype=np.float64).ravel()
        t = np.asarray(theta_target, dtype=np.float64).ravel()
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(t)
        a, b, t = a[ok], b[ok], t[ok]
        if a.size < 20:
            raise ValueError(f"Too few finite training points: {a.size}")

        X = np.column_stack([a, b])
        X_std, x_std_params = _standardize_to_unit(X)
        y_std, y_std_params = _standardize_y(t)

        n = X_std.shape[0]
        g = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(n, generator=g).numpy()
        n_test = max(1, int(n * test_split))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]

        X_train = torch.tensor(X_std[train_idx], dtype=torch.float32)
        X_test = torch.tensor(X_std[test_idx], dtype=torch.float32)
        y_train = torch.tensor(y_std[train_idx], dtype=torch.float32)
        y_test = torch.tensor(y_std[test_idx], dtype=torch.float32)

        model = MinimalKAN(layer_sizes=[2, 1], n_grid=int(n_grid),
                            spline_order=3, base_fn='identity',
                            input_extent=(-1.0, 1.0))
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))

        loss_history: list[float] = []
        for epoch in range(int(n_epochs)):
            opt.zero_grad()
            pred = model(X_train)
            mse = F.mse_loss(pred, y_train)
            reg = model.regularization_loss(lambda_l1=float(lambda_l1),
                                             lambda_entropy=0.0)
            # Total-variation surrogate on the 2 edges -- discourages wiggles.
            xs = torch.linspace(-1.0, 1.0, 32)
            tv = torch.tensor(0.0)
            for e in model.edges:
                y_e = e(xs)
                tv = tv + (y_e[1:] - y_e[:-1]).abs().sum()
            loss = mse + reg + float(lambda_complexity) * tv
            loss.backward()
            opt.step()
            loss_history.append(float(loss.item()))
            if verbose and (epoch % 1000 == 0):
                logger.info("epoch %d: loss=%.6f mse=%.6f", epoch,
                            loss.item(), mse.item())

        model.eval()
        with torch.no_grad():
            pred_train = model(X_train).numpy()
            pred_test = model(X_test).numpy()

        # Un-standardize to physical theta units.
        y_mean = y_std_params['mean']; y_sd = y_std_params['std']
        train_r2 = _r2(t[train_idx], pred_train * y_sd + y_mean)
        test_r2 = _r2(t[test_idx], pred_test * y_sd + y_mean)

        return {
            'model': model,
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'x_std_params': x_std_params,
            'y_std_params': y_std_params,
            'n_train': int(len(train_idx)),
            'n_test': int(len(test_idx)),
            'loss_history': loss_history,
        }
    except Exception as exc:
        logger.warning("train_kan_dupire_21 failed: %s", exc)
        raise


def extract_activations(model: MinimalKAN, n_points: int = 200
                         ) -> tuple[np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]:
    """Sample the two edges of a [2, 1] KAN.

    Returns
    -------
    (act_drift_x, act_drift_y, act_diff_x, act_diff_y)
        Each array has length ``n_points``. The drift edge is input 0
        (corresponds to dC/dk), the diffusion edge is input 1 (d2C/dk2).
    """
    try:
        xs_np = np.linspace(-1.0, 1.0, int(n_points)).astype(np.float32)
        xs = torch.tensor(xs_np)
        model.eval()
        with torch.no_grad():
            # Edge (0, 0, 0) -- input 0 -> output 0 (drift).
            idx0 = model._edge_index(0, 0, 0)
            y0 = model.edges[idx0](xs).numpy()
            # Edge (0, 1, 0) -- input 1 -> output 0 (diffusion).
            idx1 = model._edge_index(0, 1, 0)
            y1 = model.edges[idx1](xs).numpy()
        return xs_np, y0, xs_np, y1
    except Exception as exc:
        logger.warning("extract_activations failed: %s", exc)
        raise


def _activation_linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Best linear fit R^2 of y vs x (1D); used as a 'linearity' diagnostic."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return 0.0
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    return _r2(y, pred)


def _eval_edge_at_std(model: MinimalKAN, edge_idx_in: int,
                       x_std_vals: np.ndarray) -> np.ndarray:
    """Evaluate a single first-layer edge at standardized x values."""
    xt = torch.tensor(np.asarray(x_std_vals, dtype=np.float32))
    idx = model._edge_index(0, edge_idx_in, 0)
    model.eval()
    with torch.no_grad():
        return model.edges[idx](xt).numpy()


def extract_sigma_loc_from_kan(model: MinimalKAN, k_grid: np.ndarray,
                                 tau_grid: np.ndarray,
                                 dCdk_grid: np.ndarray,
                                 d2Cdk2_grid: np.ndarray,
                                 x_std_params: dict[str, np.ndarray],
                                 y_std_params: dict[str, float]
                                 ) -> np.ndarray:
    """Recover sigma_loc(k, tau) from a trained [2, 1] KAN.

    For each grid point (k_i, tau_j):

    - standardize the observed d2C/dk2 value with the same center/range used
      at training time,
    - evaluate phi_diffusion at that standardized x to get a standardized
      diffusion contribution,
    - un-standardize back to physical theta units (multiply by y_std),
    - solve sigma_loc^2 = 2 * diffusion_contribution / d2C/dk2.

    Returns
    -------
    ndarray
        sigma_loc grid of shape ``(n_k, n_tau)``. Negative sigma^2 entries
        and points with vanishing d2C/dk2 become NaN.
    """
    try:
        center = float(x_std_params['center'][1])
        rng = float(x_std_params['range'][1])
        y_sd = float(y_std_params['std'])

        d2 = np.asarray(d2Cdk2_grid, dtype=np.float64)
        # Standardize d2 with the same rule used in _standardize_to_unit.
        d2_std = (2.0 * (d2 - center) / rng).astype(np.float32)

        flat_std = d2_std.ravel()
        diff_contrib_std = _eval_edge_at_std(model, 1, flat_std)

        # Subtract the activation at d2=0 (a learned intercept) so we recover
        # the *marginal* diffusion contribution. The Dupire relation
        # sigma_loc^2 = 2 * diff/(d2C/dk2) requires diff(d2=0) = 0; the KAN
        # is free to absorb a constant into either edge during training.
        d2_zero_std = (2.0 * (0.0 - center) / rng)
        intercept_std = float(_eval_edge_at_std(
            model, 1, np.array([d2_zero_std], dtype=np.float32))[0])

        # Un-standardize back into physical theta units.
        diff_contrib = (diff_contrib_std - intercept_std).reshape(d2.shape) * y_sd

        with np.errstate(divide='ignore', invalid='ignore'):
            sigma_sq = np.where(np.abs(d2) > 1e-6,
                                 2.0 * diff_contrib / d2, np.nan)
        sigma_sq = np.where(np.isfinite(sigma_sq), sigma_sq, np.nan)
        sigma_sq = np.where(sigma_sq > 0, sigma_sq, np.nan)
        sigma_loc = np.sqrt(sigma_sq)
        return sigma_loc
    except Exception as exc:
        logger.warning("extract_sigma_loc_from_kan failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Step 3: real-data orchestrator
# ---------------------------------------------------------------------------


def _linear_dupire_2term_r2(dCdk: np.ndarray, d2Cdk2: np.ndarray,
                              theta: np.ndarray) -> float:
    """OLS 2-term Dupire baseline R^2 (no weighting, raw lstsq)."""
    a = np.asarray(dCdk, dtype=np.float64).ravel()
    b = np.asarray(d2Cdk2, dtype=np.float64).ravel()
    t = np.asarray(theta, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(t)
    a, b, t = a[ok], b[ok], t[ok]
    if a.size < 5:
        return float('nan')
    A = np.column_stack([a, b])
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    pred = A @ coef
    return _r2(t, pred)


def sindy_kan_dupire_on_ticker(option_data: dict[str, Any], ticker: str,
                                 S0: Optional[float] = None,
                                 r: Optional[float] = None,
                                 q: Optional[float] = None,
                                 train_split: str = 'random',
                                 test_tau_threshold: float = 0.5,
                                 n_k: int = 40, n_tau: int = 15,
                                 k_range: tuple[float, float] = (-0.20, 0.20),
                                 n_epochs: int = 3000,
                                 seed: int = 42) -> dict[str, Any]:
    """Run full SINDy-KAN-Dupire on one ticker.

    Steps
    -----
    1. Build SVI-smoothed (k, tau) surface (``build_logm_surface_svi``).
    2. Compute inputs (dC/dk, d2C/dk2) and target (analytical theta).
    3. Train a [2, 1] KAN-Dupire.
    4. Compute linear 2-term Dupire baseline R^2 on the same target.
    5. Extract sigma_loc grid via ``extract_sigma_loc_from_kan``.
    6. If ``train_split == 'temporal'``, retrain on tau < threshold and
       evaluate on tau >= threshold for out-of-sample R^2.

    Returns
    -------
    dict
        ticker, kan_train_r2, kan_test_r2, kan_oos_r2, linear_dupire_r2,
        sigma_loc_grid, sigma_loc_median, drift_activation_linear_r2,
        diffusion_activation_linear_r2, market_avg_iv, q, n_options, model.
        On failure: dict with ``error`` populated.
    """
    out: dict[str, Any] = {'ticker': ticker}
    try:
        S0_v = float(S0 if S0 is not None else option_data.get('S0'))
        r_v = float(r if r is not None else option_data.get('r', 0.05))
        q_v = float(q) if q is not None else None
        if q_v is None:
            q_v = float(option_data.get('q', get_dividend_yield(ticker)))

        df = option_data.get('option_df')
        if df is None:
            raise ValueError("option_data missing 'option_df'")

        # Accept either 'mid' or 'mid_price'. build_logm_surface_svi expects
        # 'mid_price' + 'implied_vol'.
        if 'mid_price' not in df.columns and 'mid' in df.columns:
            df = df.rename(columns={'mid': 'mid_price'})

        surf = build_logm_surface_svi(df, S0_v, r_v, q_v,
                                       n_k=n_k, k_range=k_range, n_tau=n_tau)
        C = surf['C_surface']
        k_grid = surf['k_grid']
        tau_grid = surf['tau_grid']
        svi_params = surf['svi_params']

        # Per-grid sigma_imp -- reuse v4's evaluator.
        sigma_imp = reconstruct_sigma_imp_grid(svi_params, k_grid, tau_grid)
        F_grid = compute_forward_prices(S0_v, r_v, q_v, tau_grid)
        K_grid_2d = np.outer(np.exp(k_grid), F_grid)
        theta = bs_theta_analytical(S0_v, K_grid_2d, tau_grid,
                                     sigma_imp, r_v, q_v)
        dCdk, d2Cdk2 = _central_dk(C, k_grid)

        # Linear 2-term Dupire baseline on the same target.
        lin_r2 = _linear_dupire_2term_r2(dCdk, d2Cdk2, theta)
        out['linear_dupire_r2'] = float(lin_r2)

        # Train the [2, 1] KAN on the full grid.
        res = train_kan_dupire_21(dCdk, d2Cdk2, theta, seed=seed,
                                    n_epochs=n_epochs)
        model = res['model']
        out['kan_train_r2'] = res['train_r2']
        out['kan_test_r2'] = res['test_r2']
        out['x_std_params'] = res['x_std_params']
        out['y_std_params'] = res['y_std_params']
        out['model'] = model

        # sigma_loc extraction.
        sigma_loc = extract_sigma_loc_from_kan(
            model, k_grid, tau_grid, dCdk, d2Cdk2,
            res['x_std_params'], res['y_std_params'])
        out['sigma_loc_grid'] = sigma_loc
        sl_finite = sigma_loc[np.isfinite(sigma_loc)]
        out['sigma_loc_median'] = (float(np.median(sl_finite))
                                    if sl_finite.size > 0 else float('nan'))

        # Activation linearity diagnostics.
        x0, y0, x1, y1 = extract_activations(model, n_points=200)
        out['drift_activation_linear_r2'] = _activation_linear_r2(x0, y0)
        out['diffusion_activation_linear_r2'] = _activation_linear_r2(x1, y1)

        # Market-average IV for reference.
        sigma_finite = sigma_imp[np.isfinite(sigma_imp)]
        out['market_avg_iv'] = (float(np.mean(sigma_finite))
                                 if sigma_finite.size > 0 else float('nan'))
        out['q'] = float(q_v)
        out['n_options'] = int(len(df))

        # Out-of-sample temporal split (optional).
        kan_oos_r2 = float('nan')
        if train_split == 'temporal':
            tau_thr = float(test_tau_threshold)
            tau_mat = np.tile(tau_grid.reshape(1, -1), (C.shape[0], 1))
            mask_train = tau_mat < tau_thr
            mask_test = tau_mat >= tau_thr
            if mask_train.sum() >= 20 and mask_test.sum() >= 5:
                try:
                    res_oos = train_kan_dupire_21(
                        dCdk[mask_train], d2Cdk2[mask_train],
                        theta[mask_train], seed=seed, n_epochs=n_epochs)
                    # Standardize test inputs with the trained model's
                    # standardization.
                    xp = res_oos['x_std_params']
                    yp = res_oos['y_std_params']
                    a_t = dCdk[mask_test].ravel()
                    b_t = d2Cdk2[mask_test].ravel()
                    t_t = theta[mask_test].ravel()
                    a_std = (2.0 * (a_t - xp['center'][0]) / xp['range'][0])
                    b_std = (2.0 * (b_t - xp['center'][1]) / xp['range'][1])
                    X_test = torch.tensor(
                        np.column_stack([a_std, b_std]).astype(np.float32))
                    m = res_oos['model']
                    m.eval()
                    with torch.no_grad():
                        pred_std = m(X_test).numpy()
                    pred_phys = pred_std * yp['std'] + yp['mean']
                    kan_oos_r2 = _r2(t_t, pred_phys)
                except Exception as exc:
                    logger.warning("OoS branch on %s failed: %s",
                                   ticker, exc)
        out['kan_oos_r2'] = float(kan_oos_r2)

        return out
    except Exception as exc:
        logger.warning("sindy_kan_dupire_on_ticker(%s) failed: %s",
                       ticker, exc)
        out.update({
            'error': str(exc)[:200],
            'kan_train_r2': float('nan'),
            'kan_test_r2': float('nan'),
            'kan_oos_r2': float('nan'),
            'linear_dupire_r2': float('nan'),
            'sigma_loc_median': float('nan'),
            'drift_activation_linear_r2': float('nan'),
            'diffusion_activation_linear_r2': float('nan'),
            'market_avg_iv': float('nan'),
            'q': float('nan'),
            'n_options': 0,
            'model': None,
            'sigma_loc_grid': None,
        })
        return out


def sindy_kan_dupire_all_tickers(per_ticker_results: dict[str, Any],
                                   train_split: str = 'temporal',
                                   test_tau_threshold: float = 0.5,
                                   n_epochs: int = 3000,
                                   seed: int = 42) -> dict[str, dict[str, Any]]:
    """Loop ``sindy_kan_dupire_on_ticker`` over per-ticker results.

    ``per_ticker_results`` is a dict of {ticker: {'option_data': {...}}}
    in the same shape as the v2/v3/v4 pipelines produce.
    """
    results: dict[str, dict[str, Any]] = {}
    for ticker, entry in per_ticker_results.items():
        try:
            od = entry.get('option_data')
            if od is None:
                results[ticker] = {'ticker': ticker, 'error': 'no_option_data'}
                continue
            res = sindy_kan_dupire_on_ticker(
                od, ticker, train_split=train_split,
                test_tau_threshold=test_tau_threshold,
                n_epochs=n_epochs, seed=seed)
            results[ticker] = res
        except Exception as exc:
            logger.warning("ticker %s failed: %s", ticker, exc)
            results[ticker] = {'ticker': ticker, 'error': str(exc)[:200]}
    return results


# ---------------------------------------------------------------------------
# Step 4: 2-panel comparison figure
# ---------------------------------------------------------------------------


def plot_sindy_kan_activations(synth_const_res: dict[str, Any],
                                 synth_smile_res: dict[str, Any],
                                 real_spy_res: dict[str, Any],
                                 save_path: str = 'outputs/figures/paper/sindy_kan_activations'
                                 ) -> Optional[str]:
    """Two-panel publication-quality figure comparing learned activations.

    Panel 1: phi_drift (input is dC/dk).
    Panel 2: phi_diffusion (input is d2C/dk2).
    Three curves per panel: synth const-sigma (blue), synth smile (green),
    real SPY (red).

    Saves both ``{save_path}.png`` and ``{save_path}.pdf``. Returns the
    PNG path on success, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'legend.fontsize': 9,
        })

        def _get_acts(res: dict[str, Any]):
            m = res.get('model')
            if m is None:
                return None
            try:
                return extract_activations(m, n_points=200)
            except Exception:
                return None

        a_const = _get_acts(synth_const_res)
        a_smile = _get_acts(synth_smile_res)
        a_spy = _get_acts(real_spy_res)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        # Panel 1: phi_drift
        ax = axes[0]
        for color, label, acts in (
            ('tab:blue', 'Synth const-sigma', a_const),
            ('tab:green', 'Synth smile', a_smile),
            ('tab:red', 'Real SPY', a_spy),
        ):
            if acts is not None:
                x0, y0, _, _ = acts
                ax.plot(x0, y0, color=color, label=label, lw=2)
        ax.set_title(r'$\varphi_{\mathrm{drift}}(\partial C/\partial k)$')
        ax.set_xlabel(r'standardized $\partial C/\partial k$')
        ax.set_ylabel('activation (standardized)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True)

        # Panel 2: phi_diffusion
        ax = axes[1]
        for color, label, acts in (
            ('tab:blue', 'Synth const-sigma', a_const),
            ('tab:green', 'Synth smile', a_smile),
            ('tab:red', 'Real SPY', a_spy),
        ):
            if acts is not None:
                _, _, x1, y1 = acts
                ax.plot(x1, y1, color=color, label=label, lw=2)
        ax.set_title(r'$\varphi_{\mathrm{diffusion}}(\partial^2 C/\partial k^2)$')
        ax.set_xlabel(r'standardized $\partial^2 C/\partial k^2$')
        ax.set_ylabel('activation (standardized)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True)

        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        png_path = f"{save_path}.png"
        pdf_path = f"{save_path}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        return png_path
    except Exception as exc:
        logger.warning("plot_sindy_kan_activations failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Step 5: Stability / sensitivity sweeps (Agent B1)
# ---------------------------------------------------------------------------


def _load_spy_option_data(ticker: str = 'SPY',
                            csv_path: Optional[str] = None) -> dict[str, Any]:
    """Build an ``option_data`` dict from a cached real_chain CSV.

    Looks for ``outputs/tables/real_chain_{ticker}_*.csv`` and selects the
    most recent snapshot (lexicographic max). Returns the dict shape that
    ``sindy_kan_dupire_on_ticker`` expects.
    """
    import glob
    if csv_path is None:
        pattern = os.path.join('outputs', 'tables',
                                f'real_chain_{ticker}_*.csv')
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"No cached chain matches {pattern}")
        csv_path = candidates[-1]
    df = pd.read_csv(csv_path)
    S0 = float(df['S0'].iloc[0])
    r = float(df['r'].iloc[0])
    if 'mid_price' not in df.columns and 'mid' in df.columns:
        df = df.rename(columns={'mid': 'mid_price'})
    return {
        'S0': S0,
        'r': r,
        'q': float(get_dividend_yield(ticker)),
        'option_df': df,
        'data_source': csv_path,
    }


def _eval_activations_on_grid(model: MinimalKAN, n_eval: int = 200
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate both edges on a fixed standardized [-1, 1] grid.

    Returns ``(x_grid, drift_y, diffusion_y)`` of length ``n_eval``.
    """
    xs = np.linspace(-1.0, 1.0, int(n_eval)).astype(np.float32)
    drift = _eval_edge_at_std(model, 0, xs)
    diff = _eval_edge_at_std(model, 1, xs)
    return xs.astype(np.float64), drift.astype(np.float64), diff.astype(np.float64)


def _summarize_curves(curves: np.ndarray) -> dict[str, np.ndarray]:
    """Compute mean and 95% CI (2.5 / 97.5 pct) across the first axis."""
    arr = np.asarray(curves, dtype=np.float64)
    return {
        'mean': arr.mean(axis=0),
        'lower': np.percentile(arr, 2.5, axis=0),
        'upper': np.percentile(arr, 97.5, axis=0),
    }


def _prepare_spy_inputs(ticker: str = 'SPY', n_k: int = 40, n_tau: int = 15,
                          k_range: tuple[float, float] = (-0.20, 0.20)
                          ) -> dict[str, np.ndarray]:
    """Build (dCdk, d2Cdk2, theta) for SPY once; reused by all sweeps.

    Caches the SVI surface so we don't re-fit it per sweep iteration.
    """
    od = _load_spy_option_data(ticker)
    S0 = float(od['S0'])
    r = float(od['r'])
    q = float(od['q'])
    df = od['option_df']
    surf = build_logm_surface_svi(df, S0, r, q,
                                    n_k=n_k, k_range=k_range, n_tau=n_tau)
    C = surf['C_surface']
    k_grid = surf['k_grid']
    tau_grid = surf['tau_grid']
    svi_params = surf['svi_params']
    sigma_imp = reconstruct_sigma_imp_grid(svi_params, k_grid, tau_grid)
    F_grid = compute_forward_prices(S0, r, q, tau_grid)
    K_grid_2d = np.outer(np.exp(k_grid), F_grid)
    theta = bs_theta_analytical(S0, K_grid_2d, tau_grid, sigma_imp, r, q)
    dCdk, d2Cdk2 = _central_dk(C, k_grid)
    return {
        'C': C, 'k': k_grid, 'tau': tau_grid,
        'dCdk': dCdk, 'd2Cdk2': d2Cdk2, 'theta': theta,
        'S0': S0, 'r': r, 'q': q,
    }


def _smooth_surface_with_kernel(C: np.ndarray, k_grid: np.ndarray,
                                  tau_grid: np.ndarray, kernel: str,
                                  seed: int = 42) -> np.ndarray:
    """Re-smooth the C surface with a GP using the chosen kernel.

    ``kernel`` is one of 'RBF', 'Matern32', 'Matern52'. Returns the GP-mean
    surface evaluated on the same (k, tau) grid. Falls back to the unsmoothed
    surface on GP failure.
    """
    try:
        import warnings
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF, Matern, WhiteKernel, ConstantKernel,
        )
        kname = str(kernel).lower()
        K_mesh, T_mesh = np.meshgrid(k_grid, tau_grid, indexing='ij')
        X = np.column_stack([K_mesh.ravel(), T_mesh.ravel()])
        y = np.asarray(C, dtype=np.float64).ravel()
        ok = np.isfinite(y)
        X, y = X[ok], y[ok]
        if X.shape[0] < 10:
            return C
        k_extent = float(k_grid[-1] - k_grid[0])
        t_extent = float(tau_grid[-1] - tau_grid[0])
        ls_init = [max(0.2 * k_extent, 1e-3), max(0.2 * t_extent, 1e-3)]
        y_var = float(np.var(y))
        noise_init = max(1e-6, 1e-4 * y_var)
        if 'matern32' in kname or kname == 'matern_32':
            inner = Matern(length_scale=ls_init, nu=1.5,
                            length_scale_bounds=(1e-3, 1e3))
        elif 'matern52' in kname or kname == 'matern_52':
            inner = Matern(length_scale=ls_init, nu=2.5,
                            length_scale_bounds=(1e-3, 1e3))
        else:
            inner = RBF(length_scale=ls_init,
                         length_scale_bounds=(1e-3, 1e3))
        k_obj = (
            ConstantKernel(constant_value=max(y_var, 1e-3),
                            constant_value_bounds=(1e-5, 1e6))
            * inner
            + WhiteKernel(noise_level=noise_init,
                           noise_level_bounds=(1e-10, 1e2))
        )
        gp = GaussianProcessRegressor(
            kernel=k_obj, n_restarts_optimizer=1,
            normalize_y=True, random_state=int(seed), alpha=0.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X, y)
        X_full = np.column_stack([K_mesh.ravel(), T_mesh.ravel()])
        y_smooth = gp.predict(X_full).reshape(C.shape)
        return y_smooth
    except Exception as exc:
        logger.warning("_smooth_surface_with_kernel(%s) failed: %s",
                       kernel, exc)
        return C


def activation_stability_sweep(ticker: str = 'SPY',
                                 seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
                                 n_eval: int = 200,
                                 n_epochs: int = 1500,
                                 save_csv: str = 'outputs/tables/activation_stability.csv'
                                 ) -> dict[str, Any]:
    """Train [2,1] KAN on SPY with multiple seeds; quantify activation CI.

    For each ``seed`` in ``seeds``, fits the KAN and evaluates both edges on
    a fixed ``n_eval``-point grid in [-1, 1]. Returns a dict with
    ``x_eval`` (length ``n_eval``) plus per-edge ``mean`` / ``lower`` /
    ``upper`` arrays for drift and diffusion.

    Writes ``outputs/tables/activation_stability.csv`` with columns
    ``seed, edge_idx, x_eval, activation``.
    """
    inputs = _prepare_spy_inputs(ticker)
    rows: list[dict[str, float]] = []
    drift_curves: list[np.ndarray] = []
    diff_curves: list[np.ndarray] = []
    x_eval = np.linspace(-1.0, 1.0, int(n_eval))
    for seed in seeds:
        try:
            res = train_kan_dupire_21(
                inputs['dCdk'], inputs['d2Cdk2'], inputs['theta'],
                seed=int(seed), n_epochs=int(n_epochs))
            x, ydrift, ydiff = _eval_activations_on_grid(
                res['model'], n_eval=n_eval)
            drift_curves.append(ydrift)
            diff_curves.append(ydiff)
            for i, xv in enumerate(x):
                rows.append({'seed': int(seed), 'edge_idx': 0,
                              'x_eval': float(xv),
                              'activation': float(ydrift[i])})
                rows.append({'seed': int(seed), 'edge_idx': 1,
                              'x_eval': float(xv),
                              'activation': float(ydiff[i])})
        except Exception as exc:
            logger.warning("activation_stability_sweep seed %d failed: %s",
                           seed, exc)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        pd.DataFrame(rows).to_csv(save_csv, index=False)
    drift_arr = np.asarray(drift_curves)
    diff_arr = np.asarray(diff_curves)
    drift_summary = _summarize_curves(drift_arr) if drift_arr.size > 0 \
        else {'mean': np.zeros(n_eval), 'lower': np.zeros(n_eval),
               'upper': np.zeros(n_eval)}
    diff_summary = _summarize_curves(diff_arr) if diff_arr.size > 0 \
        else {'mean': np.zeros(n_eval), 'lower': np.zeros(n_eval),
               'upper': np.zeros(n_eval)}
    return {
        'x_eval': x_eval,
        'drift': drift_summary,
        'diffusion': diff_summary,
        'seeds': list(seeds),
        'n_seeds_ok': int(len(drift_curves)),
    }


def regularization_sensitivity_sweep(
        ticker: str = 'SPY',
        l1_weights: tuple[float, ...] = (0.001, 0.01, 0.1),
        n_eval: int = 200, seed: int = 42,
        n_epochs: int = 1500,
        save_csv: str = 'outputs/tables/regularization_sensitivity.csv'
        ) -> dict[str, Any]:
    """Sweep L1 regularization strength on a fixed seed; record activations."""
    inputs = _prepare_spy_inputs(ticker)
    rows: list[dict[str, float]] = []
    curves: dict[float, dict[str, np.ndarray]] = {}
    for lam in l1_weights:
        try:
            res = train_kan_dupire_21(
                inputs['dCdk'], inputs['d2Cdk2'], inputs['theta'],
                seed=int(seed), n_epochs=int(n_epochs),
                lambda_l1=float(lam))
            x, ydrift, ydiff = _eval_activations_on_grid(
                res['model'], n_eval=n_eval)
            curves[float(lam)] = {'x': x, 'drift': ydrift, 'diffusion': ydiff}
            for i, xv in enumerate(x):
                rows.append({'l1_weight': float(lam), 'edge_idx': 0,
                              'x_eval': float(xv),
                              'activation': float(ydrift[i])})
                rows.append({'l1_weight': float(lam), 'edge_idx': 1,
                              'x_eval': float(xv),
                              'activation': float(ydiff[i])})
        except Exception as exc:
            logger.warning("regularization sweep lam=%g failed: %s",
                           lam, exc)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        pd.DataFrame(rows).to_csv(save_csv, index=False)
    return {'curves': curves, 'l1_weights': list(l1_weights)}


def kernel_sensitivity_sweep(
        ticker: str = 'SPY',
        kernels: tuple[str, ...] = ('RBF', 'Matern32', 'Matern52'),
        n_eval: int = 200, seed: int = 42,
        n_epochs: int = 1500,
        save_csv: str = 'outputs/tables/kernel_sensitivity.csv'
        ) -> dict[str, Any]:
    """Sweep GP preprocessing kernel; retrain KAN on each smoothed surface."""
    inputs = _prepare_spy_inputs(ticker)
    rows: list[dict[str, float]] = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    for kname in kernels:
        try:
            C_smooth = _smooth_surface_with_kernel(
                inputs['C'], inputs['k'], inputs['tau'],
                kernel=kname, seed=int(seed))
            dCdk, d2Cdk2 = _central_dk(C_smooth, inputs['k'])
            res = train_kan_dupire_21(
                dCdk, d2Cdk2, inputs['theta'],
                seed=int(seed), n_epochs=int(n_epochs))
            x, ydrift, ydiff = _eval_activations_on_grid(
                res['model'], n_eval=n_eval)
            curves[str(kname)] = {'x': x, 'drift': ydrift,
                                   'diffusion': ydiff}
            for i, xv in enumerate(x):
                rows.append({'kernel': str(kname), 'edge_idx': 0,
                              'x_eval': float(xv),
                              'activation': float(ydrift[i])})
                rows.append({'kernel': str(kname), 'edge_idx': 1,
                              'x_eval': float(xv),
                              'activation': float(ydiff[i])})
        except Exception as exc:
            logger.warning("kernel sweep %s failed: %s", kname, exc)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        pd.DataFrame(rows).to_csv(save_csv, index=False)
    return {'curves': curves, 'kernels': list(kernels)}


def plot_activation_stability_figure(
        stability: dict[str, Any],
        save_path: str = 'outputs/figures/paper/activation_stability_figure'
        ) -> Optional[str]:
    """2-panel figure: drift + diffusion activations with 95% CI band.

    Uses colorblind-friendly palette: line #0072B2 (blue), band #56B4E9-ish
    (lighter blue). 300 DPI, serif font, saved as both .png and .pdf.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'legend.fontsize': 9,
        })
        line_color = '#0072B2'
        band_color = '#92C5DE'  # lighter blue, colorblind-friendly

        x = stability['x_eval']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, key, title, xlab in (
            (axes[0], 'drift',
             r'$\varphi_{\mathrm{drift}}(\partial C/\partial k)$',
             r'standardized $\partial C/\partial k$'),
            (axes[1], 'diffusion',
             r'$\varphi_{\mathrm{diffusion}}(\partial^2 C/\partial k^2)$',
             r'standardized $\partial^2 C/\partial k^2$'),
        ):
            s = stability[key]
            ax.fill_between(x, s['lower'], s['upper'],
                             color=band_color, alpha=0.55,
                             label='95% CI across seeds')
            ax.plot(x, s['mean'], color=line_color, lw=2.2,
                     label='mean activation')
            ax.set_title(title)
            ax.set_xlabel(xlab)
            ax.set_ylabel('activation (standardized)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', frameon=True)

        n_ok = stability.get('n_seeds_ok', 0)
        fig.suptitle(
            f"KAN-Dupire activation stability ({n_ok} seeds, SPY)",
            y=1.02, fontsize=12)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        png_path = f"{save_path}.png"
        pdf_path = f"{save_path}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        return png_path
    except Exception as exc:
        logger.warning("plot_activation_stability_figure failed: %s", exc)
        return None
