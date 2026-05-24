"""
Discovery-method baselines + 5-fold spatial CV on real SPY GP-Dupire data.

Covers Improvements 2 and 3 of the Remaining-Feedback PRD:

PART A -- DISCOVERY METHOD COMPARISON (5 rows):
    1. STLSQ (pulled from existing CSV)
    2. STLSQ + KAN (pulled from existing CSV)
    3. Weak-form regression on the GP-smoothed surface, restricted to the
       2-term Dupire library (dC/dk, d2C/dk2) with theta as target.
    4. Ridge + threshold on the GP-derived 2-term library (call into
       src.baselines.ridge_threshold).
    5. Direct Dupire pointwise formula (pulled / recomputed from
       src.real_data_v4.direct_dupire_local_vol).

PART B -- 5-FOLD SPATIAL CV on the same GP-Dupire SPY dataset:
    - Random fold assignment of (k, tau) grid points, numpy seed=42.
    - Per-fold: fit 2-term linear Dupire on 4 folds, R^2 on held-out.
    - Same fold structure for the [2,1] KAN-Dupire (train_kan_dupire_21).
    - Records per-fold activations (200-point sweep on [-1, 1]) to assess
      cross-fold consistency, complementing the seed-stability sweep.

CPU only, seed=42 throughout. Runtime budget ~5 minutes.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils import set_all_seeds, setup_logging
from src.real_data_v2 import (
    build_logm_surface_svi,
    compute_forward_prices,
    get_dividend_yield,
)
from src.real_data_v4 import (
    _central_dk,
    _r2_from,
    bs_theta_analytical,
    direct_dupire_local_vol,
    reconstruct_sigma_imp_grid,
)
from src.sindy_kan import (
    _load_spy_option_data,
    _eval_activations_on_grid,
    _prepare_spy_inputs,
    train_kan_dupire_21,
)
from src.baselines import ridge_threshold

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Shared dataset builder
# ---------------------------------------------------------------------------


def _gp_smooth_surface(C: np.ndarray, k_grid: np.ndarray,
                        tau_grid: np.ndarray, seed: int = 42) -> np.ndarray:
    """GP-smooth the C surface on the (k, tau) grid with an RBF kernel.

    Mirrors ``src.sindy_kan._smooth_surface_with_kernel(... kernel='RBF')``
    but kept here as a local helper so the discovery-baseline pipeline
    is self-contained. Falls back to the input surface on failure.
    """
    try:
        import warnings
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF, WhiteKernel, ConstantKernel,
        )
        K_mesh, T_mesh = np.meshgrid(k_grid, tau_grid, indexing='ij')
        X = np.column_stack([K_mesh.ravel(), T_mesh.ravel()])
        y = np.asarray(C, dtype=np.float64).ravel()
        ok = np.isfinite(y)
        Xf, yf = X[ok], y[ok]
        if Xf.shape[0] < 10:
            return C
        k_extent = float(k_grid[-1] - k_grid[0])
        t_extent = float(tau_grid[-1] - tau_grid[0])
        ls_init = [max(0.2 * k_extent, 1e-3), max(0.2 * t_extent, 1e-3)]
        y_var = float(np.var(yf))
        noise_init = max(1e-6, 1e-4 * y_var)
        inner = RBF(length_scale=ls_init, length_scale_bounds=(1e-3, 1e3))
        kernel = (
            ConstantKernel(constant_value=max(y_var, 1e-3),
                            constant_value_bounds=(1e-5, 1e6))
            * inner
            + WhiteKernel(noise_level=noise_init,
                           noise_level_bounds=(1e-10, 1e2))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=1,
            normalize_y=True, random_state=int(seed), alpha=0.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(Xf, yf)
        y_smooth = gp.predict(X).reshape(C.shape)
        return y_smooth
    except Exception as exc:
        logger.warning("_gp_smooth_surface failed: %s", exc)
        return C


def build_spy_gp_dupire_dataset(snapshot_path: Optional[str] = None,
                                  n_k: int = 40, n_tau: int = 15,
                                  k_range: tuple[float, float] = (-0.20, 0.20),
                                  use_gp_smoothing: bool = True,
                                  seed: int = 42) -> dict[str, Any]:
    """Build the GP-Dupire SPY dataset used by both Part A and Part B.

    Steps:
      1. Load the SPY option chain for the requested snapshot (defaults to
         20260329 if present, else the latest cached chain).
      2. Fit the SVI surface in log-moneyness.
      3. Optionally re-smooth the C surface with a GP (RBF), giving the
         'GP-Dupire' library used throughout the PRD.
      4. Compute analytical theta and (dC/dk, d2C/dk2) on the regular grid.

    Returns a dict with C, k, tau, sigma_imp, dCdk, d2Cdk2, theta, S0, r, q
    and the snapshot path used.
    """
    set_all_seeds(seed)
    if snapshot_path is None:
        # Prefer the 20260329 snapshot called out in the PRD; fall back to
        # the loader's default (latest cached file).
        candidate = os.path.join('outputs', 'tables',
                                   'real_chain_SPY_20260329.csv')
        snapshot_path = candidate if os.path.exists(candidate) else None
    od = _load_spy_option_data('SPY', csv_path=snapshot_path)
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
    if use_gp_smoothing:
        C = _gp_smooth_surface(C, k_grid, tau_grid, seed=seed)
    sigma_imp = reconstruct_sigma_imp_grid(svi_params, k_grid, tau_grid)
    F_grid = compute_forward_prices(S0, r, q, tau_grid)
    K_2d = np.outer(np.exp(k_grid), F_grid)
    theta = bs_theta_analytical(S0, K_2d, tau_grid, sigma_imp, r, q)
    dCdk, d2Cdk2 = _central_dk(C, k_grid)
    return {
        'C': C, 'k': k_grid, 'tau': tau_grid,
        'sigma_imp': sigma_imp, 'K_2d': K_2d,
        'dCdk': dCdk, 'd2Cdk2': d2Cdk2, 'theta': theta,
        'S0': S0, 'r': r, 'q': q,
        'snapshot_path': od.get('data_source'),
    }


# ---------------------------------------------------------------------------
# PART A -- discovery-method baselines on GP-Dupire
# ---------------------------------------------------------------------------


def _flat_finite(*arrs: np.ndarray) -> tuple[np.ndarray, ...]:
    """Ravel and apply a joint finiteness mask across all inputs."""
    flats = [np.asarray(a, dtype=np.float64).ravel() for a in arrs]
    ok = np.ones_like(flats[0], dtype=bool)
    for f in flats:
        ok &= np.isfinite(f)
    return tuple(f[ok] for f in flats)


def _sigma_loc_from_2term(coef_dCdk: float, coef_d2Cdk2: float
                            ) -> tuple[float, float]:
    """Extract (sigma_loc, drift) from 2-term Dupire OLS coefficients.

    theta = c1 * dC/dk + c2 * d2C/dk2 with c2 = 0.5 * sigma^2_loc.
    Returns sigma_loc = sqrt(2 * c2) (NaN if c2 <= 0) and drift = c1.
    """
    sigma_loc = float(np.sqrt(2.0 * coef_d2Cdk2)) if coef_d2Cdk2 > 0 else float('nan')
    return sigma_loc, float(coef_dCdk)


def _sigma_loc_grid_2term(dCdk: np.ndarray, d2Cdk2: np.ndarray,
                            coef_dCdk: float, coef_d2Cdk2: float,
                            theta: np.ndarray) -> np.ndarray:
    """Per-cell sigma_loc^2 inferred by point-wise inversion of the 2-term fit.

    Given c1 and c2 from the global OLS, we report the SAME global
    sigma_loc at every point (the linear model has only one diffusion
    coefficient). The grid here is just a constant array used for IQR /
    median reporting -- it matches direct Dupire's per-cell format.
    """
    val, _ = _sigma_loc_from_2term(coef_dCdk, coef_d2Cdk2)
    return np.full_like(theta, val, dtype=np.float64)


def _summarize_sigma_loc(grid: np.ndarray) -> tuple[float, float]:
    """Return (median, IQR) of finite, positive sigma_loc values.

    For a constant-sigma grid (the case for linear 2-term fits) the IQR
    is structurally zero, which is the honest report: the model has only
    one diffusion coefficient. For per-cell grids (direct Dupire formula
    or KAN-inverted sigma_loc) both summaries are meaningful.
    """
    arr = np.asarray(grid, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return float('nan'), float('nan')
    med = float(np.median(arr))
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    return med, float(q75 - q25)


def weak_form_dupire_2term(dataset: dict[str, Any],
                             n_modes_k: int = 6, n_modes_tau: int = 6,
                             seed: int = 42) -> dict[str, Any]:
    """Weak-form discovery on the GP-smoothed (k, tau) C surface.

    Uses spectral (sine) test functions in (k, tau) and integration by
    parts in the k direction, restricted to the 2-term Dupire library:

        theta = c1 * dC/dk + c2 * d2C/dk2.

    The LHS is the integrated analytical-theta target (no time
    differentiation), and the RHS columns use the analytical derivatives
    of the test functions in k to push all k-derivatives off the data.
    """
    C = np.asarray(dataset['C'], dtype=np.float64)
    theta = np.asarray(dataset['theta'], dtype=np.float64)
    k_grid = np.asarray(dataset['k'], dtype=np.float64)
    tau_grid = np.asarray(dataset['tau'], dtype=np.float64)

    k_min, k_max = float(k_grid[0]), float(k_grid[-1])
    t_min, t_max = float(tau_grid[0]), float(tau_grid[-1])
    Lk = k_max - k_min
    Lt = t_max - t_min
    k_n = (k_grid - k_min) / Lk
    t_n = (tau_grid - t_min) / Lt
    Kn_mesh, Tn_mesh = np.meshgrid(k_n, t_n, indexing='ij')

    rows_lhs: list[float] = []
    rows_rhs: list[list[float]] = []

    # Trapezoidal weights as a sanity-stable area element
    def _integrate(f: np.ndarray) -> float:
        try:
            inner = np.trapezoid(f, k_grid, axis=0)
            return float(np.trapezoid(inner, tau_grid))
        except AttributeError:
            inner = np.trapz(f, k_grid, axis=0)
            return float(np.trapz(inner, tau_grid))

    for m in range(1, int(n_modes_k) + 1):
        kM = m * np.pi
        sin_mK = np.sin(kM * Kn_mesh)
        cos_mK = np.cos(kM * Kn_mesh)
        dsin_mK = (kM / Lk) * cos_mK         # d/dk of sin
        d2sin_mK = -((kM / Lk) ** 2) * sin_mK  # d^2/dk^2 of sin
        for n in range(1, int(n_modes_tau) + 1):
            kT = n * np.pi
            sin_nT = np.sin(kT * Tn_mesh)
            phi = sin_mK * sin_nT
            dphi_dk = dsin_mK * sin_nT
            d2phi_dk2 = d2sin_mK * sin_nT

            lhs = _integrate(phi * theta)
            # IBP once: ∫ phi * dC/dk dk = -∫ dphi/dk * C dk (boundary 0)
            rhs_dCdk = -_integrate(dphi_dk * C)
            # IBP twice: ∫ phi * d2C/dk2 dk = +∫ d2phi/dk2 * C dk
            rhs_d2Cdk2 = _integrate(d2phi_dk2 * C)
            rows_lhs.append(lhs)
            rows_rhs.append([rhs_dCdk, rhs_d2Cdk2])

    LHS = np.asarray(rows_lhs, dtype=np.float64)
    RHS = np.asarray(rows_rhs, dtype=np.float64)
    cond = float(np.linalg.cond(RHS)) if RHS.size > 0 else float('inf')
    try:
        coef, *_ = np.linalg.lstsq(RHS, LHS, rcond=None)
    except Exception as exc:
        logger.warning("weak_form_dupire_2term lstsq failed: %s", exc)
        coef = np.array([0.0, 0.0])
    pred = RHS @ coef
    r2_integral = _r2_from(LHS, pred)

    # Translate back to pointwise theta R^2 for fair comparison.
    a_flat, b_flat, t_flat = _flat_finite(dataset['dCdk'], dataset['d2Cdk2'],
                                            dataset['theta'])
    A = np.column_stack([a_flat, b_flat])
    pred_pw = A @ coef
    r2_pointwise = _r2_from(t_flat, pred_pw)

    sigma_loc, drift = _sigma_loc_from_2term(float(coef[0]), float(coef[1]))
    sigma_grid = _sigma_loc_grid_2term(dataset['dCdk'], dataset['d2Cdk2'],
                                         float(coef[0]), float(coef[1]),
                                         dataset['theta'])
    sigma_med, sigma_iqr = _summarize_sigma_loc(sigma_grid)
    return {
        'method': 'weak_form_2term',
        'coef_dCdk': float(coef[0]),
        'coef_d2Cdk2': float(coef[1]),
        'sigma_loc': float(sigma_loc),
        'drift': float(drift),
        'sigma_loc_median': sigma_med,
        'sigma_loc_iqr': sigma_iqr,
        'r2_integral': float(r2_integral),
        'r2_pointwise': float(r2_pointwise),
        'condition_number': cond,
        'n_test_functions': int(LHS.size),
    }


def ridge_threshold_dupire_2term(dataset: dict[str, Any]) -> dict[str, Any]:
    """Ridge + threshold on the GP-derived 2-term Dupire library."""
    a, b, t = _flat_finite(dataset['dCdk'], dataset['d2Cdk2'], dataset['theta'])
    library = np.column_stack([a, b])
    res = ridge_threshold(library, t)
    coef = res['coefficients']
    pred = library @ coef
    r2 = _r2_from(t, pred)
    sigma_loc, drift = _sigma_loc_from_2term(float(coef[0]), float(coef[1]))
    sigma_grid = _sigma_loc_grid_2term(dataset['dCdk'], dataset['d2Cdk2'],
                                         float(coef[0]), float(coef[1]),
                                         dataset['theta'])
    sigma_med, sigma_iqr = _summarize_sigma_loc(sigma_grid)
    return {
        'method': 'ridge_threshold_2term',
        'coef_dCdk': float(coef[0]),
        'coef_d2Cdk2': float(coef[1]),
        'sigma_loc': float(sigma_loc),
        'drift': float(drift),
        'sigma_loc_median': sigma_med,
        'sigma_loc_iqr': sigma_iqr,
        'r2': float(r2),
        'best_threshold': float(res.get('best_threshold', float('nan'))),
        'ridge_alpha': float(res.get('best_ridge_alpha', float('nan'))),
        'n_active': int(res.get('n_active', 0)),
    }


def stlsq_dupire_2term_baseline(dataset: dict[str, Any]) -> dict[str, Any]:
    """OLS 2-term Dupire used as STLSQ analogue when only 2 terms exist.

    With only 2 candidate columns, STLSQ with a low threshold reduces to
    OLS; we report OLS here as the apples-to-apples baseline. The 'true'
    multi-term STLSQ result is pulled from the existing CSV in
    ``run_discovery_method_comparison``.
    """
    a, b, t = _flat_finite(dataset['dCdk'], dataset['d2Cdk2'], dataset['theta'])
    library = np.column_stack([a, b])
    coef, *_ = np.linalg.lstsq(library, t, rcond=None)
    pred = library @ coef
    r2 = _r2_from(t, pred)
    sigma_loc, drift = _sigma_loc_from_2term(float(coef[0]), float(coef[1]))
    sigma_grid = _sigma_loc_grid_2term(dataset['dCdk'], dataset['d2Cdk2'],
                                         float(coef[0]), float(coef[1]),
                                         dataset['theta'])
    sigma_med, sigma_iqr = _summarize_sigma_loc(sigma_grid)
    return {
        'method': 'stlsq_ols_2term',
        'coef_dCdk': float(coef[0]),
        'coef_d2Cdk2': float(coef[1]),
        'sigma_loc': float(sigma_loc),
        'drift': float(drift),
        'sigma_loc_median': sigma_med,
        'sigma_loc_iqr': sigma_iqr,
        'r2': float(r2),
    }


def direct_dupire_baseline(dataset: dict[str, Any]) -> dict[str, Any]:
    """Direct Dupire pointwise formula on the GP-smoothed C surface."""
    res = direct_dupire_local_vol(
        dataset['C'], dataset['sigma_imp'],
        dataset['k'], dataset['tau'],
        dataset['S0'], dataset['r'], dataset['q'],
    )
    sigma_grid = res['sigma_loc_grid']
    sigma_med, sigma_iqr = _summarize_sigma_loc(sigma_grid)
    return {
        'method': 'direct_dupire',
        'sigma_loc_median': sigma_med,
        'sigma_loc_iqr': sigma_iqr,
        'n_valid_pct': float(res['n_valid_pct']),
    }


def _pull_existing_stlsq_row(ticker: str = 'SPY') -> dict[str, Any]:
    """Pull the SINDy linear-Dupire (STLSQ-equivalent) row from the v4 CSV.

    Prefers ``outputs/tables/sindy_kan_dupire_real.csv`` (column
    ``linear_dupire_r2``); falls back to ``gp_dupire_real_comparison.csv``
    (column ``r2_gp``).
    """
    csv1 = os.path.join('outputs', 'tables', 'sindy_kan_dupire_real.csv')
    csv2 = os.path.join('outputs', 'tables', 'gp_dupire_real_comparison.csv')
    try:
        if os.path.exists(csv1):
            df = pd.read_csv(csv1)
            row = df[df['ticker'] == ticker].iloc[0]
            return {
                'method': 'STLSQ (existing GP-Dupire, 2-term OLS)',
                'R2_SPY': float(row.get('linear_dupire_r2', float('nan'))),
                'source': os.path.basename(csv1),
            }
    except Exception as exc:
        logger.warning("Could not load STLSQ row from %s: %s", csv1, exc)
    try:
        df = pd.read_csv(csv2)
        row = df[df['ticker'] == ticker].iloc[0]
        return {
            'method': 'STLSQ (existing GP-Dupire, 2-term OLS)',
            'R2_SPY': float(row.get('r2_gp', float('nan'))),
            'sigma_loc': float(row.get('sigma_gp', float('nan'))),
            'source': os.path.basename(csv2),
        }
    except Exception as exc:
        logger.warning("Could not load STLSQ fallback from %s: %s", csv2, exc)
    return {'method': 'STLSQ (existing GP-Dupire, 2-term OLS)',
            'R2_SPY': float('nan'), 'source': 'missing'}


def _pull_existing_kan_row(ticker: str = 'SPY') -> dict[str, Any]:
    csv1 = os.path.join('outputs', 'tables', 'sindy_kan_dupire_real.csv')
    try:
        df = pd.read_csv(csv1)
        row = df[df['ticker'] == ticker].iloc[0]
        return {
            'method': 'STLSQ + KAN [2,1] (existing)',
            'R2_SPY': float(row.get('kan_test_r2', float('nan'))),
            'sigma_loc_median': float(row.get('sigma_loc_median', float('nan'))),
            'source': os.path.basename(csv1),
        }
    except Exception as exc:
        logger.warning("Could not load KAN row: %s", exc)
    return {'method': 'STLSQ + KAN [2,1] (existing)',
            'R2_SPY': float('nan'), 'source': 'missing'}


def run_discovery_method_comparison(
        dataset: Optional[dict[str, Any]] = None,
        save_csv: str = 'outputs/tables/discovery_method_comparison.csv',
        ticker: str = 'SPY') -> pd.DataFrame:
    """Run all 5 baselines and assemble the comparison table.

    The STLSQ and STLSQ+KAN rows are pulled from existing CSVs as
    instructed by the PRD; weak-form, Ridge+threshold and direct Dupire
    are computed fresh from the GP-Dupire dataset built here.
    """
    if dataset is None:
        dataset = build_spy_gp_dupire_dataset()

    # 1+2: pulled rows.
    row_stlsq = _pull_existing_stlsq_row(ticker)
    row_kan = _pull_existing_kan_row(ticker)

    # Recompute an in-house OLS sigma_loc on the same GP-Dupire library
    # so the STLSQ row has a sigma_loc figure even when the source CSV
    # doesn't carry one.
    ols = stlsq_dupire_2term_baseline(dataset)
    row_stlsq.setdefault('sigma_loc_median', ols['sigma_loc_median'])
    row_stlsq.setdefault('sigma_loc_iqr', ols['sigma_loc_iqr'])

    # 3: weak-form.
    weak = weak_form_dupire_2term(dataset)
    row_weak = {
        'method': 'Weak-form regression (2-term, GP surface)',
        'R2_SPY': weak['r2_pointwise'],
        'sigma_loc_median': weak['sigma_loc_median'],
        'sigma_loc_iqr': weak['sigma_loc_iqr'],
        'source': 'weak_form_dupire_2term',
    }
    # 4: ridge + threshold.
    ridge = ridge_threshold_dupire_2term(dataset)
    row_ridge = {
        'method': 'Ridge + threshold (2-term, GP)',
        'R2_SPY': ridge['r2'],
        'sigma_loc_median': ridge['sigma_loc_median'],
        'sigma_loc_iqr': ridge['sigma_loc_iqr'],
        'source': 'ridge_threshold',
    }
    # 5: direct Dupire formula.
    direct = direct_dupire_baseline(dataset)
    row_direct = {
        'method': 'Direct Dupire formula (pointwise)',
        'R2_SPY': float('nan'),  # n/a -- no regression
        'sigma_loc_median': direct['sigma_loc_median'],
        'sigma_loc_iqr': direct['sigma_loc_iqr'],
        'source': 'direct_dupire_local_vol',
    }

    weak_diff_note = (
        f"c2={weak['coef_d2Cdk2']:.3g}; "
        + ('positive -> sigma_loc reported' if weak['coef_d2Cdk2'] > 0
            else 'sigma_loc undefined (c2<=0)')
    )
    ridge_diff_note = (
        f"c2={ridge['coef_d2Cdk2']:.3g}; "
        + ('positive -> sigma_loc reported' if ridge['coef_d2Cdk2'] > 0
            else 'sigma_loc undefined (c2 thresholded to 0)')
    )
    rows = []
    for r, derivs, interp, notes in [
        (row_stlsq, 'GP', 'coefficients',
         'STLSQ on 2-term GP-Dupire library (existing). sigma_loc from in-house OLS.'),
        (row_kan, 'GP', 'activations',
         '[2,1] KAN-Dupire (existing). sigma_loc median over per-cell KAN-inverted grid.'),
        (row_weak, 'GP', 'integrals',
         f"Spectral sine basis, n_test={weak['n_test_functions']}, "
         f"cond={weak['condition_number']:.2e}. {weak_diff_note}"),
        (row_ridge, 'GP', 'coefficients',
         f"RidgeCV + hard threshold (alpha={ridge['ridge_alpha']:.2e}, "
         f"thr={ridge['best_threshold']:.3g}). {ridge_diff_note}"),
        (row_direct, 'FD on GP surface', 'pointwise',
         f"Per-cell formula; "
         f"{direct['n_valid_pct'] * 100.0:.1f}% valid cells. "
         "IQR captures the cross-cell dispersion."),
    ]:
        rows.append({
            'method': r['method'],
            'derivatives': derivs,
            'R2_SPY': r.get('R2_SPY', float('nan')),
            'sigma_loc_median': r.get('sigma_loc_median', float('nan')),
            'sigma_loc_iqr': r.get('sigma_loc_iqr', float('nan')),
            'interpretability': interp,
            'notes': notes,
        })
    df = pd.DataFrame(rows)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        df.to_csv(save_csv, index=False)
    return df


# ---------------------------------------------------------------------------
# PART B -- 5-fold spatial CV
# ---------------------------------------------------------------------------


def _assign_folds(n: int, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    """Random fold assignment in [0, n_folds) for ``n`` points, seed-fixed."""
    rng = np.random.default_rng(int(seed))
    base = np.tile(np.arange(int(n_folds)),
                    int(np.ceil(n / float(n_folds))))[:n]
    rng.shuffle(base)
    return base.astype(np.int64)


def _r2(target: np.ndarray, pred: np.ndarray) -> float:
    return _r2_from(np.asarray(target, dtype=np.float64),
                    np.asarray(pred, dtype=np.float64))


def cv_linear_dupire(dataset: dict[str, Any], n_folds: int = 5,
                       seed: int = 42) -> pd.DataFrame:
    """5-fold spatial CV for the 2-term OLS Dupire fit."""
    a, b, t = _flat_finite(dataset['dCdk'], dataset['d2Cdk2'], dataset['theta'])
    library = np.column_stack([a, b])
    n = library.shape[0]
    folds = _assign_folds(n, n_folds=n_folds, seed=seed)
    rows: list[dict[str, Any]] = []
    test_r2s: list[float] = []
    for k in range(int(n_folds)):
        test_mask = folds == k
        train_mask = ~test_mask
        if train_mask.sum() < 5 or test_mask.sum() < 2:
            continue
        X_tr, y_tr = library[train_mask], t[train_mask]
        X_te, y_te = library[test_mask], t[test_mask]
        coef, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        r2_train = _r2(y_tr, X_tr @ coef)
        r2_test = _r2(y_te, X_te @ coef)
        test_r2s.append(r2_test)
        rows.append({
            'model': 'linear_dupire_2term',
            'fold': int(k),
            'R2_train': float(r2_train),
            'R2_test': float(r2_test),
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
            'coef_dCdk': float(coef[0]),
            'coef_d2Cdk2': float(coef[1]),
        })
    mean_r2 = float(np.mean(test_r2s)) if test_r2s else float('nan')
    std_r2 = float(np.std(test_r2s)) if test_r2s else float('nan')
    for r in rows:
        r['mean_R2_test'] = mean_r2
        r['std_R2_test'] = std_r2
    return pd.DataFrame(rows)


def _standardize_apply(x: np.ndarray, center: np.ndarray,
                         rng: np.ndarray) -> np.ndarray:
    """Apply (x - center)/(0.5 * range) per column to match _standardize_to_unit."""
    return (2.0 * (np.asarray(x, dtype=np.float64) - center) / rng).astype(np.float32)


def cv_kan_dupire(dataset: dict[str, Any], n_folds: int = 5,
                    seed: int = 42, n_epochs: int = 1500,
                    activations_n_eval: int = 200
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """5-fold spatial CV for the [2,1] KAN-Dupire and per-fold activations.

    Returns
    -------
    (cv_df, act_df)
        cv_df : per-fold R^2 / sizes.
        act_df: per-fold activation sweep on standardized [-1, 1] grid
                with ``activations_n_eval`` points per edge.
    """
    import torch

    a, b, t = _flat_finite(dataset['dCdk'], dataset['d2Cdk2'], dataset['theta'])
    library = np.column_stack([a, b])
    n = library.shape[0]
    folds = _assign_folds(n, n_folds=n_folds, seed=seed)
    rows: list[dict[str, Any]] = []
    act_rows: list[dict[str, Any]] = []
    test_r2s: list[float] = []
    for k in range(int(n_folds)):
        test_mask = folds == k
        train_mask = ~test_mask
        if train_mask.sum() < 20 or test_mask.sum() < 2:
            continue
        try:
            # Train on a 2D grid of the training subset.  train_kan_dupire_21
            # expects (n_k, n_tau) shaped inputs but happily accepts flat
            # arrays (it ravels and masks internally).
            res = train_kan_dupire_21(
                a[train_mask], b[train_mask], t[train_mask],
                seed=int(seed), n_epochs=int(n_epochs),
            )
            xp = res['x_std_params']
            yp = res['y_std_params']
            model = res['model']

            # Apply same standardization to the test fold.
            X_test_raw = library[test_mask]
            X_test_std = _standardize_apply(X_test_raw, xp['center'], xp['range'])
            X_test_t = torch.tensor(X_test_std, dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                pred_std = model(X_test_t).numpy()
            pred = pred_std * yp['std'] + yp['mean']
            r2_train_kan = float(res['train_r2'])
            r2_test = _r2(t[test_mask], pred)
            test_r2s.append(r2_test)
            rows.append({
                'model': 'kan_dupire_21',
                'fold': int(k),
                'R2_train': r2_train_kan,
                'R2_test': float(r2_test),
                'n_train': int(train_mask.sum()),
                'n_test': int(test_mask.sum()),
            })
            # Per-fold activation summary.
            x_eval, ydrift, ydiff = _eval_activations_on_grid(
                model, n_eval=int(activations_n_eval))
            for i, xv in enumerate(x_eval):
                act_rows.append({
                    'fold': int(k), 'edge_idx': 0,
                    'x_eval': float(xv),
                    'activation': float(ydrift[i]),
                })
                act_rows.append({
                    'fold': int(k), 'edge_idx': 1,
                    'x_eval': float(xv),
                    'activation': float(ydiff[i]),
                })
        except Exception as exc:
            logger.warning("cv_kan_dupire fold %d failed: %s", k, exc)
    mean_r2 = float(np.mean(test_r2s)) if test_r2s else float('nan')
    std_r2 = float(np.std(test_r2s)) if test_r2s else float('nan')
    for r in rows:
        r['mean_R2_test'] = mean_r2
        r['std_R2_test'] = std_r2
    return pd.DataFrame(rows), pd.DataFrame(act_rows)


def run_cv(dataset: Optional[dict[str, Any]] = None,
             n_folds: int = 5, seed: int = 42, n_epochs: int = 1500,
             save_csv: str = 'outputs/tables/cv_results.csv',
             save_act_csv: str = 'outputs/tables/cv_kan_activations_summary.csv'
             ) -> dict[str, pd.DataFrame]:
    """Run 5-fold CV for linear and KAN Dupire and persist results."""
    if dataset is None:
        dataset = build_spy_gp_dupire_dataset()
    lin_df = cv_linear_dupire(dataset, n_folds=n_folds, seed=seed)
    kan_df, act_df = cv_kan_dupire(dataset, n_folds=n_folds, seed=seed,
                                     n_epochs=n_epochs)
    combined = pd.concat([lin_df, kan_df], ignore_index=True, sort=False)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        combined.to_csv(save_csv, index=False)
    if save_act_csv:
        os.makedirs(os.path.dirname(save_act_csv), exist_ok=True)
        act_df.to_csv(save_act_csv, index=False)
    return {'linear': lin_df, 'kan': kan_df, 'activations': act_df,
            'combined': combined}
