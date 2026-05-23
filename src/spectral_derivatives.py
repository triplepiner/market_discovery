"""
Spectral (FFT-based) derivative estimation for SINDy PDE discovery.

Computes dV/dt, dV/dS, d2V/dS2 via FFT differentiation with mirror-padding
to handle non-periodic boundaries and an optional dealiasing low-pass filter
(2/3 rule) to suppress high-frequency noise.

Spectral derivatives are exact (to machine precision) for band-limited signals
on periodic domains. Mirror-padding extends the signal evenly so that the
implicit FFT periodicity does not introduce a boundary discontinuity.
"""

import time

import numpy as np
import pandas as pd

from src.utils import set_all_seeds, setup_logging, safe_relative_error
from src.data_generation import generate_price_surface, add_noise
from src.sindy_discovery import (
    build_candidate_library,
    stlsq_sweep,
    format_pde_string,
    compute_r2_clean,
    compute_coefficient_metrics,
    TERM_NAMES,
)

logger = setup_logging(__name__)


def _mirror_pad(arr, axis):
    """
    Mirror-extend an array along ``axis`` so the result has 2 * N samples
    that wrap continuously: [a, b, c] -> [a, b, c, c, b, a].
    """
    flipped = np.flip(arr, axis=axis)
    return np.concatenate([arr, flipped], axis=axis)


def _crop(arr, axis, N):
    """Inverse of _mirror_pad: keep the first N samples along ``axis``."""
    slicer = [slice(None)] * arr.ndim
    slicer[axis] = slice(0, N)
    return arr[tuple(slicer)]


def _spectral_derivative_1d_axis(V_padded, dx_padded, axis, order=1,
                                   dealias_cutoff=2.0 / 3.0):
    """
    Compute derivative along ``axis`` of a (mirror-padded) array using FFT.

    Applies a low-pass filter zeroing |k| > dealias_cutoff * Nyquist.
    """
    N = V_padded.shape[axis]
    # Wavenumbers for periodic FFT on length L = N * dx_padded
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx_padded)
    # Low-pass mask in frequency space
    k_nyq = np.pi / dx_padded  # Nyquist
    mask = np.abs(k) <= dealias_cutoff * k_nyq

    # Build broadcastable shape for k and mask
    shape = [1] * V_padded.ndim
    shape[axis] = N
    k_b = k.reshape(shape)
    mask_b = mask.reshape(shape)

    Vhat = np.fft.fft(V_padded, axis=axis)
    Vhat = Vhat * mask_b

    if order == 1:
        Dhat = (1j * k_b) * Vhat
    elif order == 2:
        Dhat = (-(k_b ** 2)) * Vhat
    else:
        raise ValueError(f"order must be 1 or 2, got {order}")

    D = np.fft.ifft(Dhat, axis=axis).real
    return D


def compute_spectral_derivatives_periodic(V, dS, dt, dealias_cutoff=2.0 / 3.0):
    """
    Spectral derivatives assuming V is already periodic in both axes.

    Useful for testing on intrinsically periodic test signals (e.g. sin),
    where mirror padding would otherwise introduce a C^1 seam.
    """
    dVdS = _spectral_derivative_1d_axis(
        V, dS, axis=0, order=1, dealias_cutoff=dealias_cutoff
    )
    d2VdS2 = _spectral_derivative_1d_axis(
        V, dS, axis=0, order=2, dealias_cutoff=dealias_cutoff
    )
    dVdt = _spectral_derivative_1d_axis(
        V, dt, axis=1, order=1, dealias_cutoff=dealias_cutoff
    )
    return {'dV_dt': dVdt, 'dV_dS': dVdS, 'd2V_dS2': d2VdS2}


def compute_spectral_derivatives(V, S_grid, t_grid, dealias_cutoff=2.0 / 3.0):
    """
    Compute dV/dt, dV/dS, d2V/dS2 via FFT with mirror-padding.

    Parameters
    ----------
    V : ndarray, shape (n_S, n_t)
    S_grid, t_grid : 1D arrays (uniform spacing assumed)
    dealias_cutoff : float
        Fraction of Nyquist above which Fourier modes are zeroed.

    Returns
    -------
    dict with keys 'dV_dt', 'dV_dS', 'd2V_dS2', each shape (n_S, n_t).
    """
    n_S, n_t = V.shape
    dS = float(S_grid[1] - S_grid[0])
    dt = float(t_grid[1] - t_grid[0])

    # Mirror-pad along S (axis 0), then along t (axis 1)
    V_padS = _mirror_pad(V, axis=0)            # (2*n_S, n_t)
    V_padSt = _mirror_pad(V_padS, axis=1)      # (2*n_S, 2*n_t)

    # dV/dS: first derivative along axis 0 (uses both S and t padding)
    dVdS_pad = _spectral_derivative_1d_axis(
        V_padSt, dS, axis=0, order=1, dealias_cutoff=dealias_cutoff
    )
    # d2V/dS2: second derivative along axis 0
    d2VdS2_pad = _spectral_derivative_1d_axis(
        V_padSt, dS, axis=0, order=2, dealias_cutoff=dealias_cutoff
    )
    # dV/dt: first derivative along axis 1
    dVdt_pad = _spectral_derivative_1d_axis(
        V_padSt, dt, axis=1, order=1, dealias_cutoff=dealias_cutoff
    )

    # Crop back: first n_S along axis 0, first n_t along axis 1
    dVdS = _crop(_crop(dVdS_pad, axis=0, N=n_S), axis=1, N=n_t)
    d2VdS2 = _crop(_crop(d2VdS2_pad, axis=0, N=n_S), axis=1, N=n_t)
    dVdt = _crop(_crop(dVdt_pad, axis=0, N=n_S), axis=1, N=n_t)

    return {
        'dV_dt': dVdt,
        'dV_dS': dVdS,
        'd2V_dS2': d2VdS2,
    }


def sindy_with_spectral_derivatives(V, S_grid, t_grid, threshold=0.1,
                                      trim=5, K=100, r=0.05, sigma=0.2, T=1.0,
                                      option_type='call', true_sigma=None,
                                      true_r=None, dealias_cutoff=2.0 / 3.0,
                                      seed=42):
    """
    Run SINDy PDE discovery using spectral derivatives.

    Returns the same dict structure as ``discover_pde`` plus ``r2_clean``.
    """
    set_all_seeds(seed)

    derivs = compute_spectral_derivatives(V, S_grid, t_grid,
                                          dealias_cutoff=dealias_cutoff)

    s = slice(trim, -trim) if trim > 0 else slice(None)
    V_tr = V[s, s]
    dVdt_tr = derivs['dV_dt'][s, s]
    dVdS_tr = derivs['dV_dS'][s, s]
    d2VdS2_tr = derivs['d2V_dS2'][s, s]

    S_tr = S_grid[s]
    t_tr = t_grid[s]
    S_mesh_tr, _ = np.meshgrid(S_tr, t_tr, indexing='ij')

    library = build_candidate_library(V_tr, dVdS_tr, d2VdS2_tr, S_mesh_tr)
    target = dVdt_tr.ravel()
    cond_number = float(np.linalg.cond(library))

    thresholds = np.sort(np.unique(np.concatenate([
        np.logspace(-3, np.log10(2.0), 30),
        np.linspace(0.001, 0.1, 20),
        np.array([threshold]),
    ])))
    best, sweep_results = stlsq_sweep(library, target, thresholds=thresholds)
    discovered = best['coefficients']

    if true_sigma is None:
        true_sigma = sigma
    if true_r is None:
        true_r = r

    true_coeffs = np.array([
        true_r, 0.0, 0.0, -true_r, -0.5 * true_sigma ** 2,
    ])
    rel_errors = safe_relative_error(discovered, true_coeffs)

    try:
        r2_clean = compute_r2_clean(
            discovered, S_grid, t_grid,
            K=K, r=true_r, sigma=true_sigma, T=T,
            option_type=option_type, trim=trim,
        )
    except Exception as e:
        logger.warning(f"compute_r2_clean failed: {e}")
        r2_clean = float('nan')

    active_terms = [TERM_NAMES[i] for i in range(5) if best['active_mask'][i]]
    pde_str = format_pde_string(discovered, TERM_NAMES)

    logger.info(
        f"Spectral-SINDy: R²(noisy)={best['r2']:.6f}, R²(clean)={r2_clean:.6f}, "
        f"active={best['n_active']}, PDE: {pde_str}"
    )

    return {
        'discovered_coefficients': discovered,
        'true_coefficients': true_coeffs,
        'active_terms': active_terms,
        'term_names': TERM_NAMES,
        'relative_errors': rel_errors,
        'best_threshold': best['threshold'],
        'r2_score': best['r2'],
        'r2_clean': r2_clean,
        'bic': best['bic'],
        'condition_number': cond_number,
        'sweep_results': sweep_results,
        'human_readable_pde': pde_str,
        'active_mask': best['active_mask'],
        'n_active': best['n_active'],
        'spectral_derivatives': derivs,
    }


def run_spectral_noise_robustness(noise_levels=None, n_S=50, n_t=50, K=100,
                                    r=0.05, sigma=0.2, T=1.0, seed=42,
                                    dealias_cutoff=2.0 / 3.0):
    """
    Run spectral-based SINDy across a list of noise levels.

    Columns: noise_pct, r2_clean, r2_noisy, sigma_recovered,
             max_coeff_rel_err, runtime_s.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    V_clean, S_grid, t_grid = generate_price_surface(
        n_S=n_S, n_t=n_t, K=K, r=r, sigma=sigma, T=T,
    )

    rows = []
    for noise_pct in noise_levels:
        V_noisy = add_noise(V_clean, noise_pct, seed=seed) if noise_pct > 0 else V_clean
        t_start = time.perf_counter()
        try:
            result = sindy_with_spectral_derivatives(
                V_noisy, S_grid, t_grid,
                seed=seed, K=K, r=r, sigma=sigma, T=T,
                true_r=r, true_sigma=sigma,
                dealias_cutoff=dealias_cutoff,
            )
            runtime_s = time.perf_counter() - t_start
            cm = compute_coefficient_metrics(
                result['discovered_coefficients'],
                true_r=r, true_sigma=sigma,
            )
            c4 = float(result['discovered_coefficients'][4])
            sigma_rec = float(np.sqrt(-2.0 * c4)) if c4 < 0 else float('nan')

            rows.append({
                'noise_pct': float(noise_pct),
                'r2_clean': float(result['r2_clean']),
                'r2_noisy': float(result['r2_score']),
                'sigma_recovered': sigma_rec,
                'max_coeff_rel_err': float(cm['max_coeff_rel_error']),
                'runtime_s': float(runtime_s),
            })
        except Exception as e:
            runtime_s = time.perf_counter() - t_start
            logger.warning(
                f"Spectral-SINDy failed at noise={noise_pct}: {e}. Recording NaN."
            )
            rows.append({
                'noise_pct': float(noise_pct),
                'r2_clean': float('nan'),
                'r2_noisy': float('nan'),
                'sigma_recovered': float('nan'),
                'max_coeff_rel_err': float('nan'),
                'runtime_s': float(runtime_s),
            })

    df = pd.DataFrame(rows)
    logger.info(f"Spectral noise robustness sweep:\n{df.to_string(index=False)}")
    return df
