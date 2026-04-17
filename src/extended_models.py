"""
Extended model experiments: Merton jump-diffusion and Heston stochastic volatility.
Tests SINDy on data generated from more complex dynamics than Black-Scholes.
"""

import numpy as np
from math import sqrt

from src.utils import set_all_seeds, setup_logging, Timer, safe_relative_error
from src.sindy_discovery import (
    discover_pde, TERM_NAMES, compute_derivatives, build_candidate_library
)
from src.data_generation import (
    generate_price_surface, generate_merton_surface, bs_call_price
)

logger = setup_logging(__name__)


def run_merton_experiment(lam=0.1, mu_J=-0.05, sigma_J=0.1, sigma=0.2,
                          r=0.05, K=100, T=1.0):
    """
    Run SINDy PDE discovery on Merton jump-diffusion price data.

    Generates a Merton jump-diffusion price surface, applies SINDy to discover
    the governing PDE, and performs residual analysis to identify where the
    Black-Scholes PDE approximation breaks down.

    Parameters
    ----------
    lam : float
        Jump intensity (average number of jumps per year).
    mu_J : float
        Mean of log-jump size.
    sigma_J : float
        Std deviation of log-jump size.
    sigma : float
        Diffusion volatility.
    r : float
        Risk-free rate.
    K : float
        Strike price.
    T : float
        Option maturity.

    Returns
    -------
    dict
        discovered_coefficients : ndarray
            SINDy-discovered PDE coefficients.
        true_bs_coefficients : ndarray
            True Black-Scholes PDE coefficients for comparison.
        r2 : float
            R-squared score of the SINDy fit.
        human_readable_pde : str
            Discovered PDE as a human-readable string.
        active_terms : list of str
            Names of active (non-zero) terms in the discovered PDE.
        residual_grid : ndarray
            PDE residuals reshaped to the trimmed grid.
        S_grid : ndarray
            Stock price grid points.
        t_grid : ndarray
            Time grid points.
        V_merton : ndarray
            Merton jump-diffusion price surface.
        params : dict
            Input parameters for reproducibility.
    """
    set_all_seeds(42)

    logger.info(
        f"Running Merton experiment: lam={lam}, mu_J={mu_J}, "
        f"sigma_J={sigma_J}, sigma={sigma}, r={r}"
    )

    # Generate Merton surface
    with Timer("Merton surface generation"):
        V_merton, S_grid, t_grid = generate_merton_surface(
            S_min=50, S_max=150, n_S=100, t_min=0.0, n_t=100,
            K=K, r=r, sigma=sigma, T=T,
            lam=lam, mu_J=mu_J, sigma_J=sigma_J
        )

    # Run SINDy discovery on the Merton surface
    with Timer("SINDy discovery on Merton data"):
        sindy_result = discover_pde(
            V_merton, S_grid, t_grid,
            true_sigma=sigma, true_r=r,
            K=K, T=T, option_type='call'
        )

    discovered_coeffs = sindy_result['discovered_coefficients']

    # True BS coefficients (what a pure BS model would give)
    true_bs_coefficients = np.array([
        r,                        # V
        0.0,                      # dV/dS
        0.0,                      # d2V/dS2
        -r,                       # S*dV/dS
        -0.5 * sigma ** 2,        # S^2*d2V/dS2
    ])

    # Residual analysis: compute dV/dt - library @ coefficients on the trimmed grid
    with Timer("Residual analysis"):
        derivs = compute_derivatives(V_merton, S_grid, t_grid, trim=5)
        library = build_candidate_library(
            derivs['V'], derivs['dVdS'], derivs['d2VdS2'], derivs['S_mesh']
        )
        target = derivs['dVdt'].ravel()

        # Residuals: dV/dt - Theta @ xi
        residuals = target - library @ discovered_coeffs
        residual_grid = residuals.reshape(derivs['V'].shape)

    max_res_idx = np.unravel_index(np.argmax(np.abs(residual_grid)), residual_grid.shape)
    logger.info(
        f"Max |residual| = {np.abs(residual_grid).max():.6e} at grid index {max_res_idx} "
        f"(S={derivs['S_grid'][max_res_idx[0]]:.1f}, t={derivs['t_grid'][max_res_idx[1]]:.4f})"
    )
    logger.info(f"Discovered PDE: {sindy_result['human_readable_pde']}")
    logger.info(f"R^2 = {sindy_result['r2_score']:.6f}")

    params = {
        'lam': lam, 'mu_J': mu_J, 'sigma_J': sigma_J,
        'sigma': sigma, 'r': r, 'K': K, 'T': T,
    }

    return {
        'discovered_coefficients': discovered_coeffs,
        'true_bs_coefficients': true_bs_coefficients,
        'r2': sindy_result['r2_score'],
        'human_readable_pde': sindy_result['human_readable_pde'],
        'active_terms': sindy_result['active_terms'],
        'residual_grid': residual_grid,
        'S_grid': S_grid,
        't_grid': t_grid,
        'V_merton': V_merton,
        'params': params,
    }


def run_heston_variance_slicing(v_list=None, r=0.05, K=100, T=1.0):
    """
    Test SINDy discovery across multiple variance levels (Heston-style slicing).

    For each variance level v, generates a Black-Scholes surface with
    sigma = sqrt(v), runs SINDy, and collects the discovered diffusion
    coefficient (S^2 * d2V/dS2 term). The true coefficient is -v/2.
    Tests whether the discovered diffusion coefficient scales linearly with v.

    Parameters
    ----------
    v_list : list of float or None
        Variance levels to test. If None, uses [0.01, 0.02, 0.04, 0.08, 0.16].
    r : float
        Risk-free rate.
    K : float
        Strike price.
    T : float
        Option maturity.

    Returns
    -------
    dict
        v_list : list of float
            Variance levels tested.
        sigma_list : list of float
            Corresponding volatilities (sqrt of each v).
        discovered_diffusion_coeffs : list of float
            Discovered S^2*d2V/dS2 coefficients at each variance level.
        true_diffusion_coeffs : list of float
            True diffusion coefficients (-v/2) at each variance level.
        per_slice_results : list of dict
            Full discover_pde results for each variance level.
        linearity_r2 : float
            R-squared of linear fit of discovered coefficients vs v.
        linear_fit_slope : float
            Slope of the linear fit (should be close to -0.5).
        linear_fit_intercept : float
            Intercept of the linear fit (should be close to 0).
    """
    set_all_seeds(42)

    if v_list is None:
        v_list = [0.01, 0.02, 0.04, 0.08, 0.16]

    sigma_list = [sqrt(v) for v in v_list]

    logger.info(
        f"Running Heston variance slicing: v_list={v_list}, r={r}, K={K}, T={T}"
    )

    discovered_diffusion_coeffs = []
    true_diffusion_coeffs = [-v / 2.0 for v in v_list]
    per_slice_results = []

    for i, (v, sigma) in enumerate(zip(v_list, sigma_list)):
        logger.info(f"Slice {i+1}/{len(v_list)}: v={v:.4f}, sigma={sigma:.4f}")

        with Timer(f"Slice v={v:.4f}"):
            # Generate BS surface at this volatility
            V, S_grid, t_grid = generate_price_surface(
                S_min=50, S_max=150, n_S=100, n_t=100,
                K=K, r=r, sigma=sigma, T=T, option_type='call'
            )

            # Run SINDy
            result = discover_pde(
                V, S_grid, t_grid,
                true_sigma=sigma, true_r=r,
                K=K, T=T, option_type='call'
            )

        # The S^2*d2V/dS2 coefficient is index 4 in the 5-term library
        diffusion_coeff = result['discovered_coefficients'][4]
        discovered_diffusion_coeffs.append(diffusion_coeff)
        per_slice_results.append(result)

        logger.info(
            f"  Discovered S^2*d2V/dS2 coeff: {diffusion_coeff:.6f}, "
            f"true: {-v/2:.6f}"
        )

    # Fit a line to discovered_diffusion_coeff vs v to test linearity
    v_arr = np.array(v_list)
    d_arr = np.array(discovered_diffusion_coeffs)

    # Linear fit: d = slope * v + intercept
    coeffs_fit = np.polyfit(v_arr, d_arr, 1)
    linear_fit_slope = coeffs_fit[0]
    linear_fit_intercept = coeffs_fit[1]

    # R^2 of the linear fit
    d_predicted = np.polyval(coeffs_fit, v_arr)
    ss_res = np.sum((d_arr - d_predicted) ** 2)
    ss_tot = np.sum((d_arr - np.mean(d_arr)) ** 2)
    linearity_r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    logger.info(
        f"Linearity analysis: slope={linear_fit_slope:.6f} (true: -0.5), "
        f"intercept={linear_fit_intercept:.6f} (true: 0.0), "
        f"R^2={linearity_r2:.6f}"
    )

    return {
        'v_list': v_list,
        'sigma_list': sigma_list,
        'discovered_diffusion_coeffs': discovered_diffusion_coeffs,
        'true_diffusion_coeffs': true_diffusion_coeffs,
        'per_slice_results': per_slice_results,
        'linearity_r2': linearity_r2,
        'linear_fit_slope': linear_fit_slope,
        'linear_fit_intercept': linear_fit_intercept,
    }
