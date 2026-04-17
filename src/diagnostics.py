"""
Diagnostic checks for scientific rigor in the BS PDE Discovery pipeline.

Provides data-leakage detection, overfitting analysis, PDE residual checks,
convergence monitoring, numerical derivative validation, SINDy sparsity
stability via bootstrap, PINN generalization testing, and monotonicity/convexity
verification.
"""

import numpy as np
import torch
from src.utils import set_all_seeds, setup_logging

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# 1. Data leakage check
# ---------------------------------------------------------------------------

def check_data_leakage(train_indices, val_indices, test_indices, n_total):
    """
    Assert that train/val/test index sets are disjoint and cover all data points.

    Parameters
    ----------
    train_indices : array-like
        Indices assigned to the training set.
    val_indices : array-like
        Indices assigned to the validation set.
    test_indices : array-like
        Indices assigned to the test set.
    n_total : int
        Total number of data points expected.

    Raises
    ------
    AssertionError
        If sets overlap or do not cover all points.
    """
    train_set = set(np.asarray(train_indices).ravel())
    val_set = set(np.asarray(val_indices).ravel())
    test_set = set(np.asarray(test_indices).ravel())

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    assert len(train_val_overlap) == 0, (
        f"Train-val overlap: {len(train_val_overlap)} indices in common"
    )
    assert len(train_test_overlap) == 0, (
        f"Train-test overlap: {len(train_test_overlap)} indices in common"
    )
    assert len(val_test_overlap) == 0, (
        f"Val-test overlap: {len(val_test_overlap)} indices in common"
    )

    union = train_set | val_set | test_set
    assert union == set(range(n_total)), (
        f"Index sets do not cover all {n_total} points. "
        f"Union has {len(union)} elements, missing {n_total - len(union)}."
    )

    n_train = len(train_set)
    n_val = len(val_set)
    n_test = len(test_set)

    logger.info("Data leakage check PASSED.")
    logger.info(
        f"  Train: {n_train} ({n_train / n_total:.1%}), "
        f"Val: {n_val} ({n_val / n_total:.1%}), "
        f"Test: {n_test} ({n_test / n_total:.1%})"
    )
    print(f"Data leakage check PASSED.")
    print(f"  Train: {n_train} ({n_train / n_total:.1%})")
    print(f"  Val:   {n_val} ({n_val / n_total:.1%})")
    print(f"  Test:  {n_test} ({n_test / n_total:.1%})")
    print(f"  Total: {n_train + n_val + n_test} / {n_total}")


# ---------------------------------------------------------------------------
# 2. Overfitting check
# ---------------------------------------------------------------------------

def check_overfitting(train_losses, val_losses, window=5):
    """
    Analyze training and validation loss curves for signs of overfitting.

    Overfitting is detected when the validation loss is increasing while
    the training loss is still decreasing, or when the ratio of validation
    loss to training loss exceeds a threshold.

    Parameters
    ----------
    train_losses : array-like
        Per-epoch training losses.
    val_losses : array-like
        Per-epoch validation losses.
    window : int
        Smoothing window for trend detection.

    Returns
    -------
    dict
        'is_overfitting' : bool
        'overfit_ratio' : float
            val_loss / train_loss at the final epoch.
        'val_increasing' : bool
            Whether the validation loss is trending upward at the end.
        'best_epoch' : int
            Epoch with the lowest validation loss.
        'recommendation' : str
    """
    train_losses = np.asarray(train_losses, dtype=np.float64)
    val_losses = np.asarray(val_losses, dtype=np.float64)

    n_epochs = len(train_losses)
    assert len(val_losses) == n_epochs, (
        "train_losses and val_losses must have the same length"
    )

    # Best epoch (lowest validation loss)
    best_epoch = int(np.argmin(val_losses))

    # Overfit ratio at the final epoch
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    overfit_ratio = final_val / max(final_train, 1e-30)

    # Check if validation loss is trending upward at the end
    # Use a rolling mean over the last `window` epochs
    if n_epochs >= 2 * window:
        recent_val = val_losses[-window:]
        prior_val = val_losses[-2 * window:-window]
        val_increasing = float(np.mean(recent_val)) > float(np.mean(prior_val))
    else:
        # Not enough data for windowed comparison; compare halves
        mid = n_epochs // 2
        val_increasing = float(np.mean(val_losses[mid:])) > float(
            np.mean(val_losses[:mid])
        )

    # Check if training loss is still decreasing at the end
    if n_epochs >= 2 * window:
        recent_train = train_losses[-window:]
        prior_train = train_losses[-2 * window:-window]
        train_still_decreasing = float(np.mean(recent_train)) < float(
            np.mean(prior_train)
        )
    else:
        mid = n_epochs // 2
        train_still_decreasing = float(np.mean(train_losses[mid:])) < float(
            np.mean(train_losses[:mid])
        )

    # Determine overfitting
    is_overfitting = val_increasing and train_still_decreasing
    # Also flag if the ratio is very large even without clear trend
    if overfit_ratio > 5.0:
        is_overfitting = True

    # Build recommendation
    if is_overfitting:
        gap = n_epochs - best_epoch
        recommendation = (
            f"Overfitting detected. Validation loss started increasing after "
            f"epoch {best_epoch}. Consider early stopping at epoch {best_epoch} "
            f"(would save {gap} epochs). Also consider increasing regularization "
            f"or dropout."
        )
    elif overfit_ratio > 2.0:
        recommendation = (
            f"Mild overfitting: val/train ratio = {overfit_ratio:.2f}. "
            f"Monitor closely. Consider adding regularization."
        )
    else:
        recommendation = (
            f"No significant overfitting detected. "
            f"Val/train ratio = {overfit_ratio:.2f}, best epoch = {best_epoch}."
        )

    result = {
        'is_overfitting': bool(is_overfitting),
        'overfit_ratio': float(overfit_ratio),
        'val_increasing': bool(val_increasing),
        'best_epoch': best_epoch,
        'recommendation': recommendation,
    }

    logger.info(f"Overfitting check: is_overfitting={result['is_overfitting']}, "
                f"ratio={result['overfit_ratio']:.2f}, best_epoch={result['best_epoch']}")
    logger.info(f"  Recommendation: {result['recommendation']}")

    return result


# ---------------------------------------------------------------------------
# 3. PDE residual distribution check
# ---------------------------------------------------------------------------

def check_pde_residual_distribution(model, S_range, t_range, coefficients,
                                     term_names, n_samples=50000):
    """
    Sample random collocation points in the (S, t) domain and evaluate
    the PDE residual using the discovered PDE coefficients.

    The PDE residual is:
        R = dV/dt - sum_i(coeff_i * term_i)

    where the terms and their coefficients come from SINDy discovery,
    and the derivatives are computed via automatic differentiation on
    the PINN model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PINN model that takes (S, t) inputs and returns V.
    S_range : tuple of (float, float)
        (S_min, S_max) for sampling.
    t_range : tuple of (float, float)
        (t_min, t_max) for sampling.
    coefficients : array-like, shape (n_terms,)
        Discovered PDE coefficients for each term.
    term_names : list of str
        Names matching the coefficient array (e.g. TERM_NAMES).
    n_samples : int
        Number of random collocation points.

    Returns
    -------
    dict
        'mean_residual' : float
        'std_residual' : float
        'max_residual' : float
        'percentile_95' : float
        'worst_locations' : ndarray, shape (10, 3)
            Columns: S, t, residual value for the 10 worst points.
    """
    from src.pinn_validation import compute_pde_residual

    set_all_seeds(42)

    S_min, S_max = S_range
    t_min, t_max = t_range

    # Sample random collocation points uniformly
    S_np = S_min + (S_max - S_min) * np.random.rand(n_samples)
    t_np = t_min + (t_max - t_min) * np.random.rand(n_samples)

    # Convert to tensors for compute_pde_residual
    S_samples = torch.tensor(S_np, dtype=torch.float64).unsqueeze(-1).requires_grad_(True)
    t_samples = torch.tensor(t_np, dtype=torch.float64).unsqueeze(-1).requires_grad_(True)

    # Compute the PDE residual at each point
    residuals = compute_pde_residual(
        model, S_samples, t_samples, coefficients, term_names
    )
    residuals = residuals.detach().numpy().ravel()
    S_samples = S_np
    t_samples = t_np
    abs_residuals = np.abs(residuals)

    mean_residual = float(np.mean(abs_residuals))
    std_residual = float(np.std(abs_residuals))
    max_residual = float(np.max(abs_residuals))
    percentile_95 = float(np.percentile(abs_residuals, 95))

    # Find the 10 worst (highest absolute residual) points
    worst_idx = np.argsort(abs_residuals)[-10:][::-1]
    worst_locations = np.column_stack([
        S_samples[worst_idx],
        t_samples[worst_idx],
        residuals[worst_idx],
    ])

    result = {
        'mean_residual': mean_residual,
        'std_residual': std_residual,
        'max_residual': max_residual,
        'percentile_95': percentile_95,
        'worst_locations': worst_locations,
    }

    logger.info("PDE residual distribution over %d collocation points:", n_samples)
    logger.info(f"  Mean |R|:     {mean_residual:.6e}")
    logger.info(f"  Std  |R|:     {std_residual:.6e}")
    logger.info(f"  Max  |R|:     {max_residual:.6e}")
    logger.info(f"  95th pct |R|: {percentile_95:.6e}")
    logger.info("  Worst 10 locations (S, t, residual):")
    for row in worst_locations:
        logger.info(f"    S={row[0]:.2f}, t={row[1]:.4f}, R={row[2]:.6e}")

    return result


# ---------------------------------------------------------------------------
# 4. Training convergence check
# ---------------------------------------------------------------------------

def check_training_convergence(loss_history, min_epochs=5000):
    """
    Check whether training has converged by analyzing the tail of the loss curve.

    Convergence is declared when the relative decrease in loss over the last
    20% of epochs is below a threshold (1% relative change), and at least
    min_epochs have been run.

    Parameters
    ----------
    loss_history : array-like
        Per-epoch loss values.
    min_epochs : int
        Minimum number of epochs required before declaring convergence.

    Returns
    -------
    dict
        'converged' : bool
        'n_epochs' : int
        'final_loss' : float
        'tail_relative_change' : float
            Relative change in the mean loss between the first and second
            halves of the last 20% of the loss curve.
        'recommendation' : str
    """
    loss_history = np.asarray(loss_history, dtype=np.float64)
    n_epochs = len(loss_history)
    final_loss = float(loss_history[-1])

    # Analyze the last 20% of the training curve
    tail_start = max(0, int(0.8 * n_epochs))
    tail = loss_history[tail_start:]

    if len(tail) < 4:
        # Not enough data to judge convergence
        return {
            'converged': False,
            'n_epochs': n_epochs,
            'final_loss': final_loss,
            'tail_relative_change': float('inf'),
            'recommendation': (
                f"Only {n_epochs} epochs completed. Need at least {min_epochs} "
                f"to assess convergence. Continue training."
            ),
        }

    mid = len(tail) // 2
    first_half_mean = float(np.mean(tail[:mid]))
    second_half_mean = float(np.mean(tail[mid:]))

    # Relative change: how much the loss decreased in the second half
    denom = max(abs(first_half_mean), 1e-30)
    tail_relative_change = abs(first_half_mean - second_half_mean) / denom

    enough_epochs = n_epochs >= min_epochs
    loss_plateau = tail_relative_change < 0.01  # less than 1% relative change

    converged = enough_epochs and loss_plateau

    if converged:
        recommendation = (
            f"Training converged after {n_epochs} epochs. "
            f"Final loss = {final_loss:.6e}, tail relative change = "
            f"{tail_relative_change:.4e}."
        )
    elif not enough_epochs:
        recommendation = (
            f"Only {n_epochs}/{min_epochs} epochs completed. "
            f"Tail relative change = {tail_relative_change:.4e}. "
            f"Continue training for at least {min_epochs - n_epochs} more epochs."
        )
    else:
        recommendation = (
            f"Loss is still decreasing significantly "
            f"(tail relative change = {tail_relative_change:.4e}). "
            f"Continue training or increase learning rate schedule."
        )

    result = {
        'converged': bool(converged),
        'n_epochs': n_epochs,
        'final_loss': final_loss,
        'tail_relative_change': float(tail_relative_change),
        'recommendation': recommendation,
    }

    logger.info(f"Convergence check: converged={result['converged']}, "
                f"epochs={n_epochs}, tail_change={tail_relative_change:.4e}")
    logger.info(f"  Recommendation: {result['recommendation']}")

    return result


# ---------------------------------------------------------------------------
# 5. Numerical derivative quality check
# ---------------------------------------------------------------------------

def check_numerical_derivative_quality(V, S_grid, t_grid, K, r, sigma, T,
                                        option_type='call'):
    """
    Compare numerically computed derivatives against analytical Black-Scholes
    derivatives and report relative L2 errors.

    Parameters
    ----------
    V : ndarray, shape (n_S, n_t)
        Option price surface.
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Maturity.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    dict
        'dVdt_rel_L2' : float
        'dVdS_rel_L2' : float
        'd2VdS2_rel_L2' : float
        'quality_summary' : str
    """
    from src.sindy_discovery import compute_derivatives
    from src.data_generation import (
        bs_theta_call, bs_theta_put,
        bs_call_delta, bs_put_delta,
        bs_gamma,
    )

    derivs = compute_derivatives(V, S_grid, t_grid, smooth=False, trim=5)
    S_mesh = derivs['S_mesh']
    t_mesh = derivs['t_mesh']
    tau_mesh = T - t_mesh

    if option_type == 'call':
        theta_analytical = bs_theta_call(S_mesh, K, r, sigma, tau_mesh)
        delta_analytical = bs_call_delta(S_mesh, K, r, sigma, tau_mesh)
    else:
        theta_analytical = bs_theta_put(S_mesh, K, r, sigma, tau_mesh)
        delta_analytical = bs_put_delta(S_mesh, K, r, sigma, tau_mesh)

    gamma_analytical = bs_gamma(S_mesh, K, r, sigma, tau_mesh)

    def rel_l2(numerical, analytical):
        denom = np.linalg.norm(analytical)
        if denom < 1e-15:
            return 0.0
        return float(np.linalg.norm(numerical - analytical) / denom)

    dVdt_err = rel_l2(derivs['dVdt'], theta_analytical)
    dVdS_err = rel_l2(derivs['dVdS'], delta_analytical)
    d2VdS2_err = rel_l2(derivs['d2VdS2'], gamma_analytical)

    # Quality summary
    errors = {'dVdt': dVdt_err, 'dVdS': dVdS_err, 'd2VdS2': d2VdS2_err}
    poor = [name for name, err in errors.items() if err > 0.10]
    marginal = [name for name, err in errors.items() if 0.01 < err <= 0.10]
    good = [name for name, err in errors.items() if err <= 0.01]

    parts = []
    if good:
        parts.append(f"Good (<1%): {', '.join(good)}")
    if marginal:
        parts.append(f"Marginal (1-10%): {', '.join(marginal)}")
    if poor:
        parts.append(f"Poor (>10%): {', '.join(poor)}")
    quality_summary = "; ".join(parts)

    result = {
        'dVdt_rel_L2': dVdt_err,
        'dVdS_rel_L2': dVdS_err,
        'd2VdS2_rel_L2': d2VdS2_err,
        'quality_summary': quality_summary,
    }

    logger.info("Numerical derivative quality check:")
    logger.info(f"  dV/dt  relative L2 error: {dVdt_err:.6e}")
    logger.info(f"  dV/dS  relative L2 error: {dVdS_err:.6e}")
    logger.info(f"  d2V/dS2 relative L2 error: {d2VdS2_err:.6e}")
    logger.info(f"  Summary: {quality_summary}")

    return result


# ---------------------------------------------------------------------------
# 6. SINDy sparsity stability via bootstrap
# ---------------------------------------------------------------------------

def check_sindy_sparsity_stability(V, S_grid, t_grid, n_bootstrap=20,
                                    K=100, r=0.05, sigma=0.2, T=1.0,
                                    option_type='call'):
    """
    Assess the stability of SINDy term selection via bootstrap resampling.

    Resamples the flattened data points (with replacement) n_bootstrap times,
    runs SINDy each time, and reports the frequency of each term being
    selected and the mean/std of each coefficient across bootstrap runs.

    Parameters
    ----------
    V : ndarray, shape (n_S, n_t)
        Option price surface.
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    n_bootstrap : int
        Number of bootstrap iterations.
    K : float
    r : float
    sigma : float
    T : float
    option_type : str

    Returns
    -------
    dict
        'term_names' : list of str
        'selection_frequency' : ndarray, shape (n_terms,)
            Fraction of bootstrap runs in which each term was selected.
        'coeff_mean' : ndarray, shape (n_terms,)
        'coeff_std' : ndarray, shape (n_terms,)
        'all_coefficients' : ndarray, shape (n_bootstrap, n_terms)
        'stable' : bool
            True if the same set of terms was selected in >= 90% of runs.
    """
    from src.sindy_discovery import (
        compute_derivatives, build_candidate_library, stlsq_sweep, TERM_NAMES
    )

    set_all_seeds(42)

    # First, compute derivatives on the full grid
    derivs = compute_derivatives(V, S_grid, t_grid, smooth=False, trim=5)
    V_tr = derivs['V']
    dVdS = derivs['dVdS']
    d2VdS2 = derivs['d2VdS2']
    S_mesh = derivs['S_mesh']
    dVdt = derivs['dVdt']

    # Build full library and target
    library_full = build_candidate_library(V_tr, dVdS, d2VdS2, S_mesh)
    target_full = dVdt.ravel()
    n_points = len(target_full)
    n_terms = library_full.shape[1]

    all_coefficients = np.zeros((n_bootstrap, n_terms))
    all_active_masks = np.zeros((n_bootstrap, n_terms), dtype=bool)

    for b in range(n_bootstrap):
        # Bootstrap resample (with replacement)
        rng = np.random.RandomState(42 + b)
        idx = rng.choice(n_points, size=n_points, replace=True)
        lib_boot = library_full[idx]
        tgt_boot = target_full[idx]

        # Run STLSQ sweep on the bootstrap sample
        best, _ = stlsq_sweep(lib_boot, tgt_boot)
        all_coefficients[b] = best['coefficients']
        all_active_masks[b] = best['active_mask']

    # Compute statistics
    selection_frequency = np.mean(all_active_masks, axis=0)
    coeff_mean = np.mean(all_coefficients, axis=0)
    coeff_std = np.std(all_coefficients, axis=0)

    # Stability: check if the same terms are selected in >= 90% of runs
    # A term is considered "consistently selected" if frequency >= 0.9
    # or "consistently not selected" if frequency <= 0.1
    consistent = np.all((selection_frequency >= 0.9) | (selection_frequency <= 0.1))
    stable = bool(consistent)

    result = {
        'term_names': list(TERM_NAMES),
        'selection_frequency': selection_frequency,
        'coeff_mean': coeff_mean,
        'coeff_std': coeff_std,
        'all_coefficients': all_coefficients,
        'stable': stable,
    }

    logger.info("SINDy bootstrap stability analysis (%d runs):", n_bootstrap)
    for i, name in enumerate(TERM_NAMES):
        logger.info(
            f"  {name:>15s}: selected {selection_frequency[i]:.0%}, "
            f"coeff = {coeff_mean[i]:+.6f} +/- {coeff_std[i]:.6f}"
        )
    logger.info(f"  Overall stable: {stable}")

    return result


# ---------------------------------------------------------------------------
# 7. PINN generalization check
# ---------------------------------------------------------------------------

def check_pinn_generalization(model, S_grid_extended, t_grid, K, r, sigma, T,
                               option_type='call'):
    """
    Test PINN accuracy on an extended S domain to assess generalization.

    Splits S_grid_extended into in-domain and out-of-domain regions based on
    the model's training range (inferred as the middle portion), then computes
    errors for each region.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PINN model.
    S_grid_extended : ndarray
        Extended stock price grid (wider than training domain).
    t_grid : ndarray
        Time grid.
    K : float
    r : float
    sigma : float
    T : float
    option_type : str

    Returns
    -------
    dict
        'in_domain_rmse' : float
        'out_domain_rmse' : float
        'in_domain_max_error' : float
        'out_domain_max_error' : float
        'in_domain_rel_error' : float
        'out_domain_rel_error' : float
        'generalization_ratio' : float
            out_domain_rmse / in_domain_rmse
    """
    from src.data_generation import bs_call_price, bs_put_price

    S_ext = np.asarray(S_grid_extended, dtype=np.float64)
    t_arr = np.asarray(t_grid, dtype=np.float64)

    S_mesh, t_mesh = np.meshgrid(S_ext, t_arr, indexing='ij')
    tau_mesh = T - t_mesh

    # Analytical prices on the extended grid
    if option_type == 'call':
        V_analytical = bs_call_price(S_mesh, K, r, sigma, tau_mesh)
    else:
        V_analytical = bs_put_price(S_mesh, K, r, sigma, tau_mesh)

    # PINN predictions — BSPINN takes separate S, t arguments, uses float64
    S_flat = torch.tensor(S_mesh.ravel(), dtype=torch.float64).unsqueeze(1)
    t_flat = torch.tensor(t_mesh.ravel(), dtype=torch.float64).unsqueeze(1)

    model.eval()
    with torch.no_grad():
        V_pred_flat = model(S_flat, t_flat).numpy().ravel()
    V_pred = V_pred_flat.reshape(S_mesh.shape)

    # Determine in-domain vs out-of-domain based on model's training range
    S_min_train = float(model.S_min)
    S_max_train = float(model.S_max)

    in_S_mask = (S_mesh >= S_min_train) & (S_mesh <= S_max_train)
    out_S_mask = ~in_S_mask

    in_mask = in_S_mask
    out_mask = out_S_mask

    # In-domain errors
    in_errors = np.abs(V_pred[in_mask] - V_analytical[in_mask])
    in_domain_rmse = float(np.sqrt(np.mean(in_errors ** 2)))
    in_domain_max_error = float(np.max(in_errors))
    in_ana_norm = np.linalg.norm(V_analytical[in_mask])
    in_domain_rel_error = (
        float(np.linalg.norm(in_errors) / in_ana_norm)
        if in_ana_norm > 1e-15 else 0.0
    )

    # Out-of-domain errors
    out_errors = np.abs(V_pred[out_mask] - V_analytical[out_mask])
    out_domain_rmse = float(np.sqrt(np.mean(out_errors ** 2)))
    out_domain_max_error = float(np.max(out_errors))
    out_ana_norm = np.linalg.norm(V_analytical[out_mask])
    out_domain_rel_error = (
        float(np.linalg.norm(out_errors) / out_ana_norm)
        if out_ana_norm > 1e-15 else 0.0
    )

    generalization_ratio = (
        out_domain_rmse / max(in_domain_rmse, 1e-30)
    )

    result = {
        'in_domain_rmse': in_domain_rmse,
        'out_domain_rmse': out_domain_rmse,
        'in_domain_max_error': in_domain_max_error,
        'out_domain_max_error': out_domain_max_error,
        'in_domain_rel_error': in_domain_rel_error,
        'out_domain_rel_error': out_domain_rel_error,
        'generalization_ratio': float(generalization_ratio),
    }

    logger.info("PINN generalization check:")
    logger.info(f"  In-domain  RMSE: {in_domain_rmse:.6e}, "
                f"max: {in_domain_max_error:.6e}, "
                f"rel: {in_domain_rel_error:.6e}")
    logger.info(f"  Out-domain RMSE: {out_domain_rmse:.6e}, "
                f"max: {out_domain_max_error:.6e}, "
                f"rel: {out_domain_rel_error:.6e}")
    logger.info(f"  Generalization ratio (out/in): {generalization_ratio:.2f}")

    if generalization_ratio > 10.0:
        logger.warning(
            "PINN generalizes poorly: out-of-domain error is %.1fx "
            "larger than in-domain.", generalization_ratio
        )
    elif generalization_ratio > 3.0:
        logger.warning(
            "PINN generalization is marginal: out-of-domain error is %.1fx "
            "larger than in-domain.", generalization_ratio
        )
    else:
        logger.info("PINN generalization is acceptable.")

    return result


# ---------------------------------------------------------------------------
# 8. Monotonicity and convexity check
# ---------------------------------------------------------------------------

def check_monotonicity_and_convexity(model, S_grid, t_grid, option_type='call'):
    """
    Verify that the PINN respects option pricing monotonicity and convexity.

    For European options:
    - Calls: Delta (dV/dS) >= 0 everywhere (monotonically increasing in S).
    - Puts: Delta (dV/dS) <= 0 everywhere (monotonically decreasing in S).
    - Both: Gamma (d2V/dS2) >= 0 everywhere (convex in S).

    Derivatives are computed via automatic differentiation through the model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PINN model.
    S_grid : ndarray
    t_grid : ndarray
    option_type : str
        'call' or 'put'.

    Returns
    -------
    dict
        'delta_violation_fraction' : float
            Fraction of grid points where Delta has the wrong sign.
        'gamma_violation_fraction' : float
            Fraction of grid points where Gamma < 0.
        'n_delta_violations' : int
        'n_gamma_violations' : int
        'n_total_points' : int
        'monotonicity_ok' : bool
        'convexity_ok' : bool
    """
    S_arr = np.asarray(S_grid, dtype=np.float64)
    t_arr = np.asarray(t_grid, dtype=np.float64)

    S_mesh_np, t_mesh_np = np.meshgrid(S_arr, t_arr, indexing='ij')

    S_flat = torch.tensor(
        S_mesh_np.ravel(), dtype=torch.float64
    ).unsqueeze(1).requires_grad_(True)
    t_flat = torch.tensor(
        t_mesh_np.ravel(), dtype=torch.float64
    ).unsqueeze(1).requires_grad_(True)

    model.eval()

    # BSPINN takes separate S, t arguments
    V_pred = model(S_flat, t_flat)

    # First derivative: dV/dS via autograd
    dVdS = torch.autograd.grad(
        V_pred, S_flat,
        grad_outputs=torch.ones_like(V_pred),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Second derivative: d2V/dS2 via autograd
    d2VdS2 = torch.autograd.grad(
        dVdS, S_flat,
        grad_outputs=torch.ones_like(dVdS),
        create_graph=False,
    )[0]

    delta_np = dVdS.detach().numpy().ravel()
    gamma_np = d2VdS2.detach().numpy().ravel()
    n_total = len(delta_np)

    # Monotonicity check
    # Allow a small tolerance for numerical noise
    tol = 1e-6
    if option_type == 'call':
        n_delta_violations = int(np.sum(delta_np < -tol))
    else:
        n_delta_violations = int(np.sum(delta_np > tol))

    # Convexity check: Gamma >= 0 for both calls and puts
    n_gamma_violations = int(np.sum(gamma_np < -tol))

    delta_violation_fraction = n_delta_violations / max(n_total, 1)
    gamma_violation_fraction = n_gamma_violations / max(n_total, 1)

    # Consider it "ok" if violation fraction is below 1%
    monotonicity_ok = delta_violation_fraction < 0.01
    convexity_ok = gamma_violation_fraction < 0.01

    result = {
        'delta_violation_fraction': float(delta_violation_fraction),
        'gamma_violation_fraction': float(gamma_violation_fraction),
        'n_delta_violations': n_delta_violations,
        'n_gamma_violations': n_gamma_violations,
        'n_total_points': n_total,
        'monotonicity_ok': bool(monotonicity_ok),
        'convexity_ok': bool(convexity_ok),
    }

    logger.info("Monotonicity and convexity check (%s):", option_type)
    logger.info(
        f"  Delta violations: {n_delta_violations}/{n_total} "
        f"({delta_violation_fraction:.2%})"
    )
    logger.info(
        f"  Gamma violations: {n_gamma_violations}/{n_total} "
        f"({gamma_violation_fraction:.2%})"
    )
    logger.info(f"  Monotonicity OK: {monotonicity_ok}")
    logger.info(f"  Convexity OK:    {convexity_ok}")

    if not monotonicity_ok:
        logger.warning(
            "Significant monotonicity violations detected (%.2f%% of points). "
            "The PINN may need additional training or physics-informed loss terms.",
            delta_violation_fraction * 100,
        )
    if not convexity_ok:
        logger.warning(
            "Significant convexity violations detected (%.2f%% of points). "
            "Consider adding a Gamma-positivity penalty to the loss.",
            gamma_violation_fraction * 100,
        )

    return result
