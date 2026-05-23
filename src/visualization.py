"""
Publication-quality plotting functions for BS PDE Discovery project.

All plots saved to outputs/figures/ as 300 DPI PNGs.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from src.utils import setup_logging

logger = setup_logging(__name__)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('ggplot')

FONTSIZE_LABEL = 12
FONTSIZE_TITLE = 14
DPI = 300


def _savefig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved figure: {name}")
    return path


def plot_price_surfaces(V_call, V_put, S_grid, t_grid):
    """Two side-by-side 3D surface plots for call and put prices."""
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')
    fig = plt.figure(figsize=(14, 6))

    for idx, (V, title) in enumerate([(V_call, 'Call Price'), (V_put, 'Put Price')]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        # Subsample for cleaner 3D plot
        step_s = max(1, len(S_grid) // 40)
        step_t = max(1, len(t_grid) // 40)
        ax.plot_surface(
            S_mesh[::step_s, ::step_t], t_mesh[::step_s, ::step_t],
            V[::step_s, ::step_t],
            cmap='RdYlBu_r', alpha=0.9, edgecolor='none'
        )
        ax.set_xlabel('Stock Price S ($)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('Time t (years)', fontsize=FONTSIZE_LABEL)
        ax.set_zlabel('Option Price V ($)', fontsize=FONTSIZE_LABEL)
        ax.set_title(title, fontsize=FONTSIZE_TITLE)

    fig.tight_layout()
    return _savefig(fig, 'price_surfaces_3d.png')


def plot_sindy_threshold_sweep(sweep_results):
    """Two-panel plot: R^2 vs threshold and active terms vs threshold."""
    thresholds = [r['threshold'] for r in sweep_results]
    r2s = [r['r2'] for r in sweep_results]
    n_active = [r['n_active'] for r in sweep_results]

    # Find selected threshold (fewest terms with R2 > 0.99)
    candidates = [r for r in sweep_results if r['r2'] > 0.99]
    if candidates:
        candidates.sort(key=lambda x: (x['n_active'], x['bic']))
        selected_thr = candidates[0]['threshold']
    else:
        selected_thr = sweep_results[0]['threshold']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.semilogx(thresholds, r2s, 'b.-', linewidth=1.5)
    ax1.axhline(y=0.99, color='gray', linestyle='--', label='R$^2$ = 0.99')
    ax1.axvline(x=selected_thr, color='red', linestyle='--', alpha=0.7,
                label=f'Selected ({selected_thr:.4f})')
    ax1.set_xlabel('Threshold', fontsize=FONTSIZE_LABEL)
    ax1.set_ylabel('R$^2$', fontsize=FONTSIZE_LABEL)
    ax1.set_title('R$^2$ vs Threshold', fontsize=FONTSIZE_TITLE)
    ax1.legend(fontsize=10)

    ax2.semilogx(thresholds, n_active, 'g.-', linewidth=1.5)
    ax2.axvline(x=selected_thr, color='red', linestyle='--', alpha=0.7,
                label=f'Selected ({selected_thr:.4f})')
    ax2.set_xlabel('Threshold', fontsize=FONTSIZE_LABEL)
    ax2.set_ylabel('Number of Active Terms', fontsize=FONTSIZE_LABEL)
    ax2.set_title('Sparsity vs Threshold', fontsize=FONTSIZE_TITLE)
    ax2.legend(fontsize=10)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.tight_layout()
    return _savefig(fig, 'sindy_threshold_sweep.png')


def plot_sindy_coefficients(discovered, true, term_names):
    """Grouped bar chart comparing discovered vs true coefficients."""
    x = np.arange(len(term_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, discovered, width, label='Discovered', color='steelblue')
    bars2 = ax.bar(x + width / 2, true, width, label='True', color='darkorange')

    # Annotate relative errors
    for i in range(len(term_names)):
        denom = max(abs(true[i]), 1e-10)
        rel_err = abs(discovered[i] - true[i]) / denom
        y_pos = max(abs(discovered[i]), abs(true[i])) + 0.005
        if true[i] < 0 or discovered[i] < 0:
            y_pos = max(discovered[i], true[i]) + 0.005
        ax.text(x[i], y_pos, f'{rel_err:.1%}', ha='center', fontsize=9, color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(term_names, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Coefficient Value', fontsize=FONTSIZE_LABEL)
    ax.set_title('SINDy Coefficient Comparison', fontsize=FONTSIZE_TITLE)
    ax.legend(fontsize=11)
    ax.axhline(y=0, color='black', linewidth=0.5)

    fig.tight_layout()
    return _savefig(fig, 'sindy_coefficient_comparison.png')


def plot_pinn_results(V_pinn, V_analytical, S_grid, t_grid, option_type='call'):
    """Three-panel: PINN surface, analytical surface, error heatmap."""
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')
    error = np.abs(V_pinn - V_analytical)

    fig = plt.figure(figsize=(18, 5))

    step_s = max(1, len(S_grid) // 40)
    step_t = max(1, len(t_grid) // 40)

    # PINN surface
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(S_mesh[::step_s, ::step_t], t_mesh[::step_s, ::step_t],
                     V_pinn[::step_s, ::step_t], cmap='RdYlBu_r', alpha=0.9, edgecolor='none')
    ax1.set_xlabel('S ($)', fontsize=10)
    ax1.set_ylabel('t (years)', fontsize=10)
    ax1.set_zlabel('V ($)', fontsize=10)
    ax1.set_title(f'PINN {option_type.capitalize()} Price', fontsize=FONTSIZE_TITLE)

    # Analytical surface
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(S_mesh[::step_s, ::step_t], t_mesh[::step_s, ::step_t],
                     V_analytical[::step_s, ::step_t], cmap='RdYlBu_r', alpha=0.9, edgecolor='none')
    ax2.set_xlabel('S ($)', fontsize=10)
    ax2.set_ylabel('t (years)', fontsize=10)
    ax2.set_zlabel('V ($)', fontsize=10)
    ax2.set_title(f'Analytical {option_type.capitalize()} Price', fontsize=FONTSIZE_TITLE)

    # Error heatmap
    ax3 = fig.add_subplot(1, 3, 3)
    err_plot = error.copy()
    err_plot[err_plot == 0] = 1e-15  # avoid log(0)
    im = ax3.pcolormesh(t_grid, S_grid, err_plot, cmap='Reds',
                        shading='auto')
    cbar = fig.colorbar(im, ax=ax3)
    cbar.set_label('|Error| ($)', fontsize=10)
    ax3.set_xlabel('t (years)', fontsize=10)
    ax3.set_ylabel('S ($)', fontsize=10)
    ax3.set_title('Absolute Error', fontsize=FONTSIZE_TITLE)

    fig.tight_layout()
    return _savefig(fig, f'pinn_vs_analytical_{option_type}.png')


def plot_training_loss(loss_history):
    """Training loss curves with all components and validation on second y-axis."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    epochs = range(len(loss_history['total']))
    ax1.semilogy(epochs, loss_history['total'], 'b-', linewidth=1.5, label='Total')
    ax1.semilogy(epochs, loss_history['pde'], 'g--', linewidth=1, alpha=0.7, label='PDE')
    ax1.semilogy(epochs, loss_history['bc'], 'm--', linewidth=1, alpha=0.7, label='BC')
    ax1.semilogy(epochs, loss_history['data'], 'c--', linewidth=1, alpha=0.7, label='Data')

    ax1.set_xlabel('Epoch', fontsize=FONTSIZE_LABEL)
    ax1.set_ylabel('Training Loss (log)', fontsize=FONTSIZE_LABEL)
    ax1.set_title('PINN Training Loss', fontsize=FONTSIZE_TITLE)

    # Validation on second axis
    if 'val' in loss_history and loss_history['val']:
        ax2 = ax1.twinx()
        val_epochs = loss_history.get('val_epochs', list(range(0, len(loss_history['val']) * 500, 500)))
        if len(val_epochs) != len(loss_history['val']):
            val_epochs = list(range(0, len(loss_history['val']) * 500, 500))[:len(loss_history['val'])]
        ax2.semilogy(val_epochs, loss_history['val'], 'r--', linewidth=1.5,
                     alpha=0.8, label='Validation')
        ax2.set_ylabel('Validation Loss (log)', fontsize=FONTSIZE_LABEL, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper left', fontsize=10)

    # Early stopping marker
    if 'early_stop_epoch' in loss_history and loss_history['early_stop_epoch'] is not None:
        ax1.axvline(x=loss_history['early_stop_epoch'], color='red',
                    linestyle=':', linewidth=2, label='Early Stop')

    ax1.legend(loc='upper right', fontsize=10)
    fig.tight_layout()
    return _savefig(fig, 'pinn_training_loss.png')


def plot_greeks_comparison(pinn_delta, analytical_delta, pinn_gamma,
                           analytical_gamma, S_grid, t_grid):
    """2x2 heatmap: PINN vs analytical Delta and Gamma."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    data = [
        (pinn_delta, 'PINN Delta', axes[0, 0]),
        (analytical_delta, 'Analytical Delta', axes[0, 1]),
        (pinn_gamma, 'PINN Gamma', axes[1, 0]),
        (analytical_gamma, 'Analytical Gamma', axes[1, 1]),
    ]

    # Shared color ranges per row
    delta_vmin = min(np.nanmin(pinn_delta), np.nanmin(analytical_delta))
    delta_vmax = max(np.nanmax(pinn_delta), np.nanmax(analytical_delta))
    gamma_vmin = min(np.nanmin(pinn_gamma), np.nanmin(analytical_gamma))
    gamma_vmax = max(np.nanmax(pinn_gamma), np.nanmax(analytical_gamma))

    for i, (arr, title, ax) in enumerate(data):
        vmin = delta_vmin if i < 2 else gamma_vmin
        vmax = delta_vmax if i < 2 else gamma_vmax
        im = ax.pcolormesh(t_grid, S_grid, arr, cmap='viridis',
                           vmin=vmin, vmax=vmax, shading='auto')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('t (years)', fontsize=10)
        ax.set_ylabel('S ($)', fontsize=10)
        ax.set_title(title, fontsize=FONTSIZE_TITLE)

    fig.tight_layout()
    return _savefig(fig, 'greeks_comparison.png')


def plot_greeks_error(delta_error, gamma_error, S_grid, t_grid):
    """Two-panel heatmap of absolute errors for Delta and Gamma."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.pcolormesh(t_grid, S_grid, delta_error, cmap='Reds', shading='auto')
    fig.colorbar(im1, ax=ax1)
    ax1.set_xlabel('t (years)', fontsize=FONTSIZE_LABEL)
    ax1.set_ylabel('S ($)', fontsize=FONTSIZE_LABEL)
    ax1.set_title('Delta Absolute Error', fontsize=FONTSIZE_TITLE)

    im2 = ax2.pcolormesh(t_grid, S_grid, gamma_error, cmap='Reds', shading='auto')
    fig.colorbar(im2, ax=ax2)
    ax2.set_xlabel('t (years)', fontsize=FONTSIZE_LABEL)
    ax2.set_ylabel('S ($)', fontsize=FONTSIZE_LABEL)
    ax2.set_title('Gamma Absolute Error', fontsize=FONTSIZE_TITLE)

    fig.tight_layout()
    return _savefig(fig, 'greeks_error_heatmap.png')


def plot_noise_robustness(noise_df):
    """Multi-panel figure: coefficient errors, R^2, active terms, stability vs noise."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    noise_levels = noise_df['noise_level'].values

    # Top-left: relative errors vs noise
    ax = axes[0, 0]
    for col, label in [('rel_error_V', 'V'), ('rel_error_SdVdS', 'S*dV/dS'),
                       ('rel_error_S2d2VdS2', 'S2*d2V/dS2')]:
        if col in noise_df.columns:
            ax.plot(noise_levels, noise_df[col].values, '.-', label=label, linewidth=1.5)
    ax.set_xlabel('Noise Level', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Relative Error', fontsize=FONTSIZE_LABEL)
    ax.set_title('Coefficient Errors vs Noise', fontsize=FONTSIZE_TITLE)
    ax.legend(fontsize=10)

    # Top-right: R^2 vs noise
    ax = axes[0, 1]
    ax.plot(noise_levels, noise_df['r2'].values, 'b.-', linewidth=1.5)
    ax.set_xlabel('Noise Level', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('R$^2$', fontsize=FONTSIZE_LABEL)
    ax.set_title('R$^2$ vs Noise', fontsize=FONTSIZE_TITLE)

    # Bottom-left: active terms
    ax = axes[1, 0]
    ax.bar(noise_levels, noise_df['n_active_terms'].values, width=0.008, color='teal')
    ax.set_xlabel('Noise Level', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Active Terms', fontsize=FONTSIZE_LABEL)
    ax.set_title('Active Terms vs Noise', fontsize=FONTSIZE_TITLE)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Bottom-right: bootstrap stability
    ax = axes[1, 1]
    if 'bootstrap_stability_pct' in noise_df.columns:
        vals = noise_df['bootstrap_stability_pct'].values
        ax.plot(noise_levels, vals, 'r.-', linewidth=1.5)
        ax.set_ylabel('Stability (%)', fontsize=FONTSIZE_LABEL)
    ax.set_xlabel('Noise Level', fontsize=FONTSIZE_LABEL)
    ax.set_title('Bootstrap Stability vs Noise', fontsize=FONTSIZE_TITLE)

    # Mark critical noise threshold (first failure of correct structure)
    if 'correct_structure' in noise_df.columns:
        failures = noise_df[~noise_df['correct_structure']]
        if len(failures) > 0:
            crit = failures['noise_level'].iloc[0]
            for ax in axes.flat:
                ax.axvline(x=crit, color='red', linestyle=':', alpha=0.7)

    fig.tight_layout()
    return _savefig(fig, 'noise_robustness.png')


def plot_parameter_generalization(param_df):
    """Heatmaps of relative coefficient errors for each (sigma, r) combo."""
    sigmas = sorted(param_df['sigma'].unique())
    rs = sorted(param_df['r'].unique())

    error_cols = [('rel_error_V', 'V coefficient'),
                  ('rel_error_SdVdS', 'S*dV/dS coefficient'),
                  ('rel_error_S2d2VdS2', 'S$^2$*d$^2$V/dS$^2$ coefficient')]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (col, title) in enumerate(error_cols):
        if col not in param_df.columns:
            continue
        matrix = np.zeros((len(sigmas), len(rs)))
        for i, s in enumerate(sigmas):
            for j, r in enumerate(rs):
                row = param_df[(param_df['sigma'] == s) & (param_df['r'] == r)]
                if len(row) > 0:
                    matrix[i, j] = row[col].values[0]

        ax = axes[idx]
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', origin='lower')
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(rs)))
        ax.set_xticklabels([f'{r:.2f}' for r in rs])
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([f'{s:.1f}' for s in sigmas])
        ax.set_xlabel('r', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('sigma', fontsize=FONTSIZE_LABEL)
        ax.set_title(f'Rel. Error: {title}', fontsize=12)

        # Annotate cells
        for i in range(len(sigmas)):
            for j in range(len(rs)):
                ax.text(j, i, f'{matrix[i, j]:.2%}', ha='center', va='center', fontsize=8)

    fig.tight_layout()
    return _savefig(fig, 'parameter_generalization.png')


def plot_pde_residual_distribution(residuals, S_points, t_points):
    """Histogram of |residual| with inset scatter of worst points."""
    abs_res = np.abs(residuals)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(abs_res, bins=100, color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_yscale('log')
    ax.set_xlabel('|PDE Residual|', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Count (log)', fontsize=FONTSIZE_LABEL)
    ax.set_title('PDE Residual Distribution', fontsize=FONTSIZE_TITLE)

    # Inset: worst 1% of points
    threshold_99 = np.percentile(abs_res, 99)
    worst_mask = abs_res >= threshold_99
    if np.sum(worst_mask) > 0:
        ax_inset = fig.add_axes([0.55, 0.55, 0.35, 0.35])
        ax_inset.scatter(t_points[worst_mask], S_points[worst_mask],
                         c=abs_res[worst_mask], cmap='Reds', s=10, alpha=0.7)
        ax_inset.set_xlabel('t', fontsize=9)
        ax_inset.set_ylabel('S', fontsize=9)
        ax_inset.set_title('Worst 1% locations', fontsize=10)

    fig.tight_layout()
    return _savefig(fig, 'pde_residual_distribution.png')


def plot_data_split_visualization(train_idx, val_idx, test_idx, S_grid, t_grid):
    """Scatter plot of train/val/test points on (S, t) plane."""
    n_S = len(S_grid)
    n_t = len(t_grid)

    def idx_to_st(indices):
        si = indices // n_t
        ti = indices % n_t
        return S_grid[si], t_grid[ti]

    fig, ax = plt.subplots(figsize=(10, 6))

    S_train, t_train = idx_to_st(np.array(train_idx))
    S_val, t_val = idx_to_st(np.array(val_idx))
    S_test, t_test = idx_to_st(np.array(test_idx))

    ax.scatter(t_train, S_train, c='blue', s=2, alpha=0.3, label=f'Train ({len(train_idx)})')
    ax.scatter(t_val, S_val, c='green', s=2, alpha=0.3, label=f'Val ({len(val_idx)})')
    ax.scatter(t_test, S_test, c='red', s=2, alpha=0.3, label=f'Test ({len(test_idx)})')

    ax.set_xlabel('t (years)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('S ($)', fontsize=FONTSIZE_LABEL)
    ax.set_title('Data Split Visualization', fontsize=FONTSIZE_TITLE)
    ax.legend(fontsize=11, markerscale=5)

    fig.tight_layout()
    return _savefig(fig, 'data_split_visualization.png')


def plot_reduced_vs_full_library(full_result, reduced_result):
    """Compare full 5-term vs reduced 3-term SINDy discovery."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Coefficient comparison (true terms only)
    true_term_labels = ['V', 'S*dV/dS', 'S2*d2V/dS2']
    true_vals = [0.05, -0.05, -0.02]  # r, -r, -sigma^2/2

    # Full library: indices 0, 3, 4 map to the three true terms
    full_coeffs = full_result['discovered_coefficients']
    full_true_term_vals = [full_coeffs[0], full_coeffs[3], full_coeffs[4]]

    # Reduced library: indices 0, 1, 2
    red_coeffs = reduced_result['discovered_coefficients']
    red_true_term_vals = [red_coeffs[0], red_coeffs[1], red_coeffs[2]]

    x = np.arange(len(true_term_labels))
    width = 0.25
    ax = axes[0]
    ax.bar(x - width, true_vals, width, label='True', color='forestgreen', alpha=0.8)
    ax.bar(x, full_true_term_vals, width, label='Full (5-term)', color='steelblue', alpha=0.8)
    ax.bar(x + width, red_true_term_vals, width, label='Reduced (3-term)', color='darkorange', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(true_term_labels, fontsize=10)
    ax.set_ylabel('Coefficient Value', fontsize=FONTSIZE_LABEL)
    ax.set_title('True-Term Coefficients', fontsize=FONTSIZE_TITLE)
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Panel 2: Relative errors on true terms
    full_rel = full_result['relative_errors']
    full_true_errs = [full_rel[0], full_rel[3], full_rel[4]]
    red_rel = reduced_result['relative_errors']
    red_true_errs = [red_rel[0], red_rel[1], red_rel[2]]

    ax = axes[1]
    ax.bar(x - width / 2, [e * 100 for e in full_true_errs], width,
           label='Full (5-term)', color='steelblue', alpha=0.8)
    ax.bar(x + width / 2, [e * 100 for e in red_true_errs], width,
           label='Reduced (3-term)', color='darkorange', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(true_term_labels, fontsize=10)
    ax.set_ylabel('Relative Error (%)', fontsize=FONTSIZE_LABEL)
    ax.set_title('Coefficient Accuracy', fontsize=FONTSIZE_TITLE)
    ax.legend(fontsize=9)

    # Panel 3: Summary metrics
    ax = axes[2]
    metrics = {
        'Full Library': {
            'R2': full_result['r2_score'],
            'Active': full_result['n_active'],
            'Cond #': full_result['condition_number'],
            'Correct\nStructure': 'No' if full_result['n_active'] != 3 else 'Yes',
        },
        'Reduced Library': {
            'R2': reduced_result['r2_score'],
            'Active': reduced_result['n_active'],
            'Cond #': reduced_result['condition_number'],
            'Correct\nStructure': 'Yes' if reduced_result['n_active'] == 3 else 'No',
        },
    }
    cell_text = []
    row_labels = list(metrics['Full Library'].keys())
    for key in row_labels:
        full_val = metrics['Full Library'][key]
        red_val = metrics['Reduced Library'][key]
        if isinstance(full_val, float):
            if full_val > 100:
                cell_text.append([f'{full_val:.2e}', f'{red_val:.2e}'])
            else:
                cell_text.append([f'{full_val:.6f}', f'{red_val:.6f}'])
        else:
            cell_text.append([str(full_val), str(red_val)])
    ax.axis('off')
    table = ax.table(cellText=cell_text,
                     rowLabels=row_labels,
                     colLabels=['Full (5-term)', 'Reduced (3-term)'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax.set_title('Library Comparison Summary', fontsize=FONTSIZE_TITLE, pad=20)

    fig.tight_layout()
    return _savefig(fig, 'reduced_vs_full_library.png')


def plot_pinn_extrapolation(errors_in_domain, errors_outside, S_ext, t_grid,
                            S_min_train=50, S_max_train=150):
    """Show PINN error degradation outside training domain."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: error heatmap on extended domain
    S_mesh, t_mesh = np.meshgrid(S_ext, t_grid, indexing='ij')
    errors_full = np.zeros_like(S_mesh)
    # Combine in-domain and outside errors
    in_mask = (S_ext >= S_min_train) & (S_ext <= S_max_train)
    out_mask = ~in_mask

    if errors_in_domain is not None:
        for i, s in enumerate(S_ext):
            if in_mask[i]:
                # Find corresponding index
                in_idx = np.sum(in_mask[:i])
                if in_idx < errors_in_domain.shape[0]:
                    errors_full[i, :] = errors_in_domain[in_idx, :errors_full.shape[1]]
            else:
                out_idx = np.sum(out_mask[:i])
                if errors_outside is not None and out_idx < errors_outside.shape[0]:
                    errors_full[i, :] = errors_outside[out_idx, :errors_full.shape[1]]

    im = ax1.pcolormesh(t_grid, S_ext, errors_full, cmap='Reds', shading='auto')
    fig.colorbar(im, ax=ax1)
    ax1.axhline(y=S_min_train, color='blue', linestyle='--', linewidth=1.5, label='Training domain')
    ax1.axhline(y=S_max_train, color='blue', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('t (years)', fontsize=FONTSIZE_LABEL)
    ax1.set_ylabel('S ($)', fontsize=FONTSIZE_LABEL)
    ax1.set_title('Absolute Error (Extended Domain)', fontsize=FONTSIZE_TITLE)
    ax1.legend(fontsize=10)

    # Right: error vs S at a fixed time slice
    t_mid_idx = len(t_grid) // 2
    ax2.plot(S_ext, errors_full[:, t_mid_idx], 'b-', linewidth=1.5)
    ax2.axvline(x=S_min_train, color='red', linestyle='--', linewidth=1.5, label='Domain boundary')
    ax2.axvline(x=S_max_train, color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('S ($)', fontsize=FONTSIZE_LABEL)
    ax2.set_ylabel('|Error| ($)', fontsize=FONTSIZE_LABEL)
    ax2.set_title(f'Error vs S at t={t_grid[t_mid_idx]:.2f}', fontsize=FONTSIZE_TITLE)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    return _savefig(fig, 'pinn_extrapolation.png')


def plot_pinn_error_analysis(error_analysis, option_type='put'):
    """
    Heatmap of absolute PINN error with annotated moneyness regions.

    Parameters
    ----------
    error_analysis : dict
        Output of :func:`~src.pinn_validation.analyze_pinn_errors`.
        Must contain 'error_grid', 'S_grid', 't_grid', and region dicts
        ('atm_region', 'itm_region', 'otm_region').
    option_type : str
        'put' or 'call', used for filename and labels.

    Returns
    -------
    str
        Path to the saved figure.
    """
    error_grid = error_analysis['error_grid']
    S_grid = error_analysis['S_grid']
    t_grid = error_analysis['t_grid']

    fig, ax = plt.subplots(figsize=(10, 7))

    # Heatmap of absolute error
    im = ax.pcolormesh(t_grid, S_grid, error_grid, cmap='Reds', shading='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('|V_pred - V_true| ($)', fontsize=FONTSIZE_LABEL)

    ax.set_xlabel('t (years)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('S ($)', fontsize=FONTSIZE_LABEL)
    ax.set_title(
        f'PINN {option_type.capitalize()} Absolute Error by Region',
        fontsize=FONTSIZE_TITLE,
    )

    # Determine strike from ATM boundaries (ATM is [0.8*K, 1.2*K])
    S_min, S_max = S_grid.min(), S_grid.max()

    # Annotate ATM region boundaries (0.8*K and 1.2*K)
    atm_info = error_analysis['atm_region']
    itm_info = error_analysis['itm_region']
    otm_info = error_analysis['otm_region']

    # Infer K from the grid: ATM band is [0.8K, 1.2K]
    # Use the midpoint of the S_grid as an approximation
    K_est = (S_min + S_max) / 2.0

    # Draw horizontal lines at 0.8*K and 1.2*K to delimit ATM
    atm_lo = 0.8 * K_est
    atm_hi = 1.2 * K_est

    ax.axhline(y=K_est, color='white', linestyle='-', linewidth=1.5, alpha=0.8)
    ax.axhline(y=atm_lo, color='white', linestyle='--', linewidth=1.0, alpha=0.6)
    ax.axhline(y=atm_hi, color='white', linestyle='--', linewidth=1.0, alpha=0.6)

    # Label regions
    t_label = t_grid[len(t_grid) // 10]  # position labels near left edge

    if option_type == 'put':
        # ITM: S < K, OTM: S > K
        itm_y = (S_min + K_est) / 2.0
        otm_y = (K_est + S_max) / 2.0
    else:
        # ITM: S > K, OTM: S < K
        otm_y = (S_min + K_est) / 2.0
        itm_y = (K_est + S_max) / 2.0

    atm_y = K_est

    ax.text(
        t_label, atm_y, 'ATM', fontsize=11, fontweight='bold',
        color='white', ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5),
    )
    ax.text(
        t_label, itm_y,
        f'ITM (rel_L2={itm_info["rel_l2"]:.4f})',
        fontsize=10, color='white', ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5),
    )
    ax.text(
        t_label, otm_y,
        f'OTM (rel_L2={otm_info["rel_l2"]:.4f})',
        fontsize=10, color='white', ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5),
    )

    # Add a summary text box in lower-right corner
    full_info = error_analysis['full_grid']
    summary = (
        f"Full grid:  rel_L2={full_info['rel_l2']:.4e}  MAE={full_info['mae']:.4e}\n"
        f"ATM ({atm_info['n_points']} pts): rel_L2={atm_info['rel_l2']:.4e}  MAE={atm_info['mae']:.4e}\n"
        f"ITM ({itm_info['n_points']} pts): rel_L2={itm_info['rel_l2']:.4e}  MAE={itm_info['mae']:.4e}\n"
        f"OTM ({otm_info['n_points']} pts): rel_L2={otm_info['rel_l2']:.4e}  MAE={otm_info['mae']:.4e}"
    )
    ax.text(
        0.98, 0.02, summary, transform=ax.transAxes,
        fontsize=8, verticalalignment='bottom', horizontalalignment='right',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85),
    )

    fig.tight_layout()
    filename = f'pinn_{option_type}_error_analysis.png'
    return _savefig(fig, filename)


# ---------------------------------------------------------------------------
# Term names consistent with sindy_discovery.TERM_NAMES
# ---------------------------------------------------------------------------
_TERM_NAMES = ['V', 'dV/dS', 'd2V/dS2', 'S*dV/dS', 'S2*d2V/dS2']
_TRUE_COEFF_INDICES = [0, 3, 4]  # V, S*dV/dS, S2*d2V/dS2
_TRUE_COEFF_LABELS = ['V', 'S*dV/dS', r'S$^2$*d$^2$V/dS$^2$']
_TRUE_VALUES = [0.05, -0.05, -0.02]  # r, -r, -sigma^2/2


def plot_baseline_comparison(baseline_results, sindy_result):
    """Bar chart comparing coefficient accuracy across all methods."""
    try:
        methods = ['SINDy', 'Dense', 'Lasso', 'Ridge+Thresh']
        baseline_keys = ['dense', 'lasso', 'ridge_threshold']

        # Extract true-term coefficients (indices 0, 3, 4) for each method
        method_coeffs = {}
        sindy_c = np.array(sindy_result['discovered_coefficients'])
        method_coeffs['SINDy'] = [sindy_c[i] for i in _TRUE_COEFF_INDICES]
        for key, label in zip(baseline_keys, methods[1:]):
            c = np.array(baseline_results[key]['coefficients'])
            method_coeffs[label] = [c[i] for i in _TRUE_COEFF_INDICES]

        n_terms = len(_TRUE_COEFF_LABELS)
        n_methods = len(methods)
        x = np.arange(n_terms)
        total_width = 0.8
        bar_w = total_width / (n_methods + 1)  # +1 for the "True" group

        fig, ax = plt.subplots(figsize=(12, 6))

        # True values bars
        ax.bar(x - total_width / 2 + bar_w / 2, _TRUE_VALUES, bar_w,
               label='True', color='forestgreen', alpha=0.85)

        colors = ['steelblue', 'darkorange', 'mediumpurple', 'indianred']
        for m_idx, method in enumerate(methods):
            offset = x - total_width / 2 + bar_w * (m_idx + 1) + bar_w / 2
            coeffs = method_coeffs[method]
            ax.bar(offset, coeffs, bar_w, label=method, color=colors[m_idx],
                   alpha=0.85)
            # Annotate relative errors
            for t_idx in range(n_terms):
                denom = max(abs(_TRUE_VALUES[t_idx]), 1e-10)
                rel_err = abs(coeffs[t_idx] - _TRUE_VALUES[t_idx]) / denom
                y_pos = coeffs[t_idx]
                va = 'bottom' if y_pos >= 0 else 'top'
                ax.text(offset[t_idx], y_pos, f'{rel_err:.1%}',
                        ha='center', va=va, fontsize=7, color='red',
                        rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(_TRUE_COEFF_LABELS, fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('Coefficient Value', fontsize=FONTSIZE_LABEL)
        ax.set_title('Baseline Coefficient Comparison', fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=9, ncol=3, loc='upper right')
        ax.axhline(y=0, color='black', linewidth=0.5)
        fig.tight_layout()
        return _savefig(fig, 'baseline_coefficient_comparison.png')
    except Exception as e:
        logger.warning(f"plot_baseline_comparison failed: {e}")
        return None


def plot_lasso_path(baseline_results):
    """Plot Lasso regularization path (coefficients vs alpha)."""
    try:
        lasso_path = baseline_results['lasso']['lasso_path']
        alphas = np.array(lasso_path['alphas'])
        coefs = np.array(lasso_path['coefs'])  # shape (n_features, n_alphas)

        fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(coefs.shape[0]):
            label = _TERM_NAMES[i] if i < len(_TERM_NAMES) else f'Term {i}'
            ax.plot(alphas, coefs[i, :], linewidth=1.5, label=label)

        # Mark the selected alpha
        selected_alpha = baseline_results['lasso'].get('selected_alpha', None)
        if selected_alpha is not None:
            ax.axvline(x=selected_alpha, color='red', linestyle='--',
                       linewidth=1.5, alpha=0.7,
                       label=f'Selected alpha={selected_alpha:.2e}')

        ax.set_xscale('log')
        ax.set_xlabel('Alpha (log scale)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('Coefficient Value', fontsize=FONTSIZE_LABEL)
        ax.set_title('Lasso Regularization Path', fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=9)
        ax.axhline(y=0, color='black', linewidth=0.5)
        fig.tight_layout()
        return _savefig(fig, 'baseline_lasso_path.png')
    except Exception as e:
        logger.warning(f"plot_lasso_path failed: {e}")
        return None


def plot_baseline_runtime(baseline_results):
    """Bar chart comparing runtimes across methods."""
    try:
        method_labels = []
        runtimes = []
        colors = ['steelblue', 'darkorange', 'mediumpurple', 'indianred',
                  'teal', 'goldenrod']
        for key in baseline_results:
            if isinstance(baseline_results[key], dict) and 'runtime' in baseline_results[key]:
                method_labels.append(key.replace('_', ' ').title())
                runtimes.append(baseline_results[key]['runtime'])

        if not runtimes:
            logger.warning("plot_baseline_runtime: no runtime data found")
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(method_labels))
        bar_colors = [colors[i % len(colors)] for i in range(len(method_labels))]
        bars = ax.bar(x, runtimes, color=bar_colors, alpha=0.85)

        # Annotate bars with runtime values
        for bar, rt in zip(bars, runtimes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{rt:.4f}s', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=FONTSIZE_LABEL, rotation=30,
                           ha='right')
        ax.set_ylabel('Runtime (seconds)', fontsize=FONTSIZE_LABEL)
        ax.set_title('Baseline Method Runtimes', fontsize=FONTSIZE_TITLE)
        fig.tight_layout()
        return _savefig(fig, 'baseline_runtime_comparison.png')
    except Exception as e:
        logger.warning(f"plot_baseline_runtime failed: {e}")
        return None


def plot_merton_comparison(merton_result):
    """Two panels: coefficient comparison (Merton-discovered vs pure BS true) and residual heatmap."""
    try:
        discovered = np.array(merton_result['discovered_coefficients'])
        true_bs = np.array(merton_result['true_bs_coefficients'])
        residual_grid = np.array(merton_result['residual_grid'])
        S_grid = np.array(merton_result['S_grid'])
        t_grid = np.array(merton_result['t_grid'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left panel: grouped bar chart for all 5 terms
        n_terms = len(discovered)
        x = np.arange(n_terms)
        width = 0.35
        labels = _TERM_NAMES[:n_terms]

        ax1.bar(x - width / 2, discovered, width, label='Merton Discovered',
                color='steelblue', alpha=0.85)
        ax1.bar(x + width / 2, true_bs, width, label='True BS',
                color='darkorange', alpha=0.85)

        # Annotate relative errors
        for i in range(n_terms):
            denom = max(abs(true_bs[i]), 1e-10)
            rel_err = abs(discovered[i] - true_bs[i]) / denom
            y_pos = max(abs(discovered[i]), abs(true_bs[i])) + 0.005
            if discovered[i] < 0 or true_bs[i] < 0:
                y_pos = max(discovered[i], true_bs[i]) + 0.005
            ax1.text(x[i], y_pos, f'{rel_err:.1%}', ha='center', fontsize=8,
                     color='red')

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=10)
        ax1.set_ylabel('Coefficient Value', fontsize=FONTSIZE_LABEL)
        ax1.set_title('Merton vs True BS Coefficients', fontsize=FONTSIZE_TITLE)
        ax1.legend(fontsize=10)
        ax1.axhline(y=0, color='black', linewidth=0.5)

        # Right panel: residual heatmap
        im = ax2.pcolormesh(t_grid, S_grid, np.abs(residual_grid),
                            cmap='Reds', shading='auto')
        cbar = fig.colorbar(im, ax=ax2)
        cbar.set_label('|Residual|', fontsize=10)
        ax2.set_xlabel('t (years)', fontsize=FONTSIZE_LABEL)
        ax2.set_ylabel('S ($)', fontsize=FONTSIZE_LABEL)
        ax2.set_title('PDE Residual Heatmap', fontsize=FONTSIZE_TITLE)

        fig.tight_layout()
        return _savefig(fig, 'merton_coefficient_comparison.png')
    except Exception as e:
        logger.warning(f"plot_merton_comparison failed: {e}")
        return None


def plot_heston_variance_slicing(heston_result):
    """Plot discovered diffusion coefficient vs variance level."""
    try:
        v_list = np.array(heston_result['v_list'])
        disc_diff = np.array(heston_result['discovered_diffusion_coeffs'])
        true_diff = np.array(heston_result['true_diffusion_coeffs'])
        r2 = heston_result['linearity_r2']
        slope = heston_result['linear_fit_slope']
        intercept = heston_result['linear_fit_intercept']

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.scatter(v_list, disc_diff, color='steelblue', s=50, zorder=3,
                   label='Discovered', edgecolors='white', linewidth=0.5)
        ax.plot(v_list, true_diff, 'darkorange', linewidth=2, label='True',
                zorder=2)

        # Linear fit line
        v_fit = np.linspace(v_list.min(), v_list.max(), 100)
        ax.plot(v_fit, slope * v_fit + intercept, 'r--', linewidth=1.5,
                label=f'Linear fit (slope={slope:.4f})', zorder=1)

        # Annotate R^2
        ax.text(0.05, 0.95,
                f'R$^2$ = {r2:.6f}\nslope = {slope:.4f}\nintercept = {intercept:.4e}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Variance Level $v$', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('Diffusion Coefficient', fontsize=FONTSIZE_LABEL)
        ax.set_title('Heston: Diffusion vs Variance', fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=10)
        fig.tight_layout()
        return _savefig(fig, 'heston_variance_slicing.png')
    except Exception as e:
        logger.warning(f"plot_heston_variance_slicing failed: {e}")
        return None


def plot_ablation_heatmap(expansion_results):
    """Heatmap: active terms (rows) vs library size (columns), color = coefficient value."""
    try:
        # Collect all term names across levels (preserving order of first appearance)
        all_terms = []
        seen = set()
        for res in expansion_results:
            for name in res['term_names']:
                if name not in seen:
                    all_terms.append(name)
                    seen.add(name)

        n_rows = len(all_terms)
        n_cols = len(expansion_results)

        matrix = np.full((n_rows, n_cols), np.nan)
        col_labels = []
        for col, res in enumerate(expansion_results):
            col_labels.append(res.get('level', f'Level {col}'))
            names = res['term_names']
            coeffs = np.array(res['coefficients'])
            active = np.array(res['active_mask'])
            for i, name in enumerate(names):
                row = all_terms.index(name)
                if active[i]:
                    matrix[row, col] = coeffs[i]

        fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.5 + 2), max(5, n_rows * 0.5 + 2)))

        # Mask NaN for white cells
        masked = np.ma.masked_invalid(matrix)
        vabs = max(abs(np.nanmin(matrix[np.isfinite(matrix)])),
                   abs(np.nanmax(matrix[np.isfinite(matrix)]))) if np.any(np.isfinite(matrix)) else 1.0
        im = ax.pcolormesh(np.arange(n_cols + 1), np.arange(n_rows + 1),
                           masked, cmap='RdBu_r', vmin=-vabs, vmax=vabs,
                           shading='flat', edgecolors='grey', linewidth=0.5)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Coefficient Value', fontsize=10)

        # Annotate cells
        for r in range(n_rows):
            for c in range(n_cols):
                if np.isfinite(matrix[r, c]):
                    ax.text(c + 0.5, r + 0.5, f'{matrix[r, c]:.4f}',
                            ha='center', va='center', fontsize=8)

        # Tick labels
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_xticklabels(col_labels, fontsize=10)
        ax.set_yticks(np.arange(n_rows) + 0.5)

        # Bold the three true-term row labels
        true_term_set = {'V', 'S*dV/dS', 'S2*d2V/dS2'}
        ylabels = []
        for name in all_terms:
            ylabels.append(name)
        ax.set_yticklabels(ylabels, fontsize=10)
        for tick_label in ax.get_yticklabels():
            if tick_label.get_text() in true_term_set:
                tick_label.set_fontweight('bold')
                tick_label.set_color('darkred')

        ax.set_xlabel('Library Level', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('Term', fontsize=FONTSIZE_LABEL)
        ax.set_title('Library Expansion Ablation', fontsize=FONTSIZE_TITLE)
        ax.invert_yaxis()
        fig.tight_layout()
        return _savefig(fig, 'ablation_library_heatmap.png')
    except Exception as e:
        logger.warning(f"plot_ablation_heatmap failed: {e}")
        return None


def plot_ablation_condition_number(expansion_results):
    """Bar chart: condition number vs library size."""
    try:
        levels = []
        cond_numbers = []
        for res in expansion_results:
            levels.append(res.get('level', '?'))
            cond_numbers.append(res.get('condition_number', np.nan))

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(levels))
        bars = ax.bar(x, cond_numbers, color='teal', alpha=0.85)

        for bar, cn in zip(bars, cond_numbers):
            if np.isfinite(cn):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{cn:.1f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(levels, fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('Condition Number', fontsize=FONTSIZE_LABEL)
        ax.set_xlabel('Library Level', fontsize=FONTSIZE_LABEL)
        ax.set_title('Condition Number vs Library Size', fontsize=FONTSIZE_TITLE)
        ax.set_yscale('log')
        fig.tight_layout()
        return _savefig(fig, 'ablation_condition_numbers.png')
    except Exception as e:
        logger.warning(f"plot_ablation_condition_number failed: {e}")
        return None


def plot_real_iv_surface(option_data, surface_data, ticker):
    """Plot implied volatility surface (heatmap) and reconstructed price surface (3D)."""
    try:
        iv_surface = np.array(surface_data['iv_surface'])
        K_grid = np.array(surface_data['K_grid'])
        tau_grid = np.array(surface_data['tau_grid'])
        V_surface = np.array(surface_data['V_surface'])

        fig = plt.figure(figsize=(14, 6))

        # Left panel: IV heatmap
        ax1 = fig.add_subplot(1, 2, 1)
        im = ax1.pcolormesh(tau_grid, K_grid, iv_surface, cmap='viridis',
                            shading='auto')
        cbar = fig.colorbar(im, ax=ax1)
        cbar.set_label('Implied Volatility', fontsize=10)

        # Overlay raw data scatter if available
        if option_data is not None:
            strikes = np.array(option_data.get('strikes', []))
            tau = np.array(option_data.get('tau', []))
            ivs = np.array(option_data.get('implied_vols', []))
            if len(strikes) > 0 and len(tau) > 0:
                ax1.scatter(tau, strikes, c=ivs, cmap='viridis',
                            edgecolors='white', linewidth=0.5, s=20,
                            zorder=3, vmin=iv_surface.min(),
                            vmax=iv_surface.max())

        ax1.set_xlabel(r'$\tau$ (years to expiry)', fontsize=FONTSIZE_LABEL)
        ax1.set_ylabel('Strike ($)', fontsize=FONTSIZE_LABEL)
        ax1.set_title(f'{ticker} Implied Volatility Surface',
                      fontsize=FONTSIZE_TITLE)

        # Right panel: 3D price surface
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        K_mesh, tau_mesh = np.meshgrid(K_grid, tau_grid, indexing='ij')
        step_k = max(1, len(K_grid) // 40)
        step_t = max(1, len(tau_grid) // 40)
        ax2.plot_surface(K_mesh[::step_k, ::step_t],
                         tau_mesh[::step_k, ::step_t],
                         V_surface[::step_k, ::step_t],
                         cmap='RdYlBu_r', alpha=0.9, edgecolor='none')
        ax2.set_xlabel('Strike ($)', fontsize=10)
        ax2.set_ylabel(r'$\tau$ (years)', fontsize=10)
        ax2.set_zlabel('Option Price ($)', fontsize=10)
        ax2.set_title(f'{ticker} Price Surface', fontsize=FONTSIZE_TITLE)

        fig.tight_layout()
        return _savefig(fig, f'real_iv_surface_{ticker.lower()}.png')
    except Exception as e:
        logger.warning(f"plot_real_iv_surface failed: {e}")
        return None


def plot_real_sindy_comparison(real_results):
    """Compare SINDy coefficients from real data across tickers."""
    try:
        per_ticker = real_results['per_ticker_results']
        tickers = sorted(per_ticker.keys())

        if not tickers:
            logger.warning("plot_real_sindy_comparison: no ticker results")
            return None

        # We show the 3 true-term coefficients (indices 0, 3, 4)
        n_terms = len(_TRUE_COEFF_LABELS)
        n_tickers = len(tickers)
        x = np.arange(n_tickers)
        total_width = 0.8
        bar_w = total_width / n_terms

        fig, ax = plt.subplots(figsize=(max(8, n_tickers * 2), 6))

        colors = ['steelblue', 'darkorange', 'mediumpurple']
        for t_idx, (coeff_idx, label, color) in enumerate(
                zip(_TRUE_COEFF_INDICES, _TRUE_COEFF_LABELS, colors)):
            vals = []
            for ticker in tickers:
                coeffs = np.array(
                    per_ticker[ticker]['sindy_result']['discovered_coefficients'])
                vals.append(coeffs[coeff_idx])
            offset = x - total_width / 2 + bar_w * t_idx + bar_w / 2
            ax.bar(offset, vals, bar_w, label=label, color=color, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(tickers, fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('Coefficient Value', fontsize=FONTSIZE_LABEL)
        ax.set_title('SINDy Coefficients from Real Option Data',
                      fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=10)
        ax.axhline(y=0, color='black', linewidth=0.5)
        fig.tight_layout()
        return _savefig(fig, 'real_sindy_comparison.png')
    except Exception as e:
        logger.warning(f"plot_real_sindy_comparison failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Fix 2: Noise-vs-smoothing visualizations
# ---------------------------------------------------------------------------


def plot_noise_smoothing_matrix(matrix_results):
    """
    Heatmap of R-squared values across noise levels and smoothing settings.

    Parameters
    ----------
    matrix_results : list of dict
        Output of ``run_noise_smoothing_matrix``.  Each dict must have
        keys *noise_pct*, *smoothing*, and *r2*.

    Returns
    -------
    str
        Path to saved figure.
    """
    try:
        # Extract unique noise levels and smoothing labels (preserving order)
        noise_levels = []
        smoothing_labels = []
        seen_noise = set()
        seen_smooth = set()
        for row in matrix_results:
            n = row['noise_pct']
            s = row['smoothing']
            if n not in seen_noise:
                noise_levels.append(n)
                seen_noise.add(n)
            if s not in seen_smooth:
                smoothing_labels.append(s)
                seen_smooth.add(s)

        # Build R2 matrix  (rows = noise, cols = smoothing)
        r2_matrix = np.full((len(noise_levels), len(smoothing_labels)), np.nan)
        noise_idx = {n: i for i, n in enumerate(noise_levels)}
        smooth_idx = {s: i for i, s in enumerate(smoothing_labels)}
        for row in matrix_results:
            i = noise_idx[row['noise_pct']]
            j = smooth_idx[row['smoothing']]
            r2_matrix[i, j] = row['r2']

        fig, ax = plt.subplots(figsize=(max(8, len(smoothing_labels) * 1.8),
                                        max(5, len(noise_levels) * 1.2)))
        sns.heatmap(
            r2_matrix, annot=True, fmt='.4f', cmap='YlGnBu',
            xticklabels=smoothing_labels,
            yticklabels=[f'{n:.0%}' for n in noise_levels],
            ax=ax, vmin=0, vmax=1,
            linewidths=0.5, linecolor='grey',
        )
        ax.set_xlabel('Smoothing (window, poly)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('Noise Level', fontsize=FONTSIZE_LABEL)
        ax.set_title('R$^2$ vs Noise Level and Smoothing', fontsize=FONTSIZE_TITLE)
        fig.tight_layout()
        return _savefig(fig, 'noise_smoothing_matrix.png')
    except Exception as e:
        logger.warning(f"plot_noise_smoothing_matrix failed: {e}")
        return None


def plot_grid_resolution_vs_noise(grid_results):
    """
    Line plot of R-squared vs grid size for clean and noisy data.

    Parameters
    ----------
    grid_results : list of dict
        Output of ``run_grid_resolution_vs_noise``.  Each dict must have
        keys *grid_size*, *r2_clean*, and *r2_noisy*.

    Returns
    -------
    str
        Path to saved figure.
    """
    try:
        grid_sizes = [r['grid_size'] for r in grid_results]
        r2_clean = [r['r2_clean'] for r in grid_results]
        r2_noisy = [r['r2_noisy'] for r in grid_results]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(grid_sizes, r2_clean, 'go-', linewidth=2, markersize=8,
                label='Clean data')
        ax.plot(grid_sizes, r2_noisy, 'rs--', linewidth=2, markersize=8,
                label=f'Noisy ({grid_results[0]["noise"]:.0%})')

        ax.set_xlabel('Grid Size (n)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('R$^2$', fontsize=FONTSIZE_LABEL)
        ax.set_title('SINDy R$^2$ vs Grid Resolution', fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=11)
        ax.set_xticks(grid_sizes)

        # Annotate correct_structure status
        for r in grid_results:
            marker_clean = 'C' if r['correct_structure_clean'] else 'X'
            marker_noisy = 'C' if r['correct_structure_noisy'] else 'X'
            ax.annotate(
                marker_clean, (r['grid_size'], r['r2_clean']),
                textcoords='offset points', xytext=(10, 5),
                fontsize=9, color='green',
            )
            ax.annotate(
                marker_noisy, (r['grid_size'], r['r2_noisy']),
                textcoords='offset points', xytext=(10, -10),
                fontsize=9, color='red',
            )

        fig.tight_layout()
        return _savefig(fig, 'grid_resolution_vs_noise.png')
    except Exception as e:
        logger.warning(f"plot_grid_resolution_vs_noise failed: {e}")
        return None


def plot_smoothing_bias_variance(ablation_results, true_coeffs=None):
    """
    Two-panel plot: coefficient bias and R-squared vs smoothing window.

    Parameters
    ----------
    ablation_results : list of dict
        Output of ``run_smoothing_ablation``.  Each dict must have keys
        *smoothing*, *coefficients* (ndarray(5,)), and *r2*.
    true_coeffs : array-like of length 3 or None
        True values for [V, S*dV/dS, S2*d2V/dS2].  Defaults to the
        standard BS values [0.05, -0.05, -0.02].

    Returns
    -------
    str
        Path to saved figure.
    """
    try:
        if true_coeffs is None:
            true_coeffs = np.array([0.05, -0.05, -0.02])
        else:
            true_coeffs = np.asarray(true_coeffs)

        labels = [r['smoothing'] for r in ablation_results]
        r2_vals = [r['r2'] for r in ablation_results]

        # Extract the three key coefficients (indices 0, 3, 4)
        coeff_V = [np.asarray(r['coefficients'])[0] for r in ablation_results]
        coeff_SdVdS = [np.asarray(r['coefficients'])[3] for r in ablation_results]
        coeff_S2d2VdS2 = [np.asarray(r['coefficients'])[4] for r in ablation_results]

        x = np.arange(len(labels))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # --- Left panel: coefficient values vs smoothing ---
        ax1.plot(x, coeff_V, 'o-', color='steelblue', linewidth=1.5,
                 markersize=7, label='V')
        ax1.plot(x, coeff_SdVdS, 's-', color='darkorange', linewidth=1.5,
                 markersize=7, label='S*dV/dS')
        ax1.plot(x, coeff_S2d2VdS2, '^-', color='mediumpurple', linewidth=1.5,
                 markersize=7, label=r'S$^2$*d$^2$V/dS$^2$')

        # Horizontal lines at true values
        ax1.axhline(y=true_coeffs[0], color='steelblue', linestyle='--',
                     alpha=0.5, linewidth=1)
        ax1.axhline(y=true_coeffs[1], color='darkorange', linestyle='--',
                     alpha=0.5, linewidth=1)
        ax1.axhline(y=true_coeffs[2], color='mediumpurple', linestyle='--',
                     alpha=0.5, linewidth=1)

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=10, rotation=30, ha='right')
        ax1.set_xlabel('Smoothing (window, poly)', fontsize=FONTSIZE_LABEL)
        ax1.set_ylabel('Coefficient Value', fontsize=FONTSIZE_LABEL)
        ax1.set_title('Coefficient Bias vs Smoothing', fontsize=FONTSIZE_TITLE)
        ax1.legend(fontsize=10)
        ax1.axhline(y=0, color='black', linewidth=0.3)

        # --- Right panel: R2 vs smoothing ---
        ax2.bar(x, r2_vals, color='teal', alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=10, rotation=30, ha='right')
        ax2.set_xlabel('Smoothing (window, poly)', fontsize=FONTSIZE_LABEL)
        ax2.set_ylabel('R$^2$', fontsize=FONTSIZE_LABEL)
        ax2.set_title('R$^2$ vs Smoothing', fontsize=FONTSIZE_TITLE)

        # Annotate bars
        for i, v in enumerate(r2_vals):
            ax2.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom',
                     fontsize=8)

        fig.tight_layout()
        return _savefig(fig, 'smoothing_bias_variance.png')
    except Exception as e:
        logger.warning(f"plot_smoothing_bias_variance failed: {e}")
        return None


def plot_neural_vs_fd_derivatives(comparison_result):
    """Compare neural vs FD derivative quality side by side."""
    try:
        df = comparison_result.get('comparison_df')
        if df is None or len(df) == 0:
            return None

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        deriv_names = ['dVdS', 'd2VdS2', 'dVdt']
        deriv_labels = [r'$\partial V/\partial S$', r'$\partial^2 V/\partial S^2$',
                        r'$\partial V/\partial t$']

        method_colors = {'finite_diff': 'steelblue', 'savgol': 'darkorange', 'neural': 'green'}

        for i, (dname, dlabel) in enumerate(zip(deriv_names, deriv_labels)):
            ax = axes[i]
            metric_key = f'{dname}_rel_L2'
            if metric_key not in df.columns:
                continue
            for _, row in df.iterrows():
                method = row['method']
                val = row[metric_key]
                color = method_colors.get(method, 'gray')
                ax.bar(method, val, color=color, alpha=0.8)
            ax.set_ylabel('Relative L2 Error', fontsize=FONTSIZE_LABEL)
            ax.set_title(dlabel, fontsize=FONTSIZE_TITLE)
            ax.set_ylim(bottom=0)

        fig.suptitle('Derivative Quality: FD vs SavGol vs Neural', fontsize=FONTSIZE_TITLE + 1)
        fig.tight_layout()
        return _savefig(fig, 'neural_vs_fd_derivatives.png')
    except Exception as e:
        logger.warning(f"plot_neural_vs_fd_derivatives failed: {e}")
        return None


def plot_neural_sindy_noise_robustness(noise_results_df):
    """Plot R² vs noise level for neural SINDy."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        noise = noise_results_df['noise_level']
        r2 = noise_results_df['r2']

        ax.plot(noise * 100, r2, 'o-', color='green', linewidth=2,
                markersize=8, label='Neural SINDy')
        ax.set_xlabel('Noise Level (%)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('R$^2$', fontsize=FONTSIZE_LABEL)
        ax.set_title('Neural SINDy: R$^2$ vs Noise Level', fontsize=FONTSIZE_TITLE)
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, label='R²=0.99')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return _savefig(fig, 'neural_sindy_noise_robustness.png')
    except Exception as e:
        logger.warning(f"plot_neural_sindy_noise_robustness failed: {e}")
        return None


def plot_neural_sindy_coefficients_vs_noise(noise_results_df):
    """Plot discovered coefficients vs noise level for neural SINDy."""
    try:
        from src.sindy_discovery import TERM_NAMES
        true_coeffs = np.array([0.05, 0.0, 0.0, -0.05, -0.02])

        fig, ax = plt.subplots(figsize=(10, 6))
        noise = noise_results_df['noise_level'] * 100

        coeff_cols = [c for c in noise_results_df.columns if c.startswith('coeff_')]
        colors = ['steelblue', 'darkorange', 'mediumpurple', 'crimson', 'teal']

        for j, col in enumerate(coeff_cols):
            label = TERM_NAMES[j] if j < len(TERM_NAMES) else col
            ax.plot(noise, noise_results_df[col], 'o-', color=colors[j % len(colors)],
                    linewidth=1.5, markersize=6, label=label)
            ax.axhline(y=true_coeffs[j], color=colors[j % len(colors)],
                       linestyle='--', alpha=0.4, linewidth=1)

        ax.set_xlabel('Noise Level (%)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('Coefficient Value', fontsize=FONTSIZE_LABEL)
        ax.set_title('Neural SINDy: Coefficients vs Noise', fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return _savefig(fig, 'neural_sindy_coefficients_vs_noise.png')
    except Exception as e:
        logger.warning(f"plot_neural_sindy_coefficients_vs_noise failed: {e}")
        return None


def plot_weak_sindy_noise_robustness(noise_results_df):
    """Plot R² vs noise level for weak SINDy."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        noise = noise_results_df['noise_level']
        r2 = noise_results_df['r2']

        ax.plot(noise * 100, r2, 's-', color='darkorange', linewidth=2,
                markersize=8, label='Weak SINDy')
        ax.set_xlabel('Noise Level (%)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('R$^2$', fontsize=FONTSIZE_LABEL)
        ax.set_title('Weak SINDy: R$^2$ vs Noise Level', fontsize=FONTSIZE_TITLE)
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, label='R²=0.99')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return _savefig(fig, 'weak_sindy_noise_robustness.png')
    except Exception as e:
        logger.warning(f"plot_weak_sindy_noise_robustness failed: {e}")
        return None


def plot_weak_sindy_coefficients(noise_results_df):
    """Plot discovered coefficients vs noise level for weak SINDy."""
    try:
        from src.sindy_discovery import TERM_NAMES
        true_coeffs = np.array([0.05, 0.0, 0.0, -0.05, -0.02])

        fig, ax = plt.subplots(figsize=(10, 6))
        noise = noise_results_df['noise_level'] * 100

        coeff_cols = [c for c in noise_results_df.columns if c.startswith('coeff_')]
        colors = ['steelblue', 'darkorange', 'mediumpurple', 'crimson', 'teal']

        for j, col in enumerate(coeff_cols):
            label = TERM_NAMES[j] if j < len(TERM_NAMES) else col
            ax.plot(noise, noise_results_df[col], 's-', color=colors[j % len(colors)],
                    linewidth=1.5, markersize=6, label=label)
            ax.axhline(y=true_coeffs[j], color=colors[j % len(colors)],
                       linestyle='--', alpha=0.4, linewidth=1)

        ax.set_xlabel('Noise Level (%)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('Coefficient Value', fontsize=FONTSIZE_LABEL)
        ax.set_title('Weak SINDy: Coefficients vs Noise', fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return _savefig(fig, 'weak_sindy_coefficients.png')
    except Exception as e:
        logger.warning(f"plot_weak_sindy_coefficients failed: {e}")
        return None


def plot_all_methods_noise_comparison(fd_df, neural_df, weak_df):
    """Compare R² vs noise across all three SINDy methods."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if fd_df is not None and 'noise_level' in fd_df.columns:
            r2_col = 'r2_mean' if 'r2_mean' in fd_df.columns else 'r2'
            ax.plot(fd_df['noise_level'] * 100, fd_df[r2_col],
                    'D-', color='steelblue', linewidth=2, markersize=7,
                    label='Standard FD SINDy')

        if neural_df is not None and 'noise_level' in neural_df.columns:
            ax.plot(neural_df['noise_level'] * 100, neural_df['r2'],
                    'o-', color='green', linewidth=2, markersize=7,
                    label='Neural SINDy')

        if weak_df is not None and 'noise_level' in weak_df.columns:
            ax.plot(weak_df['noise_level'] * 100, weak_df['r2'],
                    's-', color='darkorange', linewidth=2, markersize=7,
                    label='Weak SINDy')

        ax.set_xlabel('Noise Level (%)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('R$^2$', fontsize=FONTSIZE_LABEL)
        ax.set_title('All Methods: R$^2$ vs Noise Level', fontsize=FONTSIZE_TITLE)
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, label='R²=0.99')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return _savefig(fig, 'all_methods_noise_comparison.png')
    except Exception as e:
        logger.warning(f"plot_all_methods_noise_comparison failed: {e}")
        return None


def plot_adaptive_strategy_selection(strategy_data):
    """Visualize which strategy the adaptive denoiser selects at each noise level."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        noise_levels = strategy_data['noise_level']
        strategies = strategy_data['strategy']
        r2_scores = strategy_data['r2']

        # Strategy map for colors
        strat_colors = {
            'fd': 'steelblue', 'savgol': 'darkorange', 'neural': 'green',
            'weak': 'crimson', 'unreliable': 'gray'
        }

        # Left: strategy selection vs noise
        unique_strats = sorted(set(strategies))
        strat_to_y = {s: i for i, s in enumerate(unique_strats)}
        colors = [strat_colors.get(s, 'gray') for s in strategies]

        ax1.scatter(np.array(noise_levels) * 100,
                    [strat_to_y[s] for s in strategies],
                    c=colors, s=100, zorder=5)
        ax1.set_yticks(range(len(unique_strats)))
        ax1.set_yticklabels(unique_strats, fontsize=FONTSIZE_LABEL)
        ax1.set_xlabel('Noise Level (%)', fontsize=FONTSIZE_LABEL)
        ax1.set_title('Selected Strategy vs Noise', fontsize=FONTSIZE_TITLE)
        ax1.grid(True, alpha=0.3)

        # Right: R² vs noise with strategy coloring
        for i, (n, r2, s) in enumerate(zip(noise_levels, r2_scores, strategies)):
            ax2.scatter(n * 100, r2, c=strat_colors.get(s, 'gray'), s=80, zorder=5)
        ax2.plot(np.array(noise_levels) * 100, r2_scores, '-', color='gray',
                 alpha=0.5, linewidth=1)
        ax2.set_xlabel('Noise Level (%)', fontsize=FONTSIZE_LABEL)
        ax2.set_ylabel('R$^2$', fontsize=FONTSIZE_LABEL)
        ax2.set_title('Adaptive SINDy: R$^2$ vs Noise', fontsize=FONTSIZE_TITLE)
        ax2.set_ylim(-0.1, 1.05)
        ax2.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=strat_colors.get(s, 'gray'),
                          markersize=10, label=s)
                   for s in unique_strats]
        ax2.legend(handles=handles, fontsize=10)

        fig.tight_layout()
        return _savefig(fig, 'adaptive_strategy_selection.png')
    except Exception as e:
        logger.warning(f"plot_adaptive_strategy_selection failed: {e}")
        return None


def plot_adaptive_vs_oracle(adaptive_df, fd_df=None, neural_df=None, weak_df=None):
    """Compare adaptive denoiser R² vs oracle-selected method at each noise level."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if adaptive_df is not None:
            ax.plot(np.array(adaptive_df['noise_level']) * 100,
                    adaptive_df['r2'], 'k^-', linewidth=2.5, markersize=9,
                    label='Adaptive (auto)', zorder=5)

        if fd_df is not None and 'noise_level' in fd_df.columns:
            r2_col = 'r2_mean' if 'r2_mean' in fd_df.columns else 'r2'
            ax.plot(np.array(fd_df['noise_level']) * 100, fd_df[r2_col],
                    'D--', color='steelblue', linewidth=1.5, markersize=6,
                    alpha=0.7, label='FD SINDy')

        if neural_df is not None and 'noise_level' in neural_df.columns:
            ax.plot(np.array(neural_df['noise_level']) * 100, neural_df['r2'],
                    'o--', color='green', linewidth=1.5, markersize=6,
                    alpha=0.7, label='Neural SINDy')

        if weak_df is not None and 'noise_level' in weak_df.columns:
            ax.plot(np.array(weak_df['noise_level']) * 100, weak_df['r2'],
                    's--', color='darkorange', linewidth=1.5, markersize=6,
                    alpha=0.7, label='Weak SINDy')

        ax.set_xlabel('Noise Level (%)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('R$^2$', fontsize=FONTSIZE_LABEL)
        ax.set_title('Adaptive vs Individual Methods', fontsize=FONTSIZE_TITLE)
        ax.set_ylim(-0.1, 1.05)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return _savefig(fig, 'adaptive_vs_oracle.png')
    except Exception as e:
        logger.warning(f"plot_adaptive_vs_oracle failed: {e}")
        return None


def plot_real_data_misspecification(real_results):
    """Plot BS deviation scores and cross-method comparison for real data."""
    try:
        per_ticker = real_results.get('per_ticker_results', {})
        if not per_ticker:
            return None

        tickers = list(per_ticker.keys())
        n_tickers = len(tickers)

        fig, axes = plt.subplots(1, min(n_tickers, 3), figsize=(5 * min(n_tickers, 3), 5))
        if n_tickers == 1:
            axes = [axes]

        from src.sindy_discovery import TERM_NAMES
        true_coeffs = np.array([0.05, 0.0, 0.0, -0.05, -0.02])

        for idx, ticker in enumerate(tickers[:3]):
            ax = axes[idx]
            res = per_ticker[ticker]
            sindy = res.get('sindy_result', {})
            disc = sindy.get('discovered_coefficients', np.zeros(5))
            if disc is None:
                disc = np.zeros(5)

            x = np.arange(5)
            ax.bar(x - 0.15, true_coeffs, 0.3, color='steelblue', alpha=0.7, label='BS True')
            ax.bar(x + 0.15, disc, 0.3, color='darkorange', alpha=0.7, label='Discovered')
            ax.set_xticks(x)
            ax.set_xticklabels([n.replace('*', '\n') for n in TERM_NAMES],
                               fontsize=8, rotation=30, ha='right')
            ax.set_ylabel('Coefficient', fontsize=FONTSIZE_LABEL)
            ax.set_title(f'{ticker}', fontsize=FONTSIZE_TITLE)
            ax.legend(fontsize=8)
            ax.axhline(y=0, color='black', linewidth=0.5)

            # Add deviation score
            dev = res.get('bs_deviation_score', None)
            if dev is not None:
                ax.text(0.95, 0.95, f'BS dev: {dev:.2f}',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle('Real Data: BS Misspecification Diagnostic', fontsize=FONTSIZE_TITLE + 1)
        fig.tight_layout()
        return _savefig(fig, 'real_data_misspecification.png')
    except Exception as e:
        logger.warning(f"plot_real_data_misspecification failed: {e}")
        return None


# ── Fix 1 plots ──────────────────────────────────────────────────────────


def plot_r2_clean_vs_noisy(all_methods_df):
    """Neural SINDy: R²(clean) vs R²(noisy) across noise levels.

    Shows how R²(noisy) misleadingly increases with noise while R²(clean)
    reveals the true degradation.
    """
    try:
        df = all_methods_df[all_methods_df['method'] == 'neural'].copy()
        if len(df) == 0:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df['noise_pct'] * 100, df['r2_noisy'], 'o-', color='coral',
                linewidth=2, markersize=6, label='R²(noisy) — fit to noisy target')
        ax.plot(df['noise_pct'] * 100, df['r2_clean'], 's-', color='steelblue',
                linewidth=2, markersize=6, label='R²(clean) — true accuracy')

        ax.set_xlabel('Noise Level (%)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('R² Score', fontsize=FONTSIZE_LABEL)
        ax.set_title('Neural SINDy: R²(clean) vs R²(noisy)', fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=10)
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _savefig(fig, 'neural_sindy_r2_clean_vs_noisy.png')
    except Exception as e:
        logger.warning(f"plot_r2_clean_vs_noisy failed: {e}")
        return None


def plot_all_methods_r2_clean(all_methods_df):
    """All 4 methods: R²(clean) across noise levels."""
    try:
        colors = {'fd': '#d62728', 'savgol': '#ff7f0e', 'neural': '#1f77b4', 'weak': '#2ca02c'}
        markers = {'fd': 'v', 'savgol': '^', 'neural': 'o', 'weak': 's'}
        labels = {'fd': 'Finite Diff', 'savgol': 'Savitzky-Golay', 'neural': 'Neural', 'weak': 'Weak'}

        fig, ax = plt.subplots(figsize=(9, 5))
        for method in ['fd', 'savgol', 'neural', 'weak']:
            mdf = all_methods_df[all_methods_df['method'] == method].sort_values('noise_pct')
            if len(mdf) == 0:
                continue
            ax.plot(mdf['noise_pct'] * 100, mdf['r2_clean'],
                    f'{markers[method]}-', color=colors[method],
                    linewidth=2, markersize=6, label=labels[method])

        ax.set_xlabel('Noise Level (%)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('R²(clean)', fontsize=FONTSIZE_LABEL)
        ax.set_title('All Methods: R²(clean) vs Noise', fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=10)
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _savefig(fig, 'all_methods_r2_clean.png')
    except Exception as e:
        logger.warning(f"plot_all_methods_r2_clean failed: {e}")
        return None


def plot_all_methods_coeff_error(all_methods_df):
    """All methods: max coefficient relative error across noise levels."""
    try:
        colors = {'fd': '#d62728', 'savgol': '#ff7f0e', 'neural': '#1f77b4', 'weak': '#2ca02c'}
        markers = {'fd': 'v', 'savgol': '^', 'neural': 'o', 'weak': 's'}
        labels = {'fd': 'Finite Diff', 'savgol': 'Savitzky-Golay', 'neural': 'Neural', 'weak': 'Weak'}

        fig, ax = plt.subplots(figsize=(9, 5))
        for method in ['fd', 'savgol', 'neural', 'weak']:
            mdf = all_methods_df[all_methods_df['method'] == method].sort_values('noise_pct')
            if len(mdf) == 0:
                continue
            ax.plot(mdf['noise_pct'] * 100, mdf['max_rel_err'] * 100,
                    f'{markers[method]}-', color=colors[method],
                    linewidth=2, markersize=6, label=labels[method])

        ax.axhline(y=10, color='gray', linestyle='--', alpha=0.7, label='10% threshold')
        ax.axhline(y=20, color='gray', linestyle=':', alpha=0.7, label='20% threshold')
        ax.set_xlabel('Noise Level (%)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('Max Coefficient Relative Error (%)', fontsize=FONTSIZE_LABEL)
        ax.set_title('Coefficient Accuracy vs Noise', fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=9)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _savefig(fig, 'all_methods_coeff_error.png')
    except Exception as e:
        logger.warning(f"plot_all_methods_coeff_error failed: {e}")
        return None


def plot_neural_sindy_bias_analysis(all_methods_df):
    """Neural SINDy bias analysis: per-coefficient errors across noise."""
    try:
        df = all_methods_df[all_methods_df['method'] == 'neural'].sort_values('noise_pct')
        if len(df) == 0:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: coefficient values vs noise
        ax1.axhline(y=0.05, color='steelblue', linestyle='--', alpha=0.5, label='True V=r')
        ax1.axhline(y=-0.05, color='darkorange', linestyle='--', alpha=0.5, label='True S*dV/dS=-r')
        ax1.axhline(y=-0.02, color='green', linestyle='--', alpha=0.5, label='True S²d²V/dS²')

        ax1.plot(df['noise_pct'] * 100, df['coeff_V'], 'o-', color='steelblue',
                 linewidth=2, label='Discovered V')
        ax1.plot(df['noise_pct'] * 100, df['coeff_SdVdS'], 's-', color='darkorange',
                 linewidth=2, label='Discovered S*dV/dS')
        ax1.plot(df['noise_pct'] * 100, df['coeff_S2d2VdS2'], '^-', color='green',
                 linewidth=2, label='Discovered S²d²V/dS²')

        ax1.set_xlabel('Noise Level (%)', fontsize=FONTSIZE_LABEL)
        ax1.set_ylabel('Coefficient Value', fontsize=FONTSIZE_LABEL)
        ax1.set_title('Neural SINDy: Coefficients vs Noise', fontsize=FONTSIZE_TITLE)
        ax1.legend(fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)

        # Right: per-coefficient relative errors
        ax2.plot(df['noise_pct'] * 100, df['rel_err_V'] * 100, 'o-', color='steelblue',
                 linewidth=2, label='V (r)')
        ax2.plot(df['noise_pct'] * 100, df['rel_err_SdVdS'] * 100, 's-', color='darkorange',
                 linewidth=2, label='S*dV/dS (-r)')
        ax2.plot(df['noise_pct'] * 100, df['rel_err_S2d2VdS2'] * 100, '^-', color='green',
                 linewidth=2, label='S²d²V/dS² (-0.5σ²)')

        ax2.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Noise Level (%)', fontsize=FONTSIZE_LABEL)
        ax2.set_ylabel('Relative Error (%)', fontsize=FONTSIZE_LABEL)
        ax2.set_title('Neural SINDy: Per-Coefficient Errors', fontsize=FONTSIZE_TITLE)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return _savefig(fig, 'neural_sindy_bias_analysis.png')
    except Exception as e:
        logger.warning(f"plot_neural_sindy_bias_analysis failed: {e}")
        return None


# ── Fix 2 plots ──────────────────────────────────────────────────────────


def plot_adaptive_recalibrated(adaptive_df, all_methods_df):
    """Adaptive denoiser R²(clean) after recalibration vs oracle best."""
    try:
        if len(adaptive_df) == 0 or len(all_methods_df) == 0:
            return None

        fig, ax = plt.subplots(figsize=(9, 5))

        # Adaptive R²(clean)
        r2_col = 'r2_clean' if 'r2_clean' in adaptive_df.columns else 'r2'
        ax.plot(adaptive_df['noise_level'] * 100, adaptive_df[r2_col],
                'D-', color='purple', linewidth=2, markersize=7,
                label='Adaptive (recalibrated)')

        # Oracle: best R²(clean) across methods at each noise level
        oracle_noise = []
        oracle_r2 = []
        for nl in sorted(all_methods_df['noise_pct'].unique()):
            nl_df = all_methods_df[all_methods_df['noise_pct'] == nl]
            if len(nl_df) > 0:
                oracle_noise.append(nl * 100)
                oracle_r2.append(nl_df['r2_clean'].max())

        ax.plot(oracle_noise, oracle_r2, 'k--', linewidth=1.5, alpha=0.7,
                label='Oracle (best method)')

        # Per-method lines
        colors = {'fd': '#d62728', 'savgol': '#ff7f0e', 'neural': '#1f77b4', 'weak': '#2ca02c'}
        for method in ['fd', 'savgol', 'neural', 'weak']:
            mdf = all_methods_df[all_methods_df['method'] == method].sort_values('noise_pct')
            if len(mdf) > 0:
                ax.plot(mdf['noise_pct'] * 100, mdf['r2_clean'], ':', color=colors[method],
                        alpha=0.5, linewidth=1, label=f'{method}')

        ax.set_xlabel('Noise Level (%)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('R²(clean)', fontsize=FONTSIZE_LABEL)
        ax.set_title('Adaptive Denoiser (Recalibrated) vs Oracle', fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=9)
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _savefig(fig, 'adaptive_recalibrated.png')
    except Exception as e:
        logger.warning(f"plot_adaptive_recalibrated failed: {e}")
        return None


def plot_method_crossover(all_methods_df):
    """Find where methods cross over — which method is best at each noise level."""
    try:
        if len(all_methods_df) == 0:
            return None

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = {'fd': '#d62728', 'savgol': '#ff7f0e', 'neural': '#1f77b4', 'weak': '#2ca02c'}
        labels = {'fd': 'Finite Diff', 'savgol': 'Savitzky-Golay', 'neural': 'Neural', 'weak': 'Weak'}

        for method in ['fd', 'savgol', 'neural', 'weak']:
            mdf = all_methods_df[all_methods_df['method'] == method].sort_values('noise_pct')
            if len(mdf) == 0:
                continue
            ax.plot(mdf['noise_pct'] * 100, mdf['r2_clean'],
                    'o-', color=colors[method], linewidth=2, markersize=5,
                    label=labels[method])

        # Shade best-method regions
        noise_vals = sorted(all_methods_df['noise_pct'].unique())
        for i in range(len(noise_vals)):
            nl = noise_vals[i]
            nl_df = all_methods_df[all_methods_df['noise_pct'] == nl]
            if len(nl_df) == 0:
                continue
            best = nl_df.loc[nl_df['r2_clean'].idxmax(), 'method']
            x_lo = (noise_vals[i - 1] + nl) / 2 * 100 if i > 0 else 0
            x_hi = (nl + noise_vals[i + 1]) / 2 * 100 if i < len(noise_vals) - 1 else nl * 100 + 2
            ax.axvspan(x_lo, x_hi, alpha=0.08, color=colors[best])

        ax.set_xlabel('Noise Level (%)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('R²(clean)', fontsize=FONTSIZE_LABEL)
        ax.set_title('Method Crossover: Best R²(clean) by Noise Level', fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=10)
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _savefig(fig, 'method_crossover.png')
    except Exception as e:
        logger.warning(f"plot_method_crossover failed: {e}")
        return None


def plot_surface_fitter_comparison(results_df):
    """Bar chart of R²(clean) for each neural surface fitter config."""
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        configs = results_df['config'].values
        r2_vals = results_df['sindy_r2_clean'].values
        colors = ['#e74c3c' if v < 0.9 else '#f39c12' if v < 0.95 else '#2ecc71' for v in r2_vals]

        bars = ax.bar(range(len(configs)), r2_vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('R²(clean)', fontsize=FONTSIZE_LABEL)
        ax.set_title('Neural Surface Fitter: Architecture Comparison', fontsize=FONTSIZE_TITLE)
        ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='R²=0.95')
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='R²=1.0 (FD)')
        ax.legend(fontsize=9)
        ax.set_ylim(min(0, min(r2_vals) - 0.05), 1.05)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, r2_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

        fig.tight_layout()
        return _savefig(fig, 'surface_fitter_comparison.png')
    except Exception as e:
        logger.warning(f"plot_surface_fitter_comparison failed: {e}")
        return None


def plot_weak_sindy_tuning_heatmap(results_df):
    """Heatmap of R²(clean) for weak SINDy n_functions vs width_factor."""
    try:
        pivot = results_df.pivot_table(
            values='r2_clean', index='n_functions', columns='width_factor',
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                        vmin=max(0, pivot.values[np.isfinite(pivot.values)].min() - 0.05),
                        vmax=1.0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel('Width Factor (domain/wf)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('Number of Test Functions', fontsize=FONTSIZE_LABEL)
        ax.set_title('Weak SINDy Tuning: R²(clean) on Clean Data', fontsize=FONTSIZE_TITLE)

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    color = 'white' if val < 0.5 else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            fontsize=8, color=color)

        plt.colorbar(im, ax=ax, label='R²(clean)')
        fig.tight_layout()
        return _savefig(fig, 'weak_sindy_tuning_heatmap.png')
    except Exception as e:
        logger.warning(f"plot_weak_sindy_tuning_heatmap failed: {e}")
        return None


def plot_savgol_weak_crossover(crossover_df, crossover_noise):
    """R²(clean) for SavGol and Weak at fine noise levels with crossover marked."""
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        noise = crossover_df['noise_pct'].values * 100
        ax.plot(noise, crossover_df['savgol_r2_clean'].values, 'o-',
                color='#2ecc71', label='SavGol', linewidth=2, markersize=6)
        ax.plot(noise, crossover_df['weak_r2_clean'].values, 's-',
                color='#9b59b6', label='Weak SINDy', linewidth=2, markersize=6)

        if crossover_noise is not None:
            ax.axvline(x=crossover_noise * 100, color='red', linestyle='--',
                       alpha=0.7, label=f'Crossover at {crossover_noise:.1%}')

        ax.set_xlabel('Noise Level (%)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('R²(clean)', fontsize=FONTSIZE_LABEL)
        ax.set_title('SavGol vs Weak SINDy: Fine Crossover Analysis', fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _savefig(fig, 'savgol_weak_crossover.png')
    except Exception as e:
        logger.warning(f"plot_savgol_weak_crossover failed: {e}")
        return None


def plot_adaptive_vs_oracle_v2(adaptive_df, all_methods_df):
    """Adaptive R²(clean) vs oracle R²(clean) at each noise level."""
    try:
        fig, ax = plt.subplots(figsize=(10, 5))

        noise_levels = sorted(adaptive_df['noise_level'].unique())
        adapt_r2s = []
        oracle_r2s = []

        for nl in noise_levels:
            arow = adaptive_df[adaptive_df['noise_level'] == nl]
            adapt_r2 = arow.iloc[0].get('r2_clean', arow.iloc[0].get('r2', 0)) if len(arow) > 0 else 0
            adapt_r2s.append(adapt_r2)

            nl_df = all_methods_df[all_methods_df['noise_pct'] == nl]
            oracle_r2 = nl_df['r2_clean'].max() if len(nl_df) > 0 else 0
            oracle_r2s.append(oracle_r2)

        noise_pct = [n * 100 for n in noise_levels]
        ax.plot(noise_pct, oracle_r2s, 'k--', label='Oracle (best method)', linewidth=2, alpha=0.7)
        ax.plot(noise_pct, adapt_r2s, 'o-', color='#e74c3c', label='Adaptive', linewidth=2, markersize=6)

        # Shade gap
        ax.fill_between(noise_pct, adapt_r2s, oracle_r2s, alpha=0.15, color='red')

        ax.set_xlabel('Noise Level (%)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('R²(clean)', fontsize=FONTSIZE_LABEL)
        ax.set_title('Adaptive Denoiser vs Oracle: R²(clean)', fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _savefig(fig, 'adaptive_vs_oracle_v2.png')
    except Exception as e:
        logger.warning(f"plot_adaptive_vs_oracle_v2 failed: {e}")
        return None


# ======================================================================
# Real data deep analysis plots (Improvements 1-4)
# ======================================================================

def plot_real_pde_interpretation(analyses, merton_coeffs=None):
    """Bar chart comparing BS theory, Merton synthetic, and real data coefficients.

    Parameters
    ----------
    analyses : dict
        Maps ticker -> analysis dict from ``analyze_discovered_pde``.
    merton_coeffs : array-like or None
        Merton-discovered coefficients (length 5) for reference.
    """
    try:
        tickers = list(analyses.keys())
        if not tickers:
            return None

        n_terms = 5
        term_labels = ['V', 'dV/dS', r'd$^2$V/dS$^2$', 'S dV/dS', r'S$^2$ d$^2$V/dS$^2$']

        # Collect data
        bs_theory = np.array(list(analyses.values())[0]['bs_theory_coefficients'])
        real_data = {t: np.array(a['discovered_coefficients']) for t, a in analyses.items()}

        fig, axes = plt.subplots(1, min(len(tickers), 4), figsize=(5 * min(len(tickers), 4), 5),
                                 squeeze=False)
        axes = axes.flatten()

        for idx, ticker in enumerate(tickers[:4]):
            ax = axes[idx]
            x = np.arange(n_terms)
            width = 0.25

            ax.bar(x - width, bs_theory, width, label='BS Theory',
                   color='#2ecc71', alpha=0.85)
            if merton_coeffs is not None:
                ax.bar(x, np.array(merton_coeffs), width, label='Merton Synth.',
                       color='#e67e22', alpha=0.85)
                ax.bar(x + width, real_data[ticker], width, label=f'{ticker} Real',
                       color='#3498db', alpha=0.85)
            else:
                ax.bar(x, real_data[ticker], width, label=f'{ticker} Real',
                       color='#3498db', alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(term_labels, fontsize=8, rotation=30, ha='right')
            ax.set_ylabel('Coefficient', fontsize=FONTSIZE_LABEL)
            ax.set_title(f'{ticker}', fontsize=FONTSIZE_TITLE)
            ax.legend(fontsize=8)
            ax.axhline(y=0, color='black', linewidth=0.5)

        fig.suptitle('PDE Coefficient Interpretation: BS vs Merton vs Real',
                      fontsize=FONTSIZE_TITLE + 1, y=1.02)
        fig.tight_layout()
        return _savefig(fig, 'real_pde_interpretation.png')
    except Exception as e:
        logger.warning(f"plot_real_pde_interpretation failed: {e}")
        return None


def plot_merton_real_bridge(bridge_result):
    """Two panels: coefficient profile comparison and cosine similarity bars.

    Parameters
    ----------
    bridge_result : dict
        From ``merton_real_data_bridge``, containing ``bridge_df`` and ``summary``.
    """
    try:
        bridge_df = bridge_result['bridge_df']
        if bridge_df.empty:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        tickers = bridge_df['ticker'].tolist()
        cos_bs = bridge_df['cos_sim_bs'].values
        cos_merton = bridge_df['cos_sim_merton'].values

        # Left panel: cosine similarity grouped bars
        x = np.arange(len(tickers))
        width = 0.35
        ax1.bar(x - width / 2, cos_bs, width, label='cos(real, BS)',
                color='#2ecc71', alpha=0.85)
        ax1.bar(x + width / 2, cos_merton, width, label='cos(real, Merton)',
                color='#e74c3c', alpha=0.85)

        ax1.set_xticks(x)
        ax1.set_xticklabels(tickers, fontsize=FONTSIZE_LABEL)
        ax1.set_ylabel('Cosine Similarity', fontsize=FONTSIZE_LABEL)
        ax1.set_title('Real Data: Similarity to BS vs Merton', fontsize=FONTSIZE_TITLE)
        ax1.legend(fontsize=10)
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.set_ylim(-1.1, 1.1)

        # Right panel: jump intensity estimates
        jump_est = bridge_df['jump_intensity_est'].values
        colors = ['#e74c3c' if c == 'Merton' else '#2ecc71'
                  for c in bridge_df['closer_to']]
        valid = ~np.isnan(jump_est)
        if np.any(valid):
            ax2.bar(np.array(tickers)[valid], jump_est[valid],
                    color=np.array(colors)[valid], alpha=0.85)
            ax2.set_ylabel('Estimated Jump Intensity', fontsize=FONTSIZE_LABEL)
            ax2.set_title('Jump Intensity Estimates (rough)', fontsize=FONTSIZE_TITLE)
            ax2.axhline(y=0.1, color='gray', linewidth=1, linestyle='--',
                        label=r'Merton $\lambda$=0.1')
            ax2.legend(fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'No valid estimates', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Jump Intensity (N/A)', fontsize=FONTSIZE_TITLE)

        fig.tight_layout()
        return _savefig(fig, 'merton_real_bridge.png')
    except Exception as e:
        logger.warning(f"plot_merton_real_bridge failed: {e}")
        return None


def plot_iv_regime(regime_result):
    """Two panels: sigma vs maturity regime and sigma vs moneyness regime.

    Parameters
    ----------
    regime_result : dict
        From ``iv_regime_analysis``.
    """
    try:
        ticker = regime_result['ticker']
        mat = regime_result['maturity_regimes']
        mon = regime_result['moneyness_regimes']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: maturity term structure
        _plot_regime_bars(ax1, mat, f'{ticker}: Volatility Term Structure',
                          ylabel=r'$\sigma$ (effective)')

        # Right: moneyness smile
        _plot_regime_bars(ax2, mon, f'{ticker}: Volatility Smile/Skew',
                          ylabel=r'$\sigma$ (effective)')

        fig.tight_layout()
        return _savefig(fig, f'iv_regime_{ticker}.png')
    except Exception as e:
        logger.warning(f"plot_iv_regime failed: {e}")
        return None


def _plot_regime_bars(ax, regimes, title, ylabel):
    """Helper: bar chart of discovered vs market sigma per regime."""
    names = []
    sigma_disc = []
    sigma_mkt = []
    n_opts = []

    for r in regimes:
        names.append(r['regime'])
        sd = r.get('sigma_discovered')
        sm = r.get('sigma_market')
        sigma_disc.append(sd if sd is not None else 0)
        sigma_mkt.append(sm if sm is not None else 0)
        n_opts.append(r.get('n_options', 0))

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, sigma_mkt, width, label='Market IV (avg)',
                   color='#3498db', alpha=0.85)
    has_disc = any(s > 0 for s in sigma_disc)
    if has_disc:
        bars2 = ax.bar(x + width / 2, sigma_disc, width, label=r'$\sigma_{discovered}$',
                       color='#e74c3c', alpha=0.85)

    # Annotate sample sizes
    for i, n in enumerate(n_opts):
        ax.text(x[i], max(sigma_mkt[i], sigma_disc[i]) + 0.005,
                f'n={n}', ha='center', fontsize=8, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha='right')
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
    ax.set_title(title, fontsize=FONTSIZE_TITLE)
    ax.legend(fontsize=9)


def plot_dividend_discovery(div_results):
    """Bar chart comparing implied vs actual dividend yields across tickers.

    Parameters
    ----------
    div_results : dict
        Maps ticker -> dict from ``dividend_yield_discovery``.
    """
    try:
        if not div_results:
            return None

        tickers = list(div_results.keys())
        q_implied = [div_results[t]['q_implied'] * 100 for t in tickers]
        q_actual = [
            (div_results[t]['q_actual'] or 0) * 100 for t in tickers
        ]
        plausible = [div_results[t]['plausible'] for t in tickers]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(tickers))
        width = 0.35

        colors_impl = ['#2ecc71' if p else '#e74c3c' for p in plausible]
        ax.bar(x - width / 2, q_actual, width, label='Known q (%)',
               color='#3498db', alpha=0.85)
        ax.bar(x + width / 2, q_implied, width, label='Discovered q (%)',
               color=colors_impl, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(tickers, fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('Dividend Yield (%)', fontsize=FONTSIZE_LABEL)
        ax.set_title('Dividend Yield: Discovered vs Known', fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=10)
        ax.axhline(y=0, color='black', linewidth=0.5)

        # Annotate agreement
        for i, t in enumerate(tickers):
            agree = div_results[t]['agreement']
            label = 'MATCH' if agree else ('plausible' if plausible[i] else 'implausible')
            color = '#2ecc71' if agree else ('#e67e22' if plausible[i] else '#e74c3c')
            y = max(abs(q_implied[i]), abs(q_actual[i])) + 0.15
            ax.text(x[i], y, label, ha='center', fontsize=8, color=color, fontweight='bold')

        fig.tight_layout()
        return _savefig(fig, 'dividend_discovery.png')
    except Exception as e:
        logger.warning(f"plot_dividend_discovery failed: {e}")
        return None


def plot_residual_heatmap(V, S_grid, t_grid, discovered_coefficients,
                          term_names, output_filename, K=100, title=None):
    """
    Compute and plot PDE residual |dV/dt - sum(coef_i * library_i)| as a heatmap.

    Overlays contour lines of V for context, marks the ATM line (S=K) with a
    vertical dashed white line, and annotates the top 3 "hot spots" (residual
    peaks exceeding 3 standard deviations) with red circles.

    Parameters
    ----------
    V : ndarray, shape (n_S, n_t)
        Option price surface.
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    discovered_coefficients : array-like
        Coefficients corresponding to the 5-term library
        [V, dV/dS, d2V/dS2, S*dV/dS, S^2*d2V/dS2].
    term_names : list of str
        Names of the candidate library terms (used in the title).
    output_filename : str
        Filename (e.g., 'residual_clean.png') saved under outputs/figures/.
    K : float, optional
        Strike price for the ATM line. Default 100.
    title : str or None, optional
        Plot title. Defaults to a generic descriptive title.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    # Import here to avoid circular import at module load time.
    from src.sindy_discovery import compute_derivatives

    derivs = compute_derivatives(V, S_grid, t_grid, smooth=False, trim=5)
    V_tr = derivs['V']
    dVdt = derivs['dVdt']
    dVdS = derivs['dVdS']
    d2VdS2 = derivs['d2VdS2']
    S_mesh = derivs['S_mesh']
    S_tr = derivs['S_grid']
    t_tr = derivs['t_grid']

    coeffs = np.asarray(discovered_coefficients, dtype=float).ravel()
    if coeffs.size != 5:
        raise ValueError(
            f"Expected 5 discovered coefficients, got {coeffs.size}"
        )

    # Build the 5-term library directly on the 2D grid (no flatten).
    lib_terms = [
        V_tr,
        dVdS,
        d2VdS2,
        S_mesh * dVdS,
        S_mesh ** 2 * d2VdS2,
    ]
    predicted = sum(c * term for c, term in zip(coeffs, lib_terms))
    residual = np.abs(dVdt - predicted)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))

    extent = [t_tr[0], t_tr[-1], S_tr[0], S_tr[-1]]
    im = ax.imshow(
        residual,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap='YlOrRd',
        interpolation='nearest',
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('|Residual|', fontsize=FONTSIZE_LABEL)

    # Contour lines of V for context.
    try:
        T_mesh_plot, S_mesh_plot = np.meshgrid(t_tr, S_tr, indexing='xy')
        # V_tr is (n_S, n_t); contour expects (X, Y, Z) with Z shape (len(Y), len(X)).
        cs = ax.contour(
            T_mesh_plot, S_mesh_plot, V_tr,
            levels=8, colors='black', alpha=0.35, linewidths=0.6,
        )
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f')
    except Exception as e:
        logger.debug(f"Contour overlay failed: {e}")

    # ATM line at S = K (only if it falls inside the trimmed S range).
    # Note: S is plotted on the y-axis here, so use axhline for the ATM line.
    if S_tr[0] <= K <= S_tr[-1]:
        ax.axhline(y=K, color='white', linestyle='--', linewidth=1.5,
                   alpha=0.9, label=f'ATM (S=K={K:g})')
        ax.legend(loc='upper right', fontsize=9)

    # Annotate top-3 hot spots where residual > 3 sigma.
    res_mean = float(np.mean(residual))
    res_std = float(np.std(residual))
    hot_threshold = res_mean + 3.0 * res_std
    flat = residual.ravel()
    # Sort indices descending by residual magnitude.
    order = np.argsort(flat)[::-1]
    spots_marked = 0
    for idx in order:
        if spots_marked >= 3:
            break
        if flat[idx] <= hot_threshold:
            break
        i, j = np.unravel_index(idx, residual.shape)
        s_pt = S_tr[i]
        t_pt = t_tr[j]
        ax.scatter(
            [t_pt], [s_pt],
            s=180, facecolors='none', edgecolors='red', linewidths=2.0,
            zorder=5,
        )
        spots_marked += 1

    ax.set_xlabel('Time t (years)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Stock Price S ($)', fontsize=FONTSIZE_LABEL)
    if title is None:
        title = 'PDE Residual Heatmap'
    ax.set_title(title, fontsize=FONTSIZE_TITLE)

    fig.tight_layout()
    return _savefig(fig, output_filename)


def generate_all_residual_maps(clean_data=None, noisy_data=None,
                               merton_data=None, real_data_dict=None,
                               clean_sindy=None, noisy_sindy=None,
                               merton_sindy=None, real_sindy=None,
                               K=100):
    """
    Generate residual heatmaps for up to four datasets.

    Each dataset is expected to be a tuple/dict with keys 'V', 'S_grid',
    't_grid' (or a 3-tuple in that order). The corresponding sindy_results
    must contain 'discovered_coefficients' and 'term_names'. Real data may
    be a dict keyed by ticker (e.g. {'SPY': {...}}); only the SPY entry
    is rendered.

    Each plot is wrapped in try/except so a single failure does not abort
    the others. Returns a dict mapping output filename -> saved path (or
    None on failure / skipped).
    """
    def _unpack(data):
        if data is None:
            return None
        if isinstance(data, dict):
            return data.get('V'), data.get('S_grid'), data.get('t_grid')
        if isinstance(data, (tuple, list)) and len(data) >= 3:
            return data[0], data[1], data[2]
        return None

    def _run_one(data, sindy_res, fname, plot_title):
        try:
            unpacked = _unpack(data)
            if unpacked is None or sindy_res is None:
                logger.info(f"Skipping {fname}: missing data or sindy results.")
                return None
            V, S_grid, t_grid = unpacked
            if V is None or S_grid is None or t_grid is None:
                logger.info(f"Skipping {fname}: incomplete data tuple.")
                return None
            coeffs = sindy_res.get('discovered_coefficients')
            term_names = sindy_res.get('term_names')
            if coeffs is None or term_names is None:
                logger.info(
                    f"Skipping {fname}: sindy results missing coefficients/term_names."
                )
                return None
            return plot_residual_heatmap(
                V, S_grid, t_grid, coeffs, term_names, fname,
                K=K, title=plot_title,
            )
        except Exception as e:
            logger.warning(f"generate_all_residual_maps: {fname} failed: {e}")
            return None

    results = {}
    results['residual_clean.png'] = _run_one(
        clean_data, clean_sindy, 'residual_clean.png',
        'PDE Residual: Clean BS Surface',
    )
    results['residual_noisy_5pct.png'] = _run_one(
        noisy_data, noisy_sindy, 'residual_noisy_5pct.png',
        'PDE Residual: Noisy BS Surface (5%)',
    )
    results['residual_merton.png'] = _run_one(
        merton_data, merton_sindy, 'residual_merton.png',
        'PDE Residual: Merton Jump-Diffusion Surface',
    )

    # Real data: only emit SPY plot if SPY entry exists.
    if real_data_dict and isinstance(real_data_dict, dict) and 'SPY' in real_data_dict:
        spy_sindy = None
        if isinstance(real_sindy, dict) and 'SPY' in real_sindy:
            spy_sindy = real_sindy['SPY']
        else:
            spy_sindy = real_sindy
        results['residual_real_spy.png'] = _run_one(
            real_data_dict['SPY'], spy_sindy, 'residual_real_spy.png',
            'PDE Residual: Real SPY Surface',
        )

    return results
