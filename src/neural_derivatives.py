"""
Neural derivative estimation for noise-robust SINDy PDE discovery.

Instead of computing derivatives via finite differences (which amplifies noise
at rate O(noise/h²) for second derivatives), fits a small neural network to
the noisy surface and differentiates via autograd.
"""

import time
import numpy as np
import torch
import torch.nn as nn

from src.utils import set_all_seeds, get_device, setup_logging, safe_relative_error
from src.sindy_discovery import (
    build_candidate_library, stlsq_sweep, format_pde_string,
    TERM_NAMES, check_derivative_quality,
)

logger = setup_logging(__name__)


class SurfaceFitter(nn.Module):
    """
    Small neural network for fitting a smooth approximation to a price surface.

    Architecture: 2 inputs (S, t) -> n_layers hidden layers x width neurons -> tanh -> 1 output V.
    Deliberately smaller than the PINN by default because we only need
    a smooth approximation, not a PDE solution.
    """

    def __init__(self, S_min, S_max, t_min, t_max, V_mean=0.0, V_std=1.0,
                 n_layers=3, width=32):
        super().__init__()

        # Input normalization buffers
        self.register_buffer('S_min', torch.tensor(float(S_min), dtype=torch.float64))
        self.register_buffer('S_max', torch.tensor(float(S_max), dtype=torch.float64))
        self.register_buffer('t_min', torch.tensor(float(t_min), dtype=torch.float64))
        self.register_buffer('t_max', torch.tensor(float(t_max), dtype=torch.float64))

        # Output scaling buffers
        self.register_buffer('V_mean', torch.tensor(float(V_mean), dtype=torch.float64))
        self.register_buffer('V_std', torch.tensor(float(V_std), dtype=torch.float64))

        # Build network: n_layers hidden layers x width neurons, tanh activation
        layers = []
        layers.append(nn.Linear(2, width))
        layers.append(nn.Tanh())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers).double()

        # Xavier uniform initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, S, t):
        """Forward pass with input normalization and output denormalization."""
        S_norm = (S - self.S_min) / (self.S_max - self.S_min + 1e-10)
        t_norm = (t - self.t_min) / (self.t_max - self.t_min + 1e-10)

        x = torch.stack([S_norm, t_norm], dim=-1)
        raw = self.net(x).squeeze(-1)

        return raw * self.V_std + self.V_mean


def fit_surface(V_noisy, S_grid, t_grid, epochs=1500, lr=1e-3, seed=42,
                val_fraction=0.1, patience=500, n_layers=3, width=32):
    """
    Fit a SurfaceFitter network to a noisy price surface.

    Parameters
    ----------
    V_noisy : ndarray, shape (n_S, n_t)
        Noisy price surface.
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    epochs : int
        Maximum training epochs.
    lr : float
        Initial learning rate.
    seed : int
        Random seed.
    val_fraction : float
        Fraction of points held out for early stopping.
    patience : int
        Stop if validation loss hasn't improved for this many epochs.
    n_layers : int
        Number of hidden layers in the surface fitter.
    width : int
        Number of neurons per hidden layer.

    Returns
    -------
    model : SurfaceFitter
        Trained model.
    fit_info : dict
        Training metadata (final_mse, n_epochs, fit_time).
    """
    set_all_seeds(seed)
    device = get_device()

    n_S, n_t = V_noisy.shape
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')

    # Flatten to training pairs
    S_flat = S_mesh.ravel().astype(np.float64)
    t_flat = t_mesh.ravel().astype(np.float64)
    V_flat = V_noisy.ravel().astype(np.float64)

    # Train/val split
    n_total = len(V_flat)
    n_val = max(1, int(n_total * val_fraction))
    idx = np.random.permutation(n_total)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    S_train = torch.tensor(S_flat[train_idx], dtype=torch.float64, device=device)
    t_train = torch.tensor(t_flat[train_idx], dtype=torch.float64, device=device)
    V_train = torch.tensor(V_flat[train_idx], dtype=torch.float64, device=device)

    S_val = torch.tensor(S_flat[val_idx], dtype=torch.float64, device=device)
    t_val = torch.tensor(t_flat[val_idx], dtype=torch.float64, device=device)
    V_val = torch.tensor(V_flat[val_idx], dtype=torch.float64, device=device)

    # Create model
    V_mean = float(np.mean(V_flat))
    V_std = float(np.std(V_flat)) + 1e-10

    model = SurfaceFitter(
        S_min=float(S_grid.min()), S_max=float(S_grid.max()),
        t_min=float(t_grid.min()), t_max=float(t_grid.max()),
        V_mean=V_mean, V_std=V_std,
        n_layers=n_layers, width=width,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=300, factor=0.5, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0
    actual_epochs = 0

    t_start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        V_pred = model(S_train, t_train)
        loss = torch.mean((V_pred - V_train) ** 2)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            V_val_pred = model(S_val, t_val)
            val_loss = torch.mean((V_val_pred - V_val) ** 2).item()

        scheduler.step(val_loss)
        actual_epochs = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break

    fit_time = time.perf_counter() - t_start

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final MSE on all data
    model.eval()
    with torch.no_grad():
        S_all = torch.tensor(S_flat, dtype=torch.float64, device=device)
        t_all = torch.tensor(t_flat, dtype=torch.float64, device=device)
        V_pred_all = model(S_all, t_all)
        final_mse = torch.mean((V_pred_all - torch.tensor(V_flat, dtype=torch.float64, device=device)) ** 2).item()

    logger.info(
        f"Surface fit complete: {actual_epochs} epochs, "
        f"final MSE={final_mse:.6e}, val MSE={best_val_loss:.6e}, "
        f"time={fit_time:.1f}s"
    )

    return model, {
        'final_mse': final_mse,
        'best_val_mse': best_val_loss,
        'n_epochs': actual_epochs,
        'fit_time': fit_time,
    }


def compute_neural_derivatives(model, S_grid, t_grid):
    """
    Compute partial derivatives of a fitted SurfaceFitter via autograd.

    Parameters
    ----------
    model : SurfaceFitter
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)

    Returns
    -------
    dict with keys:
        'V_smooth': ndarray (n_S, n_t) - network's predicted surface
        'dVdt': ndarray (n_S, n_t)
        'dVdS': ndarray (n_S, n_t)
        'd2VdS2': ndarray (n_S, n_t)
    """
    device = get_device()
    model.eval()

    n_S, n_t = len(S_grid), len(t_grid)
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')

    S_flat = torch.tensor(S_mesh.ravel(), dtype=torch.float64, device=device, requires_grad=True)
    t_flat = torch.tensor(t_mesh.ravel(), dtype=torch.float64, device=device, requires_grad=True)

    # Forward pass
    V = model(S_flat, t_flat)

    # dV/dS
    dVdS = torch.autograd.grad(
        V, S_flat, grad_outputs=torch.ones_like(V),
        create_graph=True, retain_graph=True
    )[0]

    # dV/dt
    dVdt = torch.autograd.grad(
        V, t_flat, grad_outputs=torch.ones_like(V),
        create_graph=False, retain_graph=True
    )[0]

    # d2V/dS2
    d2VdS2 = torch.autograd.grad(
        dVdS, S_flat, grad_outputs=torch.ones_like(dVdS),
        create_graph=False, retain_graph=False
    )[0]

    # Convert to numpy and reshape
    V_smooth = V.detach().cpu().numpy().reshape(n_S, n_t)
    dVdt_np = dVdt.detach().cpu().numpy().reshape(n_S, n_t)
    dVdS_np = dVdS.detach().cpu().numpy().reshape(n_S, n_t)
    d2VdS2_np = d2VdS2.detach().cpu().numpy().reshape(n_S, n_t)

    return {
        'V_smooth': V_smooth,
        'dVdt': dVdt_np,
        'dVdS': dVdS_np,
        'd2VdS2': d2VdS2_np,
    }


def sindy_with_neural_derivatives(V_noisy, S_grid, t_grid, true_sigma=None,
                                    true_r=None, fit_epochs=1500, seed=42,
                                    K=100, T=1.0, option_type='call', trim=5,
                                    n_layers=3, width=32, lr=1e-3):
    """
    SINDy PDE discovery using neural derivative estimation.

    Chains: fit_surface -> compute_neural_derivatives -> build_candidate_library
    -> stlsq_sweep -> format results.

    Returns the SAME dict format as discover_pde() for seamless integration.

    Parameters
    ----------
    V_noisy : ndarray, shape (n_S, n_t)
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    true_sigma : float or None
    true_r : float or None
    fit_epochs : int
    seed : int
    K : float
    T : float
    option_type : str
    trim : int
        Number of boundary rows/cols to trim.

    Returns
    -------
    dict with same keys as discover_pde()
    """
    set_all_seeds(seed)

    # Step 1: Fit surface
    model, fit_info = fit_surface(
        V_noisy, S_grid, t_grid, epochs=fit_epochs, lr=lr, seed=seed,
        n_layers=n_layers, width=width,
    )

    # Step 2: Compute neural derivatives on full grid
    neural_derivs = compute_neural_derivatives(model, S_grid, t_grid)

    # Step 3: Trim boundaries (same as standard SINDy)
    s = slice(trim, -trim) if trim > 0 else slice(None)
    V_tr = neural_derivs['V_smooth'][s, s]
    dVdt_tr = neural_derivs['dVdt'][s, s]
    dVdS_tr = neural_derivs['dVdS'][s, s]
    d2VdS2_tr = neural_derivs['d2VdS2'][s, s]
    S_tr = S_grid[s]
    t_tr = t_grid[s]
    S_mesh_tr, _ = np.meshgrid(S_tr, t_tr, indexing='ij')

    # Step 4: Build candidate library and run SINDy
    library = build_candidate_library(V_tr, dVdS_tr, d2VdS2_tr, S_mesh_tr)
    target = dVdt_tr.ravel()
    cond_number = np.linalg.cond(library)

    best, sweep_results = stlsq_sweep(library, target)
    discovered = best['coefficients']

    # True coefficients
    true_coeffs = None
    rel_errors = None
    if true_sigma is not None and true_r is not None:
        true_coeffs = np.array([
            true_r,
            0.0,
            0.0,
            -true_r,
            -0.5 * true_sigma ** 2,
        ])
        rel_errors = safe_relative_error(discovered, true_coeffs)

    # Check derivative quality if analytical params known
    deriv_quality = {}
    if true_sigma is not None and true_r is not None:
        deriv_dict = {
            'V': V_tr, 'dVdt': dVdt_tr, 'dVdS': dVdS_tr, 'd2VdS2': d2VdS2_tr,
            'S_grid': S_tr, 't_grid': t_tr,
            'S_mesh': S_mesh_tr,
            't_mesh': np.meshgrid(S_tr, t_tr, indexing='ij')[1],
        }
        try:
            deriv_quality = check_derivative_quality(
                deriv_dict, K, true_r, true_sigma, T, option_type
            )
        except Exception as e:
            logger.warning(f"Derivative quality check failed: {e}")

    active_terms = [TERM_NAMES[i] for i in range(5) if best['active_mask'][i]]
    pde_str = format_pde_string(discovered, TERM_NAMES)

    logger.info(
        f"Neural SINDy: R²={best['r2']:.6f}, active={best['n_active']}, "
        f"PDE: {pde_str}"
    )

    result = {
        'discovered_coefficients': discovered,
        'true_coefficients': true_coeffs,
        'active_terms': active_terms,
        'term_names': TERM_NAMES,
        'relative_errors': rel_errors,
        'best_threshold': best['threshold'],
        'r2_score': best['r2'],
        'bic': best['bic'],
        'condition_number': cond_number,
        'derivative_quality': deriv_quality,
        'sweep_results': sweep_results,
        'human_readable_pde': pde_str,
        'active_mask': best['active_mask'],
        'n_active': best['n_active'],
    }

    # Attach extra info
    result['fit_info'] = fit_info
    result['neural_derivatives'] = neural_derivs

    return result


def compare_derivative_methods(V_clean, V_noisy, S_grid, t_grid,
                                 K=100, r=0.05, sigma=0.2, T=1.0,
                                 option_type='call', noise_pct=0.05,
                                 fit_epochs=1500, seed=42):
    """
    Compare derivative estimation: finite differences vs Savitzky-Golay vs neural.

    Parameters
    ----------
    V_clean : ndarray, shape (n_S, n_t)
        Clean price surface.
    V_noisy : ndarray, shape (n_S, n_t)
        Noisy price surface.
    S_grid, t_grid : ndarray
    K, r, sigma, T : float
    option_type : str
    noise_pct : float
    fit_epochs : int
    seed : int

    Returns
    -------
    dict with keys 'comparison_df' (DataFrame) and 'derivatives' (dict of arrays).
    """
    import pandas as pd
    from src.sindy_discovery import compute_derivatives
    from src.data_generation import (
        bs_theta_call, bs_theta_put, bs_call_delta, bs_put_delta, bs_gamma
    )

    set_all_seeds(seed)
    trim = 5

    # Analytical derivatives on trimmed grid
    s = slice(trim, -trim)
    S_tr = S_grid[s]
    t_tr = t_grid[s]
    S_mesh_tr, t_mesh_tr = np.meshgrid(S_tr, t_tr, indexing='ij')
    tau_mesh = T - t_mesh_tr

    if option_type == 'call':
        theta_ana = bs_theta_call(S_mesh_tr, K, r, sigma, tau_mesh)
        delta_ana = bs_call_delta(S_mesh_tr, K, r, sigma, tau_mesh)
    else:
        theta_ana = bs_theta_put(S_mesh_tr, K, r, sigma, tau_mesh)
        delta_ana = bs_put_delta(S_mesh_tr, K, r, sigma, tau_mesh)
    gamma_ana = bs_gamma(S_mesh_tr, K, r, sigma, tau_mesh)

    def rel_l2(num, ana):
        denom = np.linalg.norm(ana)
        if denom < 1e-15:
            return 0.0
        return float(np.linalg.norm(num - ana) / denom)

    results = []
    derivatives = {}

    # Method 1: Finite differences on noisy data
    fd_derivs = compute_derivatives(V_noisy, S_grid, t_grid, smooth=False, trim=trim)
    results.append({
        'method': 'finite_diff',
        'noise_pct': noise_pct,
        'dVdt_rel_L2': rel_l2(fd_derivs['dVdt'], theta_ana),
        'dVdS_rel_L2': rel_l2(fd_derivs['dVdS'], delta_ana),
        'd2VdS2_rel_L2': rel_l2(fd_derivs['d2VdS2'], gamma_ana),
    })
    derivatives['fd'] = fd_derivs

    # Method 2: Savitzky-Golay + FD on noisy data
    sg_derivs = compute_derivatives(
        V_noisy, S_grid, t_grid, smooth=True,
        savgol_window=21, savgol_poly=5, trim=trim
    )
    results.append({
        'method': 'savgol',
        'noise_pct': noise_pct,
        'dVdt_rel_L2': rel_l2(sg_derivs['dVdt'], theta_ana),
        'dVdS_rel_L2': rel_l2(sg_derivs['dVdS'], delta_ana),
        'd2VdS2_rel_L2': rel_l2(sg_derivs['d2VdS2'], gamma_ana),
    })
    derivatives['savgol'] = sg_derivs

    # Method 3: Neural derivatives on noisy data
    model, _ = fit_surface(V_noisy, S_grid, t_grid, epochs=fit_epochs, seed=seed)
    neural_d = compute_neural_derivatives(model, S_grid, t_grid)
    # Trim neural derivatives
    neural_trimmed = {
        'dVdt': neural_d['dVdt'][s, s],
        'dVdS': neural_d['dVdS'][s, s],
        'd2VdS2': neural_d['d2VdS2'][s, s],
        'V': neural_d['V_smooth'][s, s],
    }
    results.append({
        'method': 'neural',
        'noise_pct': noise_pct,
        'dVdt_rel_L2': rel_l2(neural_trimmed['dVdt'], theta_ana),
        'dVdS_rel_L2': rel_l2(neural_trimmed['dVdS'], delta_ana),
        'd2VdS2_rel_L2': rel_l2(neural_trimmed['d2VdS2'], gamma_ana),
    })
    derivatives['neural'] = neural_trimmed

    # Add analytical for reference
    derivatives['analytical'] = {
        'dVdt': theta_ana,
        'dVdS': delta_ana,
        'd2VdS2': gamma_ana,
    }

    df = pd.DataFrame(results)
    logger.info(f"Derivative comparison at {noise_pct:.0%} noise:\n{df.to_string(index=False)}")

    return {
        'comparison_df': df,
        'derivatives': derivatives,
        'S_grid_trimmed': S_tr,
        't_grid_trimmed': t_tr,
    }


def diagnose_surface_fitter(V_clean, S_grid, t_grid, configs=None, seed=42,
                             K=100, r=0.05, sigma=0.2, T=1.0,
                             option_type='call', trim=5):
    """
    Test multiple SurfaceFitter configurations to find the one with lowest
    approximation bias on clean data.

    For each config: fit surface, compute neural derivatives via autograd,
    run SINDy, and compute R²(clean) against analytical dV/dt.

    Parameters
    ----------
    V_clean : ndarray, shape (n_S, n_t)
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    configs : list of dict or None
        Each dict has keys: n_layers, width, epochs, lr.
        If None, uses a default sweep.
    seed : int
    K, r, sigma, T : float
    option_type : str
    trim : int

    Returns
    -------
    dict with keys:
        'results_df': DataFrame with per-config metrics
        'best_config': dict with the best configuration
        'best_r2_clean': float
    """
    import pandas as pd
    from src.sindy_discovery import compute_r2_clean, compute_coefficient_metrics
    from src.data_generation import (
        bs_theta_call, bs_theta_put, bs_call_delta, bs_put_delta, bs_gamma,
    )

    set_all_seeds(seed)

    if configs is None:
        configs = [
            {'n_layers': 3, 'width': 32, 'epochs': 2000, 'lr': 1e-3},
            {'n_layers': 3, 'width': 48, 'epochs': 2000, 'lr': 1e-3},
            {'n_layers': 4, 'width': 32, 'epochs': 2000, 'lr': 1e-3},
            {'n_layers': 4, 'width': 48, 'epochs': 2000, 'lr': 1e-3},
            {'n_layers': 4, 'width': 64, 'epochs': 3000, 'lr': 1e-3},
            {'n_layers': 3, 'width': 32, 'epochs': 5000, 'lr': 1e-3},
            {'n_layers': 3, 'width': 64, 'epochs': 3000, 'lr': 5e-4},
            {'n_layers': 5, 'width': 32, 'epochs': 3000, 'lr': 1e-3},
        ]

    # Analytical derivatives for comparison
    s = slice(trim, -trim) if trim > 0 else slice(None)
    S_tr = S_grid[s]
    t_tr = t_grid[s]
    S_mesh_tr, t_mesh_tr = np.meshgrid(S_tr, t_tr, indexing='ij')
    tau_mesh = T - t_mesh_tr

    if option_type == 'call':
        theta_ana = bs_theta_call(S_mesh_tr, K, r, sigma, tau_mesh)
        delta_ana = bs_call_delta(S_mesh_tr, K, r, sigma, tau_mesh)
    else:
        theta_ana = bs_theta_put(S_mesh_tr, K, r, sigma, tau_mesh)
        delta_ana = bs_put_delta(S_mesh_tr, K, r, sigma, tau_mesh)
    gamma_ana = bs_gamma(S_mesh_tr, K, r, sigma, tau_mesh)

    def rel_l2(num, ana):
        denom = np.linalg.norm(ana)
        if denom < 1e-15:
            return 0.0
        return float(np.linalg.norm(num - ana) / denom)

    rows = []
    for cfg in configs:
        nl = cfg['n_layers']
        w = cfg['width']
        ep = cfg['epochs']
        lr_val = cfg['lr']
        label = f"{nl}x{w}_ep{ep}_lr{lr_val}"

        t_start = time.perf_counter()
        try:
            set_all_seeds(seed)
            model, fit_info = fit_surface(
                V_clean, S_grid, t_grid, epochs=ep, lr=lr_val,
                seed=seed, n_layers=nl, width=w,
            )
            neural_derivs = compute_neural_derivatives(model, S_grid, t_grid)

            # Trimmed derivatives
            dVdt_tr = neural_derivs['dVdt'][s, s]
            dVdS_tr = neural_derivs['dVdS'][s, s]
            d2VdS2_tr = neural_derivs['d2VdS2'][s, s]
            V_tr = neural_derivs['V_smooth'][s, s]

            # Derivative L2 errors
            dVdt_l2 = rel_l2(dVdt_tr, theta_ana)
            dVdS_l2 = rel_l2(dVdS_tr, delta_ana)
            d2VdS2_l2 = rel_l2(d2VdS2_tr, gamma_ana)

            # Run SINDy on neural derivatives
            library = build_candidate_library(V_tr, dVdS_tr, d2VdS2_tr, S_mesh_tr)
            target = dVdt_tr.ravel()
            best, _ = stlsq_sweep(library, target)
            coeffs = best['coefficients']

            # R²(clean) and coefficient metrics
            r2_clean = compute_r2_clean(
                coeffs, S_grid, t_grid, K=K, r=r, sigma=sigma, T=T,
                option_type=option_type,
            )
            cm = compute_coefficient_metrics(coeffs, true_r=r, true_sigma=sigma)

            elapsed = time.perf_counter() - t_start

            rows.append({
                'config': label,
                'n_layers': nl,
                'width': w,
                'epochs': ep,
                'lr': lr_val,
                'fit_mse': fit_info['final_mse'],
                'dVdt_rel_L2': dVdt_l2,
                'dVdS_rel_L2': dVdS_l2,
                'd2VdS2_rel_L2': d2VdS2_l2,
                'sindy_r2_clean': r2_clean,
                'max_coeff_err': cm['max_coeff_rel_error'],
                'mean_coeff_err': cm['mean_coeff_rel_error'],
                'correct_structure': cm['correct_structure'],
                'time_s': elapsed,
            })
            logger.info(
                f"Config {label}: MSE={fit_info['final_mse']:.2e}, "
                f"R²(clean)={r2_clean:.4f}, d2VdS2_L2={d2VdS2_l2:.4f}, "
                f"time={elapsed:.1f}s"
            )
        except Exception as e:
            elapsed = time.perf_counter() - t_start
            logger.warning(f"Config {label} failed: {e}")
            rows.append({
                'config': label, 'n_layers': nl, 'width': w,
                'epochs': ep, 'lr': lr_val,
                'fit_mse': float('nan'), 'dVdt_rel_L2': float('nan'),
                'dVdS_rel_L2': float('nan'), 'd2VdS2_rel_L2': float('nan'),
                'sindy_r2_clean': float('nan'), 'max_coeff_err': float('nan'),
                'mean_coeff_err': float('nan'), 'correct_structure': False,
                'time_s': elapsed,
            })

    results_df = pd.DataFrame(rows)
    best_idx = results_df['sindy_r2_clean'].idxmax()
    best_row = results_df.loc[best_idx]
    best_config = {
        'n_layers': int(best_row['n_layers']),
        'width': int(best_row['width']),
        'epochs': int(best_row['epochs']),
        'lr': float(best_row['lr']),
    }

    logger.info(
        f"Best neural config: {best_config}, "
        f"R²(clean)={best_row['sindy_r2_clean']:.4f}"
    )

    return {
        'results_df': results_df,
        'best_config': best_config,
        'best_r2_clean': float(best_row['sindy_r2_clean']),
    }
