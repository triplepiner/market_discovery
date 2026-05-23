"""
Physics-Informed Neural Network (PINN) validation for the Black-Scholes PDE.

Trains a PINN to solve the Black-Scholes PDE using discovered coefficients
from the SINDy module, enforcing the PDE as a soft constraint through the
loss function alongside boundary conditions and observed data.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.utils import set_all_seeds, get_device, setup_logging, Timer
from src.data_generation import (
    generate_price_surface,
    bs_call_price,
    bs_put_price,
)

logger = setup_logging(__name__)

TERM_NAMES = ['V', 'dV/dS', 'd2V/dS2', 'S*dV/dS', 'S2*d2V/dS2']


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------

class BSPINN(nn.Module):
    """
    Physics-Informed Neural Network for the Black-Scholes PDE.

    Architecture: 2 inputs (S, t) -> 4 hidden layers x 64 neurons with tanh
    activation -> 1 output (option price V).

    Inputs are normalized to [0, 1] and the output is denormalized using
    learnable scale/offset buffers so the network operates on O(1) quantities.

    Parameters
    ----------
    S_min : float
        Minimum stock price in the domain.
    S_max : float
        Maximum stock price in the domain.
    t_min : float
        Minimum calendar time in the domain.
    t_max : float
        Maximum calendar time in the domain.
    V_scale : float
        Scale factor for output denormalization: V = V_raw * V_scale + V_offset.
    V_offset : float
        Offset for output denormalization.
    """

    def __init__(self, S_min, S_max, t_min, t_max, V_scale=1.0, V_offset=0.0):
        super().__init__()

        # Input normalization buffers (not trainable parameters)
        self.register_buffer('S_min', torch.tensor(S_min, dtype=torch.float64))
        self.register_buffer('S_max', torch.tensor(S_max, dtype=torch.float64))
        self.register_buffer('t_min', torch.tensor(t_min, dtype=torch.float64))
        self.register_buffer('t_max', torch.tensor(t_max, dtype=torch.float64))

        # Output denormalization buffers
        self.register_buffer('V_scale', torch.tensor(V_scale, dtype=torch.float64))
        self.register_buffer('V_offset', torch.tensor(V_offset, dtype=torch.float64))

        # Network: 4 hidden layers x 64 neurons, tanh activation
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        ).double()

        # Xavier uniform initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, S, t):
        """
        Forward pass: normalize inputs, run network, denormalize output.

        Parameters
        ----------
        S : Tensor
            Raw stock prices.
        t : Tensor
            Raw calendar times.

        Returns
        -------
        V : Tensor
            Predicted option prices.
        """
        # Normalize to [0, 1]
        S_norm = (S - self.S_min) / (self.S_max - self.S_min + 1e-30)
        t_norm = (t - self.t_min) / (self.t_max - self.t_min + 1e-30)

        x = torch.cat([S_norm, t_norm], dim=-1)
        V_raw = self.net(x)

        # Denormalize
        V = V_raw * self.V_scale + self.V_offset
        return V


# ---------------------------------------------------------------------------
# PDE residual computation
# ---------------------------------------------------------------------------

def compute_pde_residual(model, S, t, discovered_coefficients, term_names):
    """
    Compute the PDE residual using automatic differentiation.

    The Black-Scholes PDE (in calendar time) is:

        dV/dt + c3*S*dV/dS + c4*S^2*d2V/dS2 + c0*V = 0

    where the residual is defined as:

        residual = dV/dt - (c0*V + c1*dV/dS + c2*d2V/dS2 + c3*S*dV/dS + c4*S^2*d2V/dS2)

    Parameters
    ----------
    model : BSPINN
        The neural network model.
    S : Tensor, shape (N, 1), requires_grad=True
        Stock prices.
    t : Tensor, shape (N, 1), requires_grad=True
        Calendar times.
    discovered_coefficients : array-like, length 5
        Coefficients [c0, c1, c2, c3, c4] corresponding to term_names.
    term_names : list of str
        Names of the PDE terms: ['V', 'dV/dS', 'd2V/dS2', 'S*dV/dS', 'S2*d2V/dS2'].

    Returns
    -------
    residual : Tensor, shape (N, 1)
    """
    V = model(S, t)

    # dV/dt
    dV_dt = torch.autograd.grad(
        V, t,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True,
    )[0]

    if dV_dt is None:
        dV_dt = torch.zeros_like(V)

    # dV/dS
    dV_dS = torch.autograd.grad(
        V, S,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True,
    )[0]

    if dV_dS is None:
        dV_dS = torch.zeros_like(V)

    # d2V/dS2
    d2V_dS2 = torch.autograd.grad(
        dV_dS, S,
        grad_outputs=torch.ones_like(dV_dS),
        create_graph=True,
        retain_graph=True,
    )[0]

    if d2V_dS2 is None:
        d2V_dS2 = torch.zeros_like(V)

    # Build the RHS of the PDE: sum of c_i * term_i
    coeffs = np.asarray(discovered_coefficients, dtype=np.float64)

    # Map term names to computed quantities
    term_map = {
        'V': V,
        'dV/dS': dV_dS,
        'd2V/dS2': d2V_dS2,
        'S*dV/dS': S * dV_dS,
        'S2*d2V/dS2': S ** 2 * d2V_dS2,
    }

    rhs = torch.zeros_like(V)
    for i, name in enumerate(term_names):
        if abs(coeffs[i]) > 1e-30:
            rhs = rhs + coeffs[i] * term_map[name]

    residual = dV_dt - rhs
    return residual


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PINNTrainer:
    """
    Trainer for the Black-Scholes PINN.

    Loss function:
        L = lambda_pde * L_pde + lambda_bc * L_bc + lambda_data * L_data

    - L_pde: PDE residual on collocation points (resampled periodically).
    - L_bc: terminal condition + spatial boundary conditions.
    - L_data: supervised loss on a training subset of observed prices.

    Parameters
    ----------
    model : BSPINN
        The network to train.
    S_data : ndarray, shape (N,)
        Stock price observations (flattened).
    t_data : ndarray, shape (N,)
        Time observations (flattened).
    V_data : ndarray, shape (N,)
        Option price observations (flattened).
    discovered_coefficients : array-like, length 5
        PDE coefficients from SINDy.
    term_names : list of str
        PDE term names.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    T : float
        Option maturity.
    option_type : str
        'call' or 'put'.
    lambda_pde : float
        Weight for PDE residual loss.
    lambda_bc : float
        Weight for boundary condition loss.
    lambda_data : float
        Weight for data loss.
    n_collocation : int
        Number of collocation points for PDE residual.
    resample_every : int
        Resample collocation points every this many epochs.
    n_epochs : int
        Total training epochs.
    lr : float
        Initial learning rate.
    log_every : int
        Log training progress every this many epochs.
    """

    def __init__(
        self,
        model,
        S_data,
        t_data,
        V_data,
        discovered_coefficients,
        term_names,
        K=100.0,
        r=0.05,
        T=1.0,
        option_type='call',
        lambda_pde=1.0,
        lambda_bc=10.0,
        lambda_data=1.0,
        n_collocation=10000,
        resample_every=1000,
        n_epochs=10000,
        lr=1e-3,
        log_every=500,
    ):
        self.model = model
        self.discovered_coefficients = np.asarray(discovered_coefficients, dtype=np.float64)
        self.term_names = list(term_names)
        self.K = K
        self.r = r
        self.T = T
        self.option_type = option_type
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_data = lambda_data
        self.n_collocation = n_collocation
        self.resample_every = resample_every
        self.n_epochs = n_epochs
        self.lr = lr
        self.log_every = log_every

        self.device = get_device()

        # Domain bounds (from model buffers)
        self.S_min = float(model.S_min)
        self.S_max = float(model.S_max)
        self.t_min = float(model.t_min)
        self.t_max = float(model.t_max)

        # ------------------------------------------------------------------
        # 60 / 20 / 20 data split
        # ------------------------------------------------------------------
        N = len(S_data)
        indices = np.arange(N)

        idx_train, idx_temp = train_test_split(
            indices, train_size=0.6, random_state=42
        )
        idx_val, idx_test = train_test_split(
            idx_temp, train_size=0.5, random_state=42
        )

        self.S_train = torch.tensor(S_data[idx_train], dtype=torch.float64, device=self.device).unsqueeze(-1)
        self.t_train = torch.tensor(t_data[idx_train], dtype=torch.float64, device=self.device).unsqueeze(-1)
        self.V_train = torch.tensor(V_data[idx_train], dtype=torch.float64, device=self.device).unsqueeze(-1)

        self.S_val = torch.tensor(S_data[idx_val], dtype=torch.float64, device=self.device).unsqueeze(-1)
        self.t_val = torch.tensor(t_data[idx_val], dtype=torch.float64, device=self.device).unsqueeze(-1)
        self.V_val = torch.tensor(V_data[idx_val], dtype=torch.float64, device=self.device).unsqueeze(-1)

        self.S_test = torch.tensor(S_data[idx_test], dtype=torch.float64, device=self.device).unsqueeze(-1)
        self.t_test = torch.tensor(t_data[idx_test], dtype=torch.float64, device=self.device).unsqueeze(-1)
        self.V_test = torch.tensor(V_data[idx_test], dtype=torch.float64, device=self.device).unsqueeze(-1)

        logger.info(
            f"Data split: train={len(idx_train)}, val={len(idx_val)}, "
            f"test={len(idx_test)}"
        )

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=1000,
            factor=0.5,
            min_lr=1e-6,
        )

        # Loss history
        self.history = {
            'total_loss': [],
            'pde_loss': [],
            'bc_loss': [],
            'data_loss': [],
            'val_loss': [],
        }

        # Collocation points (initialized lazily)
        self._S_col = None
        self._t_col = None

        # Validation-based early stopping state
        self._val_increase_count = 0
        self._prev_val_loss = float('inf')

    # ------------------------------------------------------------------
    # Collocation sampling
    # ------------------------------------------------------------------

    def _sample_collocation(self):
        """Sample uniform random collocation points inside the domain."""
        S_col = (
            torch.rand(self.n_collocation, 1, dtype=torch.float64, device=self.device)
            * (self.S_max - self.S_min)
            + self.S_min
        )
        t_col = (
            torch.rand(self.n_collocation, 1, dtype=torch.float64, device=self.device)
            * (self.t_max - self.t_min)
            + self.t_min
        )
        S_col.requires_grad_(True)
        t_col.requires_grad_(True)
        self._S_col = S_col
        self._t_col = t_col

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def _pde_loss(self):
        """Mean squared PDE residual on collocation points."""
        residual = compute_pde_residual(
            self.model,
            self._S_col,
            self._t_col,
            self.discovered_coefficients,
            self.term_names,
        )
        return torch.mean(residual ** 2)

    def _boundary_loss(self):
        """
        Boundary condition loss.

        For a call:
            - Terminal (t = T): V = max(S - K, 0)
            - S_min boundary: V ~ 0
            - S_max boundary: V ~ S - K * exp(-r * (T - t))

        For a put:
            - Terminal (t = T): V = max(K - S, 0)
            - S_min boundary: V ~ K * exp(-r * (T - t))
            - S_max boundary: V ~ 0
        """
        n_bc = 200  # points per boundary segment

        # --- Terminal condition: t close to T ---
        S_term = (
            torch.rand(n_bc, 1, dtype=torch.float64, device=self.device)
            * (self.S_max - self.S_min)
            + self.S_min
        )
        t_term = torch.full((n_bc, 1), self.t_max, dtype=torch.float64, device=self.device)

        V_pred_term = self.model(S_term, t_term)

        if self.option_type == 'call':
            V_true_term = torch.clamp(S_term - self.K, min=0.0)
        else:
            V_true_term = torch.clamp(self.K - S_term, min=0.0)

        loss_terminal = torch.mean((V_pred_term - V_true_term) ** 2)

        # --- S_min boundary ---
        t_bnd = (
            torch.rand(n_bc, 1, dtype=torch.float64, device=self.device)
            * (self.t_max - self.t_min)
            + self.t_min
        )
        S_lo = torch.full((n_bc, 1), self.S_min, dtype=torch.float64, device=self.device)

        V_pred_lo = self.model(S_lo, t_bnd)

        if self.option_type == 'call':
            # Call at S_min ~ 0
            V_true_lo = torch.zeros_like(V_pred_lo)
        else:
            # Put at S_min ~ K * exp(-r*(T-t))
            tau_bnd = self.T - t_bnd
            V_true_lo = self.K * torch.exp(-self.r * tau_bnd)

        loss_lo = torch.mean((V_pred_lo - V_true_lo) ** 2)

        # --- S_max boundary ---
        S_hi = torch.full((n_bc, 1), self.S_max, dtype=torch.float64, device=self.device)
        t_bnd2 = (
            torch.rand(n_bc, 1, dtype=torch.float64, device=self.device)
            * (self.t_max - self.t_min)
            + self.t_min
        )

        V_pred_hi = self.model(S_hi, t_bnd2)

        if self.option_type == 'call':
            # Call at S_max ~ S - K*exp(-r*(T-t))
            tau_bnd2 = self.T - t_bnd2
            V_true_hi = self.S_max - self.K * torch.exp(-self.r * tau_bnd2)
        else:
            # Put at S_max ~ 0
            V_true_hi = torch.zeros_like(V_pred_hi)

        loss_hi = torch.mean((V_pred_hi - V_true_hi) ** 2)

        return loss_terminal + loss_lo + loss_hi

    def _data_loss(self):
        """Mean squared error on training data."""
        V_pred = self.model(self.S_train, self.t_train)
        return torch.mean((V_pred - self.V_train) ** 2)

    def _val_loss(self):
        """Mean squared error on validation data (no grad)."""
        with torch.no_grad():
            V_pred = self.model(self.S_val, self.t_val)
            return torch.mean((V_pred - self.V_val) ** 2).item()

    # ------------------------------------------------------------------
    # Gradient pathology detection
    # ------------------------------------------------------------------

    def _check_gradient_pathology(self):
        """
        Detect gradient pathology by comparing gradient magnitudes
        across loss components.

        Logs a warning if any component's gradient norm is more than
        100x larger or smaller than the others.
        """
        # Compute each loss component
        loss_pde = self.lambda_pde * self._pde_loss()
        loss_bc = self.lambda_bc * self._boundary_loss()
        loss_data = self.lambda_data * self._data_loss()

        norms = {}
        for name, loss_val in [('pde', loss_pde), ('bc', loss_bc), ('data', loss_data)]:
            self.optimizer.zero_grad()
            loss_val.backward(retain_graph=True)
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            norms[name] = total_norm ** 0.5

        self.optimizer.zero_grad()

        max_norm = max(norms.values()) if norms.values() else 1.0
        min_norm = min(norms.values()) if norms.values() else 1.0

        if min_norm > 0 and max_norm / min_norm > 100:
            logger.warning(
                f"Gradient pathology detected: norms = "
                f"pde={norms['pde']:.2e}, bc={norms['bc']:.2e}, "
                f"data={norms['data']:.2e} (ratio {max_norm / min_norm:.1f}x)"
            )
        else:
            logger.info(
                f"Gradient norms: pde={norms['pde']:.2e}, "
                f"bc={norms['bc']:.2e}, data={norms['data']:.2e}"
            )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        """
        Run the full training loop for n_epochs.

        Returns
        -------
        dict
            Training history with all loss components.
        """
        logger.info(
            f"Starting PINN training: {self.n_epochs} epochs, "
            f"lr={self.lr}, lambdas=(pde={self.lambda_pde}, "
            f"bc={self.lambda_bc}, data={self.lambda_data})"
        )

        self.model.train()

        # Initial collocation sampling
        self._sample_collocation()

        for epoch in range(1, self.n_epochs + 1):
            # Resample collocation points periodically
            if epoch > 1 and (epoch - 1) % self.resample_every == 0:
                self._sample_collocation()

            self.optimizer.zero_grad()

            loss_pde = self._pde_loss()
            loss_bc = self._boundary_loss()
            loss_data = self._data_loss()

            total_loss = (
                self.lambda_pde * loss_pde
                + self.lambda_bc * loss_bc
                + self.lambda_data * loss_data
            )

            total_loss.backward()
            self.optimizer.step()

            # Record history
            self.history['total_loss'].append(total_loss.item())
            self.history['pde_loss'].append(loss_pde.item())
            self.history['bc_loss'].append(loss_bc.item())
            self.history['data_loss'].append(loss_data.item())

            # Validation loss
            val_loss = self._val_loss()
            self.history['val_loss'].append(val_loss)

            # Update scheduler with validation loss
            self.scheduler.step(val_loss)

            # Logging
            if epoch % self.log_every == 0 or epoch == 1:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch:5d}/{self.n_epochs}: "
                    f"total={total_loss.item():.6e}, "
                    f"pde={loss_pde.item():.6e}, "
                    f"bc={loss_bc.item():.6e}, "
                    f"data={loss_data.item():.6e}, "
                    f"val={val_loss:.6e}, "
                    f"lr={current_lr:.2e}"
                )

            # Gradient pathology detection every 1000 epochs
            if epoch % 1000 == 0:
                self._check_gradient_pathology()

            # Overfitting check every 2000 epochs
            if epoch % 2000 == 0:
                if val_loss > self._prev_val_loss:
                    self._val_increase_count += 1
                    logger.info(
                        f"Validation loss increased ({self._prev_val_loss:.6e} -> "
                        f"{val_loss:.6e}), count={self._val_increase_count}/3"
                    )
                else:
                    self._val_increase_count = 0
                self._prev_val_loss = val_loss

                if self._val_increase_count >= 3:
                    logger.warning(
                        f"Early stopping at epoch {epoch}: validation loss "
                        f"increased for 3 consecutive checks."
                    )
                    break

        logger.info("PINN training complete.")
        return self.history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_test(self):
        """
        Evaluate the trained model on the test set.

        Returns
        -------
        dict
            Metrics: relative_l2_error, mae, max_error, r2.
        """
        self.model.eval()
        with torch.no_grad():
            V_pred = self.model(self.S_test, self.t_test)
            V_true = self.V_test

            diff = V_pred - V_true

            # Relative L2 error
            l2_error = torch.norm(diff) / (torch.norm(V_true) + 1e-30)

            # MAE
            mae = torch.mean(torch.abs(diff))

            # Max error
            max_err = torch.max(torch.abs(diff))

            # R^2
            ss_res = torch.sum(diff ** 2)
            ss_tot = torch.sum((V_true - torch.mean(V_true)) ** 2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-30)

        metrics = {
            'relative_l2_error': l2_error.item(),
            'mae': mae.item(),
            'max_error': max_err.item(),
            'r2': r2.item(),
        }

        logger.info(
            f"Test-set metrics: rel_L2={metrics['relative_l2_error']:.6e}, "
            f"MAE={metrics['mae']:.6e}, max_err={metrics['max_error']:.6e}, "
            f"R2={metrics['r2']:.6f}"
        )

        return metrics


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _run_sanity_checks(model, S_min, S_max, t_min, t_max, K, r, T, option_type):
    """
    Run sanity checks on the trained PINN predictions.

    Checks:
        1. Non-negative prices across the domain.
        2. Monotonicity: call prices increase in S, put prices decrease in S.
        3. Boundary condition satisfaction.

    Parameters
    ----------
    model : BSPINN
    S_min, S_max, t_min, t_max : float
    K : float
    r : float
    T : float
    option_type : str

    Returns
    -------
    dict
        Results of each sanity check.
    """
    model.eval()
    checks = {}

    # Grid for checking
    n_check = 200
    S_check = torch.linspace(S_min, S_max, n_check, dtype=torch.float64).unsqueeze(-1)
    t_mid = torch.full((n_check, 1), (t_min + t_max) / 2.0, dtype=torch.float64)

    with torch.no_grad():
        V_check = model(S_check, t_mid)

    # 1. Non-negative prices
    n_negative = int(torch.sum(V_check < -1e-8).item())
    checks['non_negative'] = n_negative == 0
    if n_negative > 0:
        logger.warning(
            f"Sanity check FAILED: {n_negative}/{n_check} negative prices "
            f"(min={V_check.min().item():.6f})"
        )
    else:
        logger.info("Sanity check PASSED: all prices non-negative.")

    # 2. Monotonicity
    V_vals = V_check.squeeze().numpy()
    diffs = np.diff(V_vals)

    if option_type == 'call':
        # Call price should be non-decreasing in S
        n_violations = int(np.sum(diffs < -1e-6))
        checks['monotonicity'] = n_violations == 0
        direction = "non-decreasing"
    else:
        # Put price should be non-increasing in S
        n_violations = int(np.sum(diffs > 1e-6))
        checks['monotonicity'] = n_violations == 0
        direction = "non-increasing"

    if n_violations > 0:
        logger.warning(
            f"Sanity check FAILED: {n_violations}/{n_check - 1} monotonicity "
            f"violations (expected {direction} in S)."
        )
    else:
        logger.info(f"Sanity check PASSED: prices are {direction} in S.")

    # 3. Boundary condition satisfaction (terminal condition)
    S_bc = torch.linspace(S_min, S_max, 100, dtype=torch.float64).unsqueeze(-1)
    t_bc = torch.full((100, 1), t_max, dtype=torch.float64)

    with torch.no_grad():
        V_bc = model(S_bc, t_bc).squeeze().numpy()

    if option_type == 'call':
        V_true_bc = np.maximum(S_bc.squeeze().numpy() - K, 0.0)
    else:
        V_true_bc = np.maximum(K - S_bc.squeeze().numpy(), 0.0)

    bc_error = np.mean(np.abs(V_bc - V_true_bc))
    bc_rel_error = bc_error / (np.mean(np.abs(V_true_bc)) + 1e-10)
    checks['bc_satisfaction'] = bc_rel_error < 0.05
    checks['bc_mean_abs_error'] = float(bc_error)
    checks['bc_relative_error'] = float(bc_rel_error)

    if checks['bc_satisfaction']:
        logger.info(
            f"Sanity check PASSED: terminal BC rel error = {bc_rel_error:.4f}"
        )
    else:
        logger.warning(
            f"Sanity check FAILED: terminal BC rel error = {bc_rel_error:.4f} "
            f"(threshold 5%)"
        )

    return checks


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def train_pinn(
    V_surface,
    S_grid,
    t_grid,
    discovered_coefficients,
    term_names=None,
    K=100.0,
    r=0.05,
    sigma=0.2,
    T=1.0,
    option_type='call',
    n_epochs=10000,
    lr=1e-3,
    lambda_pde=1.0,
    lambda_bc=10.0,
    lambda_data=1.0,
):
    """
    Top-level function: set up a BSPINN, train it with PDE + BC + data loss,
    evaluate on held-out test set, and run sanity checks.

    Parameters
    ----------
    V_surface : ndarray, shape (n_S, n_t)
        Option price surface (may include noise).
    S_grid : ndarray, shape (n_S,)
        Stock price grid.
    t_grid : ndarray, shape (n_t,)
        Calendar time grid.
    discovered_coefficients : array-like, length 5
        PDE coefficients from SINDy discovery.
    term_names : list of str or None
        Term names; defaults to TERM_NAMES.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility (for reference; not used in PDE directly).
    T : float
        Maturity.
    option_type : str
        'call' or 'put'.
    n_epochs : int
        Training epochs.
    lr : float
        Learning rate.
    lambda_pde : float
    lambda_bc : float
    lambda_data : float

    Returns
    -------
    dict
        Comprehensive results including:
        - 'test_metrics': dict with relative_l2_error, mae, max_error, r2
        - 'loss_history': dict with per-epoch losses
        - 'model': the trained BSPINN
        - 'sanity_checks': dict of sanity check results
        - 'discovered_coefficients': the PDE coefficients used
        - 'option_type': call or put
        - 'training_params': training hyperparameters
    """
    set_all_seeds(42)

    if term_names is None:
        term_names = TERM_NAMES

    device = get_device()

    # Flatten data
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')
    S_flat = S_mesh.ravel().astype(np.float64)
    t_flat = t_mesh.ravel().astype(np.float64)
    V_flat = V_surface.ravel().astype(np.float64)

    S_min = float(S_grid.min())
    S_max = float(S_grid.max())
    t_min = float(t_grid.min())
    t_max = float(t_grid.max())

    # Output normalization: compute scale and offset from data
    V_offset = float(np.mean(V_flat))
    V_scale = float(np.std(V_flat))
    if V_scale < 1e-10:
        V_scale = 1.0

    logger.info(
        f"PINN setup: domain S=[{S_min:.1f}, {S_max:.1f}], "
        f"t=[{t_min:.3f}, {t_max:.3f}], "
        f"V_scale={V_scale:.4f}, V_offset={V_offset:.4f}"
    )
    logger.info(
        f"Discovered PDE coefficients: "
        + ", ".join(f"{n}={c:.6f}" for n, c in zip(term_names, discovered_coefficients))
    )

    # Create model
    torch.manual_seed(42)
    model = BSPINN(
        S_min=S_min,
        S_max=S_max,
        t_min=t_min,
        t_max=t_max,
        V_scale=V_scale,
        V_offset=V_offset,
    ).to(device)

    # Create trainer
    trainer = PINNTrainer(
        model=model,
        S_data=S_flat,
        t_data=t_flat,
        V_data=V_flat,
        discovered_coefficients=discovered_coefficients,
        term_names=term_names,
        K=K,
        r=r,
        T=T,
        option_type=option_type,
        lambda_pde=lambda_pde,
        lambda_bc=lambda_bc,
        lambda_data=lambda_data,
        n_collocation=10000,
        resample_every=1000,
        n_epochs=n_epochs,
        lr=lr,
        log_every=500,
    )

    # Train
    with Timer("PINN training"):
        history = trainer.train()

    # Evaluate on test set
    test_metrics = trainer.evaluate_test()

    # Sanity checks
    sanity_checks = _run_sanity_checks(
        model, S_min, S_max, t_min, t_max, K, r, T, option_type
    )

    # Generate analytical reference for comparison
    if option_type == 'call':
        V_analytical = bs_call_price(S_mesh, K, r, sigma, T - t_mesh)
    else:
        V_analytical = bs_put_price(S_mesh, K, r, sigma, T - t_mesh)

    # Full-grid evaluation
    model.eval()
    with torch.no_grad():
        S_tensor = torch.tensor(S_flat, dtype=torch.float64, device=device).unsqueeze(-1)
        t_tensor = torch.tensor(t_flat, dtype=torch.float64, device=device).unsqueeze(-1)
        V_pred_full = model(S_tensor, t_tensor).squeeze().numpy()

    V_analytical_flat = V_analytical.ravel()
    full_grid_rel_l2 = (
        np.linalg.norm(V_pred_full - V_analytical_flat)
        / (np.linalg.norm(V_analytical_flat) + 1e-30)
    )
    full_grid_mae = np.mean(np.abs(V_pred_full - V_analytical_flat))

    logger.info(
        f"Full-grid vs analytical: rel_L2={full_grid_rel_l2:.6e}, "
        f"MAE={full_grid_mae:.6e}"
    )

    results = {
        'test_metrics': test_metrics,
        'loss_history': history,
        'model': model,
        'sanity_checks': sanity_checks,
        'discovered_coefficients': list(discovered_coefficients),
        'term_names': term_names,
        'option_type': option_type,
        'training_params': {
            'n_epochs': n_epochs,
            'lr': lr,
            'lambda_pde': lambda_pde,
            'lambda_bc': lambda_bc,
            'lambda_data': lambda_data,
            'n_collocation': 10000,
            'resample_every': 1000,
        },
        'full_grid_metrics': {
            'relative_l2_error': float(full_grid_rel_l2),
            'mae': float(full_grid_mae),
        },
        'V_predicted': V_pred_full.reshape(V_surface.shape),
        'V_analytical': V_analytical,
    }

    return results


# ---------------------------------------------------------------------------
# Log-transform wrapper
# ---------------------------------------------------------------------------

class BSPINNLogTransform(BSPINN):
    """
    BSPINN variant that applies a log transform to the output.

    The network predicts log(V + 1) internally, then applies exp() - 1
    to produce the final option price.  This compresses the dynamic range
    and can improve accuracy for puts whose prices span several orders of
    magnitude.

    Parameters are identical to BSPINN.
    """

    def forward(self, S, t):
        """Forward pass with exp-transform on the raw output."""
        # Normalize to [0, 1]
        S_norm = (S - self.S_min) / (self.S_max - self.S_min + 1e-30)
        t_norm = (t - self.t_min) / (self.t_max - self.t_min + 1e-30)

        x = torch.cat([S_norm, t_norm], dim=-1)
        raw_output = self.net(x)

        # Denormalize raw output
        raw_output = raw_output * self.V_scale + self.V_offset

        # Log-transform: network predicts log(V + 1), so V = exp(raw) - 1
        V_out = torch.exp(raw_output) - 1.0
        return V_out


# ---------------------------------------------------------------------------
# Improved trainer for v2 features
# ---------------------------------------------------------------------------

class PINNTrainerV2(PINNTrainer):
    """
    Extended PINN trainer supporting relative data loss.

    Inherits from PINNTrainer and only overrides ``_data_loss`` when
    ``use_relative_loss=True``.

    Parameters
    ----------
    use_relative_loss : bool
        If True, use relative MSE for the data loss.
    **kwargs
        All other arguments forwarded to PINNTrainer.
    """

    def __init__(self, *, use_relative_loss=False, **kwargs):
        super().__init__(**kwargs)
        self.use_relative_loss = use_relative_loss

    def _data_loss(self):
        """MSE or relative MSE on training data."""
        V_pred = self.model(self.S_train, self.t_train)
        if self.use_relative_loss:
            epsilon = 0.01 * torch.mean(self.V_train ** 2)
            return torch.mean(
                (V_pred - self.V_train) ** 2 / (self.V_train ** 2 + epsilon)
            )
        return torch.mean((V_pred - self.V_train) ** 2)


# ---------------------------------------------------------------------------
# train_pinn_v2
# ---------------------------------------------------------------------------

def train_pinn_v2(
    V_surface,
    S_grid,
    t_grid,
    discovered_coefficients,
    term_names=None,
    K=100.0,
    r=0.05,
    sigma=0.2,
    T=1.0,
    option_type='call',
    n_epochs=10000,
    lr=1e-3,
    lambda_pde=1.0,
    lambda_bc=10.0,
    lambda_data=1.0,
    use_relative_loss=False,
    use_log_transform=False,
):
    """
    Improved PINN training with optional relative loss and log transform.

    This is an enhanced version of :func:`train_pinn` that adds two optional
    features aimed at improving put-option accuracy:

    * **Relative data loss** (``use_relative_loss=True``): replaces the
      standard MSE data loss with a relative variant that normalises by
      the squared true value (plus a small epsilon), helping the network
      achieve uniform relative accuracy.
    * **Log transform** (``use_log_transform=True``): the network
      internally predicts ``log(V + 1)`` and applies ``exp() - 1`` in the
      forward pass, compressing the dynamic range of option prices.

    Parameters
    ----------
    V_surface : ndarray, shape (n_S, n_t)
        Option price surface (may include noise).
    S_grid : ndarray, shape (n_S,)
        Stock price grid.
    t_grid : ndarray, shape (n_t,)
        Calendar time grid.
    discovered_coefficients : array-like, length 5
        PDE coefficients from SINDy discovery.
    term_names : list of str or None
        Term names; defaults to TERM_NAMES.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility (for reference; not used in PDE directly).
    T : float
        Maturity.
    option_type : str
        'call' or 'put'.
    n_epochs : int
        Training epochs.
    lr : float
        Learning rate.
    lambda_pde : float
    lambda_bc : float
    lambda_data : float
    use_relative_loss : bool
        If True, use relative MSE for the data loss.
    use_log_transform : bool
        If True, network predicts log(V+1) and applies exp()-1 in forward.

    Returns
    -------
    dict
        Same structure as :func:`train_pinn` plus:
        - 'used_relative_loss': bool
        - 'used_log_transform': bool
    """
    set_all_seeds(42)

    if term_names is None:
        term_names = TERM_NAMES

    device = get_device()

    # Flatten data
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')
    S_flat = S_mesh.ravel().astype(np.float64)
    t_flat = t_mesh.ravel().astype(np.float64)
    V_flat = V_surface.ravel().astype(np.float64)

    S_min = float(S_grid.min())
    S_max = float(S_grid.max())
    t_min = float(t_grid.min())
    t_max = float(t_grid.max())

    # Output normalization
    if use_log_transform:
        # For log transform, normalise in log-space: log(V + 1)
        V_log = np.log(V_flat + 1.0)
        V_offset = float(np.mean(V_log))
        V_scale = float(np.std(V_log))
        if V_scale < 1e-10:
            V_scale = 1.0
    else:
        V_offset = float(np.mean(V_flat))
        V_scale = float(np.std(V_flat))
        if V_scale < 1e-10:
            V_scale = 1.0

    logger.info(
        f"PINN v2 setup: domain S=[{S_min:.1f}, {S_max:.1f}], "
        f"t=[{t_min:.3f}, {t_max:.3f}], "
        f"V_scale={V_scale:.4f}, V_offset={V_offset:.4f}, "
        f"relative_loss={use_relative_loss}, log_transform={use_log_transform}"
    )
    logger.info(
        f"Discovered PDE coefficients: "
        + ", ".join(f"{n}={c:.6f}" for n, c in zip(term_names, discovered_coefficients))
    )

    # Create model
    torch.manual_seed(42)
    if use_log_transform:
        model = BSPINNLogTransform(
            S_min=S_min,
            S_max=S_max,
            t_min=t_min,
            t_max=t_max,
            V_scale=V_scale,
            V_offset=V_offset,
        ).to(device)
    else:
        model = BSPINN(
            S_min=S_min,
            S_max=S_max,
            t_min=t_min,
            t_max=t_max,
            V_scale=V_scale,
            V_offset=V_offset,
        ).to(device)

    # Create trainer
    trainer = PINNTrainerV2(
        use_relative_loss=use_relative_loss,
        model=model,
        S_data=S_flat,
        t_data=t_flat,
        V_data=V_flat,
        discovered_coefficients=discovered_coefficients,
        term_names=term_names,
        K=K,
        r=r,
        T=T,
        option_type=option_type,
        lambda_pde=lambda_pde,
        lambda_bc=lambda_bc,
        lambda_data=lambda_data,
        n_collocation=10000,
        resample_every=1000,
        n_epochs=n_epochs,
        lr=lr,
        log_every=500,
    )

    # Train
    with Timer("PINN v2 training"):
        history = trainer.train()

    # Evaluate on test set
    test_metrics = trainer.evaluate_test()

    # Sanity checks
    sanity_checks = _run_sanity_checks(
        model, S_min, S_max, t_min, t_max, K, r, T, option_type
    )

    # Generate analytical reference for comparison
    if option_type == 'call':
        V_analytical = bs_call_price(S_mesh, K, r, sigma, T - t_mesh)
    else:
        V_analytical = bs_put_price(S_mesh, K, r, sigma, T - t_mesh)

    # Full-grid evaluation
    model.eval()
    with torch.no_grad():
        S_tensor = torch.tensor(S_flat, dtype=torch.float64, device=device).unsqueeze(-1)
        t_tensor = torch.tensor(t_flat, dtype=torch.float64, device=device).unsqueeze(-1)
        V_pred_full = model(S_tensor, t_tensor).squeeze().numpy()

    V_analytical_flat = V_analytical.ravel()
    full_grid_rel_l2 = (
        np.linalg.norm(V_pred_full - V_analytical_flat)
        / (np.linalg.norm(V_analytical_flat) + 1e-30)
    )
    full_grid_mae = np.mean(np.abs(V_pred_full - V_analytical_flat))

    logger.info(
        f"Full-grid vs analytical: rel_L2={full_grid_rel_l2:.6e}, "
        f"MAE={full_grid_mae:.6e}"
    )

    results = {
        'test_metrics': test_metrics,
        'loss_history': history,
        'model': model,
        'sanity_checks': sanity_checks,
        'discovered_coefficients': list(discovered_coefficients),
        'term_names': term_names,
        'option_type': option_type,
        'training_params': {
            'n_epochs': n_epochs,
            'lr': lr,
            'lambda_pde': lambda_pde,
            'lambda_bc': lambda_bc,
            'lambda_data': lambda_data,
            'n_collocation': 10000,
            'resample_every': 1000,
        },
        'full_grid_metrics': {
            'relative_l2_error': float(full_grid_rel_l2),
            'mae': float(full_grid_mae),
        },
        'V_predicted': V_pred_full.reshape(V_surface.shape),
        'V_analytical': V_analytical,
        'used_relative_loss': use_relative_loss,
        'used_log_transform': use_log_transform,
    }

    return results


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def analyze_pinn_errors(pinn_result, K=100.0):
    """
    Analyze where PINN errors concentrate across moneyness regions.

    Partitions the (S, t) domain into ATM, ITM, and OTM regions based on
    the strike price *K* and computes error metrics for each.

    Parameters
    ----------
    pinn_result : dict
        Result dictionary from :func:`train_pinn` or :func:`train_pinn_v2`.
        Must contain 'V_predicted', 'V_analytical', and 'model' (with
        ``S_min`` / ``S_max`` buffers).
    K : float
        Strike price used to define moneyness regions.

    Returns
    -------
    dict
        Error analysis with keys:
        - 'full_grid': {'rel_l2', 'mae', 'r2'}
        - 'atm_region': {'rel_l2', 'mae', 'n_points'}
        - 'otm_region': {'rel_l2', 'mae', 'n_points'}
        - 'itm_region': {'rel_l2', 'mae', 'n_points'}
        - 'error_grid': ndarray of |V_pred - V_true|
        - 'S_grid': ndarray
        - 't_grid': ndarray
    """
    V_pred = np.asarray(pinn_result['V_predicted'])
    V_true = np.asarray(pinn_result['V_analytical'])
    model = pinn_result['model']
    option_type = pinn_result.get('option_type', 'call')

    S_min = float(model.S_min)
    S_max = float(model.S_max)

    # Reconstruct grids from model and surface shape
    n_S, n_t = V_pred.shape
    S_grid = np.linspace(S_min, S_max, n_S)

    t_min = float(model.t_min)
    t_max = float(model.t_max)
    t_grid = np.linspace(t_min, t_max, n_t)

    # Absolute error grid
    error_grid = np.abs(V_pred - V_true)

    # Full-grid metrics
    diff_flat = (V_pred - V_true).ravel()
    V_true_flat = V_true.ravel()

    rel_l2_full = float(
        np.linalg.norm(diff_flat) / (np.linalg.norm(V_true_flat) + 1e-30)
    )
    mae_full = float(np.mean(np.abs(diff_flat)))

    ss_res = float(np.sum(diff_flat ** 2))
    ss_tot = float(np.sum((V_true_flat - np.mean(V_true_flat)) ** 2))
    r2_full = 1.0 - ss_res / (ss_tot + 1e-30)

    # Build S mesh for region masking (broadcast over t)
    S_mesh = np.tile(S_grid[:, np.newaxis], (1, n_t))  # (n_S, n_t)

    # Region masks
    atm_mask = (S_mesh >= 0.8 * K) & (S_mesh <= 1.2 * K)

    if option_type == 'put':
        otm_mask = S_mesh > K
        itm_mask = S_mesh < K
    else:
        otm_mask = S_mesh < K
        itm_mask = S_mesh > K

    def _region_metrics(mask):
        pred_r = V_pred[mask]
        true_r = V_true[mask]
        n_pts = int(np.sum(mask))
        if n_pts == 0:
            return {'rel_l2': 0.0, 'mae': 0.0, 'n_points': 0}
        diff_r = pred_r - true_r
        rel_l2 = float(
            np.linalg.norm(diff_r) / (np.linalg.norm(true_r) + 1e-30)
        )
        mae = float(np.mean(np.abs(diff_r)))
        return {'rel_l2': rel_l2, 'mae': mae, 'n_points': n_pts}

    analysis = {
        'full_grid': {'rel_l2': rel_l2_full, 'mae': mae_full, 'r2': r2_full},
        'atm_region': _region_metrics(atm_mask),
        'otm_region': _region_metrics(otm_mask),
        'itm_region': _region_metrics(itm_mask),
        'error_grid': error_grid,
        'S_grid': S_grid,
        't_grid': t_grid,
    }

    logger.info(
        f"Error analysis: full rel_L2={rel_l2_full:.6e}, "
        f"ATM rel_L2={analysis['atm_region']['rel_l2']:.6e}, "
        f"ITM rel_L2={analysis['itm_region']['rel_l2']:.6e}, "
        f"OTM rel_L2={analysis['otm_region']['rel_l2']:.6e}"
    )

    return analysis


# ---------------------------------------------------------------------------
# Improvement #16 -- Warmup learning-rate schedule
# ---------------------------------------------------------------------------

def _make_warmup_lr_lambda(warmup_epochs=1000, target_lr=1e-3, initial_lr=1e-4):
    """
    Build a learning-rate multiplier function for ``torch.optim.lr_scheduler.LambdaLR``.

    Linearly ramps the LR multiplier from ``initial_lr / target_lr`` up to ``1.0``
    over ``warmup_epochs`` epochs, then holds constant at ``1.0``.

    Notes
    -----
    The Adam optimizer is created with ``lr=target_lr``. The lambda returned
    here multiplies that base LR.  At epoch 0 the multiplier corresponds to
    ``initial_lr`` and at epoch ``warmup_epochs`` it corresponds to
    ``target_lr``.

    Parameters
    ----------
    warmup_epochs : int
        Number of epochs over which to ramp.
    target_lr : float
        The target (peak) learning rate -- matches the optimizer's base LR.
    initial_lr : float
        The starting learning rate.

    Returns
    -------
    callable
        A function ``lr_lambda(epoch) -> float`` suitable for ``LambdaLR``.
    """
    if target_lr <= 0:
        raise ValueError("target_lr must be positive.")
    start_ratio = float(initial_lr) / float(target_lr)
    warmup_epochs = max(int(warmup_epochs), 1)

    def lr_lambda(epoch):
        if epoch >= warmup_epochs:
            return 1.0
        # Linear interpolation from start_ratio at epoch=0 to 1.0 at warmup_epochs
        return start_ratio + (1.0 - start_ratio) * (epoch / warmup_epochs)

    return lr_lambda


# ---------------------------------------------------------------------------
# Improvement #2 -- Hard-constraint PINN
# ---------------------------------------------------------------------------

class HardConstraintPINN(nn.Module):
    """
    Hard-constraint PINN that bakes the terminal payoff into the architecture.

    The output is constructed as::

        V(S, t) = payoff(S) + (T - t) * raw_network(S, t)

    so that at ``t = T`` the network output is *exactly* equal to the payoff,
    regardless of the network's weights.  This removes the need for a strong
    terminal-condition penalty in the loss.

    Parameters
    ----------
    S_min, S_max : float
        Stock price domain bounds (used for input normalisation).
    t_min, t_max : float
        Time domain bounds (used for input normalisation).
    T : float
        Option maturity time.  The factor ``(T - t)`` vanishes at ``t = T``.
    K : float
        Strike price for the payoff.
    option_type : str
        ``'call'`` or ``'put'``.
    width : int
        Hidden-layer width (default 64).  Tests may pass a smaller value.
    """

    def __init__(self, S_min, S_max, t_min, t_max, T, K,
                 option_type='call', width=64):
        super().__init__()

        self.register_buffer('S_min', torch.tensor(S_min, dtype=torch.float64))
        self.register_buffer('S_max', torch.tensor(S_max, dtype=torch.float64))
        self.register_buffer('t_min', torch.tensor(t_min, dtype=torch.float64))
        self.register_buffer('t_max', torch.tensor(t_max, dtype=torch.float64))
        self.register_buffer('T', torch.tensor(T, dtype=torch.float64))
        self.register_buffer('K', torch.tensor(K, dtype=torch.float64))

        if option_type not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'.")
        self.option_type = option_type

        self.net = nn.Sequential(
            nn.Linear(2, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        ).double()

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def payoff(self, S):
        """Compute the terminal payoff for the given stock prices."""
        if self.option_type == 'call':
            return torch.clamp(S - self.K, min=0.0)
        return torch.clamp(self.K - S, min=0.0)

    def forward(self, S, t):
        S_norm = (S - self.S_min) / (self.S_max - self.S_min + 1e-30)
        t_norm = (t - self.t_min) / (self.t_max - self.t_min + 1e-30)
        x = torch.cat([S_norm, t_norm], dim=-1)
        raw = self.net(x)
        # (T - t) factor ensures payoff is exact at maturity
        tau = self.T - t
        return self.payoff(S) + tau * raw


# ---------------------------------------------------------------------------
# Improvement #8 -- Log-price PINN
# ---------------------------------------------------------------------------

class LogPricePINN(nn.Module):
    """
    PINN that operates in log-price coordinates ``x = log(S)``.

    The Black-Scholes PDE has constant coefficients in log-price space, which
    can improve PINN trainability.  The forward method accepts the *raw*
    stock price ``S`` and time ``t`` and internally computes ``x = log(S)``
    before feeding into the network.

    Parameters
    ----------
    S_min, S_max : float
        Stock price domain bounds.  Used to derive ``x_min = log(S_min)``,
        ``x_max = log(S_max)`` for input normalisation.
    t_min, t_max : float
        Time domain bounds.
    T : float
        Option maturity.
    K : float
        Strike price (used for the optional hard-constraint payoff).
    option_type : str
        ``'call'`` or ``'put'``.
    use_hard_constraint : bool
        If True, output is ``payoff(S) + (T - t) * raw(x, t)``.
    width : int
        Hidden-layer width.
    """

    def __init__(self, S_min, S_max, t_min, t_max, T, K,
                 option_type='call', use_hard_constraint=False, width=64):
        super().__init__()

        if S_min <= 0:
            raise ValueError("S_min must be > 0 for log-price PINN.")

        self.register_buffer('S_min', torch.tensor(S_min, dtype=torch.float64))
        self.register_buffer('S_max', torch.tensor(S_max, dtype=torch.float64))
        self.register_buffer('x_min', torch.tensor(np.log(S_min), dtype=torch.float64))
        self.register_buffer('x_max', torch.tensor(np.log(S_max), dtype=torch.float64))
        self.register_buffer('t_min', torch.tensor(t_min, dtype=torch.float64))
        self.register_buffer('t_max', torch.tensor(t_max, dtype=torch.float64))
        self.register_buffer('T', torch.tensor(T, dtype=torch.float64))
        self.register_buffer('K', torch.tensor(K, dtype=torch.float64))

        if option_type not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'.")
        self.option_type = option_type
        self.use_hard_constraint = bool(use_hard_constraint)

        self.net = nn.Sequential(
            nn.Linear(2, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        ).double()

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def payoff(self, S):
        if self.option_type == 'call':
            return torch.clamp(S - self.K, min=0.0)
        return torch.clamp(self.K - S, min=0.0)

    def forward(self, S, t):
        # x = log(S); normalise to [0, 1]
        x = torch.log(S)
        x_norm = (x - self.x_min) / (self.x_max - self.x_min + 1e-30)
        t_norm = (t - self.t_min) / (self.t_max - self.t_min + 1e-30)
        inp = torch.cat([x_norm, t_norm], dim=-1)
        raw = self.net(inp)
        if self.use_hard_constraint:
            tau = self.T - t
            return self.payoff(S) + tau * raw
        return raw


# ---------------------------------------------------------------------------
# Shared training utilities for the hard-constraint / log-price trainers
# ---------------------------------------------------------------------------

def _bs_pde_residual(model, S, t, r, sigma):
    """
    Compute the Black-Scholes PDE residual for a model that returns V(S, t).

    Uses the *true* BS coefficients (r and sigma) rather than discovered
    coefficients so these trainers can be used standalone.

    Residual: dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V
    """
    V = model(S, t)

    dV_dt = torch.autograd.grad(
        V, t, grad_outputs=torch.ones_like(V),
        create_graph=True, retain_graph=True,
    )[0]
    dV_dS = torch.autograd.grad(
        V, S, grad_outputs=torch.ones_like(V),
        create_graph=True, retain_graph=True,
    )[0]
    d2V_dS2 = torch.autograd.grad(
        dV_dS, S, grad_outputs=torch.ones_like(dV_dS),
        create_graph=True, retain_graph=True,
    )[0]

    residual = dV_dt + 0.5 * sigma ** 2 * S ** 2 * d2V_dS2 + r * S * dV_dS - r * V
    return residual


def _prepare_split(S_flat, t_flat, V_flat, device):
    """60/20/20 split matching the existing PINNTrainer convention."""
    N = len(S_flat)
    idx = np.arange(N)
    idx_train, idx_temp = train_test_split(idx, train_size=0.6, random_state=42)
    idx_val, idx_test = train_test_split(idx_temp, train_size=0.5, random_state=42)

    def _t(arr):
        return torch.tensor(arr, dtype=torch.float64, device=device).unsqueeze(-1)

    return (
        _t(S_flat[idx_train]), _t(t_flat[idx_train]), _t(V_flat[idx_train]),
        _t(S_flat[idx_val]),   _t(t_flat[idx_val]),   _t(V_flat[idx_val]),
        _t(S_flat[idx_test]),  _t(t_flat[idx_test]),  _t(V_flat[idx_test]),
    )


def _generate_grid_and_truth(option_type, K, r, sigma, T,
                             S_min, S_max, n_S=30, n_t=30):
    """Build a (S, t) grid plus the analytical option-price surface."""
    S_grid = np.linspace(S_min, S_max, n_S)
    t_grid = np.linspace(0.0, T - 0.01, n_t)
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')
    tau = T - t_mesh
    if option_type == 'call':
        V = bs_call_price(S_mesh, K, r, sigma, tau)
    else:
        V = bs_put_price(S_mesh, K, r, sigma, tau)
    return S_grid, t_grid, S_mesh, t_mesh, V


def _generic_train_loop(model, S_train, t_train, V_train,
                        S_val, t_val, V_val,
                        S_min, S_max, t_min, t_max,
                        r, sigma, T, K, option_type,
                        n_epochs, lr, lambda_pde, lambda_bc, lambda_data,
                        n_collocation, device, use_warmup,
                        warmup_epochs=1000, warmup_initial_lr=1e-4):
    """
    Shared training loop used by both hard-constraint and log-price trainers.

    Returns
    -------
    (train_loss_history, val_loss_history)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if use_warmup:
        lr_lambda = _make_warmup_lr_lambda(
            warmup_epochs=warmup_epochs,
            target_lr=lr,
            initial_lr=warmup_initial_lr,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_loss_history = []
    val_loss_history = []

    model.train()
    for epoch in range(n_epochs):
        # Sample collocation points (fresh each epoch -- cheap on CPU)
        S_col = (torch.rand(n_collocation, 1, dtype=torch.float64, device=device)
                 * (S_max - S_min) + S_min)
        t_col = (torch.rand(n_collocation, 1, dtype=torch.float64, device=device)
                 * (t_max - t_min) + t_min)
        S_col.requires_grad_(True)
        t_col.requires_grad_(True)

        optimizer.zero_grad()

        # PDE residual
        residual = _bs_pde_residual(model, S_col, t_col, r, sigma)
        loss_pde = torch.mean(residual ** 2)

        # Data loss
        V_pred_data = model(S_train, t_train)
        loss_data = torch.mean((V_pred_data - V_train) ** 2)

        # Boundary loss -- much smaller weight since architecture handles
        # the terminal condition (for hard-constraint) or because we're
        # mainly relying on data + PDE
        if lambda_bc > 0:
            n_bc = 100
            t_bc = (torch.rand(n_bc, 1, dtype=torch.float64, device=device)
                    * (t_max - t_min) + t_min)
            S_lo = torch.full((n_bc, 1), S_min, dtype=torch.float64, device=device)
            S_hi = torch.full((n_bc, 1), S_max, dtype=torch.float64, device=device)
            V_lo = model(S_lo, t_bc)
            V_hi = model(S_hi, t_bc)
            tau_bc = T - t_bc
            if option_type == 'call':
                V_lo_true = torch.zeros_like(V_lo)
                V_hi_true = S_hi - K * torch.exp(-r * tau_bc)
            else:
                V_lo_true = K * torch.exp(-r * tau_bc)
                V_hi_true = torch.zeros_like(V_hi)
            loss_bc = (torch.mean((V_lo - V_lo_true) ** 2)
                       + torch.mean((V_hi - V_hi_true) ** 2))
        else:
            loss_bc = torch.tensor(0.0, dtype=torch.float64, device=device)

        total = (lambda_pde * loss_pde
                 + lambda_data * loss_data
                 + lambda_bc * loss_bc)

        total.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        train_loss_history.append(float(total.item()))

        with torch.no_grad():
            V_pred_val = model(S_val, t_val)
            v_loss = torch.mean((V_pred_val - V_val) ** 2).item()
            val_loss_history.append(float(v_loss))

    return train_loss_history, val_loss_history


def _evaluate_metrics(model, S_test, t_test, V_test):
    model.eval()
    with torch.no_grad():
        V_pred = model(S_test, t_test)
        diff = V_pred - V_test
        l2 = (torch.norm(diff) / (torch.norm(V_test) + 1e-30)).item()
        mae = torch.mean(torch.abs(diff)).item()
        ss_res = torch.sum(diff ** 2)
        ss_tot = torch.sum((V_test - torch.mean(V_test)) ** 2)
        r2 = (1.0 - ss_res / (ss_tot + 1e-30)).item()
    return {'relative_l2_error': float(l2), 'mae': float(mae), 'r2': float(r2)}


def _boundary_error_at_maturity(model, K, S_min, S_max, T, option_type, n=200):
    """Average |V(S, T) - payoff(S)| over a grid."""
    model.eval()
    S = torch.linspace(S_min, S_max, n, dtype=torch.float64).unsqueeze(-1)
    t = torch.full((n, 1), T, dtype=torch.float64)
    with torch.no_grad():
        V_pred = model(S, t).squeeze().numpy()
    S_np = S.squeeze().numpy()
    if option_type == 'call':
        payoff = np.maximum(S_np - K, 0.0)
    else:
        payoff = np.maximum(K - S_np, 0.0)
    return float(np.mean(np.abs(V_pred - payoff)))


# ---------------------------------------------------------------------------
# train_hard_constraint_pinn (Improvement #2)
# ---------------------------------------------------------------------------

def train_hard_constraint_pinn(
    option_type='call',
    K=100.0,
    r=0.05,
    sigma=0.2,
    T=1.0,
    S_min=10.0,
    S_max=200.0,
    n_epochs=5000,
    lr=1e-3,
    seed=42,
    device='cpu',
    use_warmup=False,
    n_S=30,
    n_t=30,
    width=64,
    n_collocation=2000,
    lambda_pde=1.0,
    lambda_bc=0.1,
    lambda_data=1.0,
    warmup_epochs=1000,
    warmup_initial_lr=1e-4,
):
    """
    Train a HardConstraintPINN on the Black-Scholes PDE.

    Because the terminal payoff is enforced exactly by the architecture, the
    boundary-loss weight is reduced (default ``lambda_bc=0.1``).

    Returns
    -------
    dict with keys:
        - model
        - train_loss_history
        - val_loss_history
        - test_metrics: {relative_l2_error, mae, r2}
        - boundary_error
    """
    try:
        set_all_seeds(seed)
        torch.manual_seed(seed)
        dev = torch.device(device)

        S_grid, t_grid, S_mesh, t_mesh, V = _generate_grid_and_truth(
            option_type, K, r, sigma, T, S_min, S_max, n_S=n_S, n_t=n_t,
        )
        t_min = float(t_grid.min())
        t_max = float(t_grid.max())

        S_flat = S_mesh.ravel().astype(np.float64)
        t_flat = t_mesh.ravel().astype(np.float64)
        V_flat = V.ravel().astype(np.float64)

        (S_tr, t_tr, V_tr,
         S_va, t_va, V_va,
         S_te, t_te, V_te) = _prepare_split(S_flat, t_flat, V_flat, dev)

        model = HardConstraintPINN(
            S_min=S_min, S_max=S_max, t_min=t_min, t_max=t_max,
            T=T, K=K, option_type=option_type, width=width,
        ).to(dev)

        train_hist, val_hist = _generic_train_loop(
            model,
            S_tr, t_tr, V_tr, S_va, t_va, V_va,
            S_min, S_max, t_min, t_max,
            r, sigma, T, K, option_type,
            n_epochs, lr, lambda_pde, lambda_bc, lambda_data,
            n_collocation, dev, use_warmup,
            warmup_epochs=warmup_epochs,
            warmup_initial_lr=warmup_initial_lr,
        )

        test_metrics = _evaluate_metrics(model, S_te, t_te, V_te)
        boundary_error = _boundary_error_at_maturity(
            model, K, S_min, S_max, T, option_type,
        )

        return {
            'model': model,
            'train_loss_history': train_hist,
            'val_loss_history': val_hist,
            'test_metrics': test_metrics,
            'boundary_error': boundary_error,
            'option_type': option_type,
            'used_warmup': bool(use_warmup),
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("train_hard_constraint_pinn failed: %s", exc)
        return {
            'model': None,
            'train_loss_history': [],
            'val_loss_history': [],
            'test_metrics': {'relative_l2_error': float('nan'),
                             'mae': float('nan'), 'r2': float('nan')},
            'boundary_error': float('nan'),
            'error': str(exc),
        }


# ---------------------------------------------------------------------------
# train_log_price_pinn (Improvement #8)
# ---------------------------------------------------------------------------

def train_log_price_pinn(
    option_type='call',
    K=100.0,
    r=0.05,
    sigma=0.2,
    T=1.0,
    S_min=10.0,
    S_max=200.0,
    n_epochs=5000,
    lr=1e-3,
    seed=42,
    device='cpu',
    use_hard_constraint=False,
    use_warmup=False,
    n_S=30,
    n_t=30,
    width=64,
    n_collocation=2000,
    lambda_pde=1.0,
    lambda_bc=0.1,
    lambda_data=1.0,
    warmup_epochs=1000,
    warmup_initial_lr=1e-4,
):
    """
    Train a LogPricePINN.  If ``use_hard_constraint=True`` the architecture
    also bakes in the terminal payoff.

    Returns the same dict shape as :func:`train_hard_constraint_pinn`.
    """
    try:
        set_all_seeds(seed)
        torch.manual_seed(seed)
        dev = torch.device(device)

        S_grid, t_grid, S_mesh, t_mesh, V = _generate_grid_and_truth(
            option_type, K, r, sigma, T, S_min, S_max, n_S=n_S, n_t=n_t,
        )
        t_min = float(t_grid.min())
        t_max = float(t_grid.max())

        S_flat = S_mesh.ravel().astype(np.float64)
        t_flat = t_mesh.ravel().astype(np.float64)
        V_flat = V.ravel().astype(np.float64)

        (S_tr, t_tr, V_tr,
         S_va, t_va, V_va,
         S_te, t_te, V_te) = _prepare_split(S_flat, t_flat, V_flat, dev)

        model = LogPricePINN(
            S_min=S_min, S_max=S_max, t_min=t_min, t_max=t_max,
            T=T, K=K, option_type=option_type,
            use_hard_constraint=use_hard_constraint,
            width=width,
        ).to(dev)

        # If using hard constraint we can de-emphasise the bc loss
        effective_lambda_bc = lambda_bc if not use_hard_constraint else min(lambda_bc, 0.1)

        train_hist, val_hist = _generic_train_loop(
            model,
            S_tr, t_tr, V_tr, S_va, t_va, V_va,
            S_min, S_max, t_min, t_max,
            r, sigma, T, K, option_type,
            n_epochs, lr, lambda_pde, effective_lambda_bc, lambda_data,
            n_collocation, dev, use_warmup,
            warmup_epochs=warmup_epochs,
            warmup_initial_lr=warmup_initial_lr,
        )

        test_metrics = _evaluate_metrics(model, S_te, t_te, V_te)
        boundary_error = _boundary_error_at_maturity(
            model, K, S_min, S_max, T, option_type,
        )

        return {
            'model': model,
            'train_loss_history': train_hist,
            'val_loss_history': val_hist,
            'test_metrics': test_metrics,
            'boundary_error': boundary_error,
            'option_type': option_type,
            'used_warmup': bool(use_warmup),
            'used_hard_constraint': bool(use_hard_constraint),
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("train_log_price_pinn failed: %s", exc)
        return {
            'model': None,
            'train_loss_history': [],
            'val_loss_history': [],
            'test_metrics': {'relative_l2_error': float('nan'),
                             'mae': float('nan'), 'r2': float('nan')},
            'boundary_error': float('nan'),
            'error': str(exc),
        }
