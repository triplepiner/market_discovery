"""
Shared utilities for the BS PDE Discovery project.

Provides seed management, logging, timing, numerical differentiation,
and error computation utilities used across all modules.
"""

import os
import json
import random
import time
import logging
import numpy as np
import torch
from scipy.signal import savgol_filter


# Global timing registry – populated automatically by Timer.__exit__
_TIMING_REGISTRY = {}


def set_all_seeds(seed=42):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Return CPU device for reproducibility."""
    return torch.device('cpu')


def setup_logging(name, level=logging.INFO):
    """
    Return a configured logger with console and file handlers.

    Parameters
    ----------
    name : str
        Logger name (typically module name).
    level : int
        Logging level.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'pipeline.log'), mode='a')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class Timer:
    """Context manager that measures and prints wall-clock time.

    Usage:
        with Timer("SINDy discovery"):
            ...
    """

    def __init__(self, description="Block", silent=False):
        self.description = description
        self.elapsed = 0.0
        self.silent = silent

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        _TIMING_REGISTRY[self.description] = self.elapsed
        if not self.silent:
            print(f"  [{self.description}] elapsed: {self.elapsed:.2f}s")


def get_all_timings() -> dict:
    """Return a copy of the global timing registry."""
    return dict(_TIMING_REGISTRY)


def save_timings(filepath: str) -> None:
    """Save the global timing registry to a JSON file at *filepath*."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(_TIMING_REGISTRY, f, indent=2)


def reset_timings() -> None:
    """Clear the global timing registry."""
    _TIMING_REGISTRY.clear()


def safe_relative_error(discovered, true, eps=1e-10):
    """
    Compute |discovered - true| / max(|true|, eps).

    Avoids division by zero when true coefficients are near zero.

    Parameters
    ----------
    discovered : float or ndarray
    true : float or ndarray
    eps : float
        Floor for the denominator.

    Returns
    -------
    float or ndarray
    """
    discovered = np.asarray(discovered, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return np.abs(discovered - true) / np.maximum(np.abs(true), eps)


class NumericalDifferentiator:
    """
    Numerical differentiation with optional Savitzky-Golay pre-smoothing.

    Supports 2nd-order and 4th-order central differences. Falls back to
    forward/backward differences at boundaries automatically via numpy's
    gradient function.

    Parameters
    ----------
    order : int
        Finite difference order: 2 (default) or 4.
    smooth : bool
        Whether to apply Savitzky-Golay smoothing before differentiation.
    savgol_window : int
        Window length for Savitzky-Golay filter.
    savgol_poly : int
        Polynomial order for Savitzky-Golay filter.
    """

    def __init__(self, order=2, smooth=False, savgol_window=7, savgol_poly=3):
        if order not in (2, 4):
            raise ValueError("order must be 2 or 4")
        self.order = order
        self.smooth = smooth
        self.savgol_window = savgol_window
        self.savgol_poly = savgol_poly

    def _maybe_smooth(self, f, axis):
        """Apply Savitzky-Golay smoothing along the given axis if enabled."""
        if not self.smooth:
            return f
        win = self.savgol_window
        # Ensure window length is odd and <= array size along axis
        n = f.shape[axis]
        if win > n:
            win = n if n % 2 == 1 else n - 1
        if win < self.savgol_poly + 2:
            return f  # array too small for smoothing
        return savgol_filter(f, window_length=win, polyorder=self.savgol_poly, axis=axis)

    def first_derivative(self, f, dx, axis=0):
        """
        Compute first derivative df/dx along the specified axis.

        Parameters
        ----------
        f : ndarray
            Function values on a uniform grid.
        dx : float
            Grid spacing.
        axis : int
            Axis along which to differentiate.

        Returns
        -------
        ndarray
            Same shape as f.
        """
        f_s = self._maybe_smooth(f, axis)
        if self.order == 2:
            return np.gradient(f_s, dx, axis=axis)
        else:
            return self._fourth_order_first(f_s, dx, axis)

    def second_derivative(self, f, dx, axis=0):
        """
        Compute second derivative d2f/dx2 along the specified axis.

        Uses a direct second-derivative stencil for accuracy:
        f''(x) = (f(x+h) - 2f(x) + f(x-h)) / h^2

        Parameters
        ----------
        f : ndarray
        dx : float
        axis : int

        Returns
        -------
        ndarray
            Same shape as f.
        """
        f_s = self._maybe_smooth(f, axis)
        if self.order == 2:
            return self._second_deriv_direct(f_s, dx, axis)
        else:
            return self._fourth_order_second(f_s, dx, axis)

    def _second_deriv_direct(self, f, dx, axis):
        """Direct second-derivative stencil: (f[i+1] - 2f[i] + f[i-1]) / h^2."""
        n = f.shape[axis]
        result = np.zeros_like(f)

        # Build slices for central difference
        slc_m = [slice(None)] * f.ndim
        slc_c = [slice(None)] * f.ndim
        slc_p = [slice(None)] * f.ndim

        slc_m[axis] = slice(0, n - 2)
        slc_c[axis] = slice(1, n - 1)
        slc_p[axis] = slice(2, n)

        interior = [slice(None)] * f.ndim
        interior[axis] = slice(1, n - 1)

        result[tuple(interior)] = (f[tuple(slc_p)] - 2 * f[tuple(slc_c)] + f[tuple(slc_m)]) / (dx ** 2)

        # Forward difference at left boundary: f''(0) ≈ (f[2] - 2f[1] + f[0]) / h^2
        slc_b = [slice(None)] * f.ndim
        slc_b[axis] = 0
        slc_b1 = [slice(None)] * f.ndim
        slc_b1[axis] = 1
        slc_b2 = [slice(None)] * f.ndim
        slc_b2[axis] = 2
        result[tuple(slc_b)] = (f[tuple(slc_b2)] - 2 * f[tuple(slc_b1)] + f[tuple(slc_b)]) / (dx ** 2)

        # Backward difference at right boundary
        slc_e = [slice(None)] * f.ndim
        slc_e[axis] = -1
        slc_e1 = [slice(None)] * f.ndim
        slc_e1[axis] = -2
        slc_e2 = [slice(None)] * f.ndim
        slc_e2[axis] = -3
        result[tuple(slc_e)] = (f[tuple(slc_e)] - 2 * f[tuple(slc_e1)] + f[tuple(slc_e2)]) / (dx ** 2)

        return result

    def _fourth_order_first(self, f, dx, axis):
        """4th-order central difference for first derivative, with fallback at edges."""
        n = f.shape[axis]
        result = np.zeros_like(f)

        if n < 5:
            return np.gradient(f, dx, axis=axis)

        def _sl(axis, idx):
            s = [slice(None)] * f.ndim
            s[axis] = idx
            return tuple(s)

        # Interior: (-f[i+2] + 8f[i+1] - 8f[i-1] + f[i-2]) / (12h)
        for i in range(2, n - 2):
            result[_sl(axis, i)] = (
                -f[_sl(axis, i + 2)] + 8 * f[_sl(axis, i + 1)]
                - 8 * f[_sl(axis, i - 1)] + f[_sl(axis, i - 2)]
            ) / (12 * dx)

        # Boundaries: use numpy gradient (2nd order)
        grad_fallback = np.gradient(f, dx, axis=axis)
        for i in [0, 1, n - 2, n - 1]:
            result[_sl(axis, i)] = grad_fallback[_sl(axis, i)]

        return result

    def _fourth_order_second(self, f, dx, axis):
        """4th-order central difference for second derivative, with fallback at edges."""
        n = f.shape[axis]
        result = np.zeros_like(f)

        if n < 5:
            return self._second_deriv_direct(f, dx, axis)

        def _sl(axis, idx):
            s = [slice(None)] * f.ndim
            s[axis] = idx
            return tuple(s)

        # Interior: (-f[i+2] + 16f[i+1] - 30f[i] + 16f[i-1] - f[i-2]) / (12h^2)
        for i in range(2, n - 2):
            result[_sl(axis, i)] = (
                -f[_sl(axis, i + 2)] + 16 * f[_sl(axis, i + 1)]
                - 30 * f[_sl(axis, i)]
                + 16 * f[_sl(axis, i - 1)] - f[_sl(axis, i - 2)]
            ) / (12 * dx ** 2)

        # Boundaries: fallback to direct stencil
        fallback = self._second_deriv_direct(f, dx, axis)
        for i in [0, 1, n - 2, n - 1]:
            result[_sl(axis, i)] = fallback[_sl(axis, i)]

        return result
