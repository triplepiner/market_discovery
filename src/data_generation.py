"""
Analytical Black-Scholes price surfaces and noise generation.

Generates synthetic option price data from closed-form Black-Scholes formulas
to serve as ground truth for PDE discovery experiments.
"""

import numpy as np
from scipy.stats import norm
from scipy.special import factorial
from src.utils import set_all_seeds, setup_logging

logger = setup_logging(__name__)


def _validate_inputs(S=None, sigma=None, tau=None):
    """Validate common BS inputs."""
    if S is not None:
        S = np.asarray(S, dtype=np.float64)
        if np.any(S <= 0):
            raise ValueError("Stock price S must be positive.")
    if sigma is not None and np.any(np.asarray(sigma) <= 0):
        raise ValueError("Volatility sigma must be positive.")
    if tau is not None:
        tau = np.asarray(tau, dtype=np.float64)
        if np.any(tau < 0):
            raise ValueError(
                "Time-to-maturity tau must be non-negative. "
                "tau = T - t; ensure t <= T."
            )


def bs_d1(S, K, r, sigma, tau):
    """
    Compute Black-Scholes d1.

    d1 = [ln(S/K) + (r + 0.5*sigma^2)*tau] / (sigma * sqrt(tau))

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

    Returns
    -------
    float or ndarray
    """
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    _validate_inputs(S=S, sigma=sigma, tau=tau)

    result = np.zeros_like(S * tau, dtype=np.float64)
    mask = tau > 0
    if np.any(mask):
        sqrt_tau = np.sqrt(np.where(mask, tau, 1.0))
        result = np.where(
            mask,
            (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_tau),
            0.0
        )
    # At tau=0: limits
    at_maturity = ~mask
    if np.any(at_maturity):
        result = np.where(
            at_maturity & (S > K), np.inf,
            np.where(at_maturity & (S < K), -np.inf,
                     np.where(at_maturity & np.isclose(S, K), 0.0, result))
        )
    return result


def bs_d2(S, K, r, sigma, tau):
    """
    Compute Black-Scholes d2 = d1 - sigma * sqrt(tau).

    Parameters match bs_d1.
    """
    tau = np.asarray(tau, dtype=np.float64)
    _validate_inputs(sigma=sigma, tau=tau)
    d1 = bs_d1(S, K, r, sigma, tau)
    return d1 - sigma * np.sqrt(np.maximum(tau, 0.0))


def bs_call_price(S, K, r, sigma, tau):
    """
    European call price: C = S*N(d1) - K*exp(-r*tau)*N(d2).

    At tau=0 returns intrinsic value max(S-K, 0).

    Parameters
    ----------
    S : float or ndarray
    K : float
    r : float
    sigma : float
    tau : float or ndarray

    Returns
    -------
    ndarray
    """
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    _validate_inputs(S=S, sigma=sigma, tau=tau)

    result = np.maximum(S - K, 0.0)  # default for tau=0
    mask = tau > 0
    if np.any(mask):
        d1 = bs_d1(S, K, r, sigma, tau)
        d2 = bs_d2(S, K, r, sigma, tau)
        c = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        result = np.where(mask, c, result)
    return result


def bs_put_price(S, K, r, sigma, tau):
    """
    European put price: P = K*exp(-r*tau)*N(-d2) - S*N(-d1).

    Also verified via put-call parity: P = C - S + K*exp(-r*tau).

    Parameters match bs_call_price.
    """
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    _validate_inputs(S=S, sigma=sigma, tau=tau)

    result = np.maximum(K - S, 0.0)
    mask = tau > 0
    if np.any(mask):
        d1 = bs_d1(S, K, r, sigma, tau)
        d2 = bs_d2(S, K, r, sigma, tau)
        p_direct = K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)
        # Verify via put-call parity
        c = bs_call_price(S, K, r, sigma, tau)
        p_parity = c - S + K * np.exp(-r * tau)
        diff = np.abs(np.where(mask, p_direct - p_parity, 0.0))
        if np.max(diff) > 1e-10:
            logger.warning(
                f"Put-call parity violation: max diff = {np.max(diff):.2e}"
            )
        result = np.where(mask, p_direct, result)
    return result


def bs_call_delta(S, K, r, sigma, tau):
    """Analytical call Delta = N(d1)."""
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    _validate_inputs(S=S, sigma=sigma, tau=tau)

    mask = tau > 0
    d1 = bs_d1(S, K, r, sigma, tau)
    delta = np.where(mask, norm.cdf(d1), 0.0)
    at_mat = ~mask
    delta = np.where(at_mat & (S > K), 1.0, delta)
    delta = np.where(at_mat & (S < K), 0.0, delta)
    delta = np.where(at_mat & np.isclose(S, K), 0.5, delta)
    return delta


def bs_put_delta(S, K, r, sigma, tau):
    """Analytical put Delta = N(d1) - 1."""
    return bs_call_delta(S, K, r, sigma, tau) - 1.0


def bs_gamma(S, K, r, sigma, tau):
    """
    Analytical Gamma = N'(d1) / (S * sigma * sqrt(tau)).

    Same for calls and puts. Returns 0 at tau=0.
    """
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    _validate_inputs(S=S, sigma=sigma, tau=tau)

    mask = tau > 0
    d1 = bs_d1(S, K, r, sigma, tau)
    sqrt_tau = np.sqrt(np.where(mask, tau, 1.0))
    gamma = np.where(
        mask,
        norm.pdf(d1) / (S * sigma * sqrt_tau),
        0.0
    )
    return gamma


def bs_theta_call(S, K, r, sigma, tau):
    """
    Analytical Theta for call option.

    Theta = -S*N'(d1)*sigma / (2*sqrt(tau)) - r*K*exp(-r*tau)*N(d2)

    This is dV/dt (calendar time), which equals -dV/dtau.
    """
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    _validate_inputs(S=S, sigma=sigma, tau=tau)

    mask = tau > 0
    d1 = bs_d1(S, K, r, sigma, tau)
    d2 = bs_d2(S, K, r, sigma, tau)
    sqrt_tau = np.sqrt(np.where(mask, tau, 1.0))

    theta = np.where(
        mask,
        -S * norm.pdf(d1) * sigma / (2 * sqrt_tau) - r * K * np.exp(-r * tau) * norm.cdf(d2),
        0.0
    )
    return theta


def bs_theta_put(S, K, r, sigma, tau):
    """
    Analytical Theta for put option.

    Theta_put = -S*N'(d1)*sigma / (2*sqrt(tau)) + r*K*exp(-r*tau)*N(-d2)
    """
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    _validate_inputs(S=S, sigma=sigma, tau=tau)

    mask = tau > 0
    d1 = bs_d1(S, K, r, sigma, tau)
    d2 = bs_d2(S, K, r, sigma, tau)
    sqrt_tau = np.sqrt(np.where(mask, tau, 1.0))

    theta = np.where(
        mask,
        -S * norm.pdf(d1) * sigma / (2 * sqrt_tau) + r * K * np.exp(-r * tau) * norm.cdf(-d2),
        0.0
    )
    return theta


def generate_price_surface(S_min=50, S_max=150, n_S=100, t_min=0.0, t_max=None,
                           n_t=100, K=100, r=0.05, sigma=0.2, T=1.0,
                           option_type='call'):
    """
    Generate a 2D option price surface on a uniform grid.

    Parameters
    ----------
    S_min, S_max : float
        Stock price range.
    n_S : int
        Number of stock price grid points.
    t_min : float
        Start of calendar time grid.
    t_max : float or None
        End of calendar time grid. Defaults to T - 0.01 to avoid the
        maturity singularity where the payoff kink makes derivatives blow up.
    n_t : int
        Number of time grid points.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Option maturity.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    V : ndarray, shape (n_S, n_t)
        Option prices.
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    """
    _validate_inputs(sigma=sigma)
    if r < 0:
        logger.warning("Negative risk-free rate r=%f is unusual.", r)

    if t_max is None:
        t_max = T - 0.01  # Avoid maturity singularity

    S_grid = np.linspace(S_min, S_max, n_S)
    t_grid = np.linspace(t_min, t_max, n_t)

    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')
    tau_mesh = T - t_mesh  # time to maturity

    if option_type == 'call':
        V = bs_call_price(S_mesh, K, r, sigma, tau_mesh)
    elif option_type == 'put':
        V = bs_put_price(S_mesh, K, r, sigma, tau_mesh)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    return V, S_grid, t_grid


def add_noise(V, noise_pct, seed=42):
    """
    Add Gaussian noise to a price surface.

    noise_std = noise_pct * std(V). Clips result to >= 0 since option
    prices cannot be negative.

    Parameters
    ----------
    V : ndarray
        Clean price surface.
    noise_pct : float
        Noise level as fraction of the surface's standard deviation.
    seed : int
        Random seed.

    Returns
    -------
    V_noisy : ndarray
    """
    set_all_seeds(seed)
    if noise_pct == 0:
        return V.copy()

    noise_std = noise_pct * np.std(V)
    noise = noise_std * np.random.randn(*V.shape)
    V_noisy = V + noise

    n_clipped = np.sum(V_noisy < 0)
    if n_clipped > 0:
        frac_clipped = n_clipped / V_noisy.size
        logger.warning(
            f"Clipped {n_clipped} negative prices ({frac_clipped:.1%} of grid). "
            f"Noise level {noise_pct} may be too high."
        )
        if frac_clipped > 0.05:
            logger.warning("More than 5% of points clipped — noise is very high.")
        V_noisy = np.clip(V_noisy, 0.0, None)

    return V_noisy


def generate_multi_param_surfaces(sigma_list, r_list, S_min=50, S_max=150,
                                  n_S=100, n_t=100, K=100, T=1.0,
                                  option_type='call'):
    """
    Generate price surfaces for all combinations of sigma and r.

    Returns
    -------
    dict
        Keyed by (sigma, r) tuples, values are (V, S_grid, t_grid).
    """
    surfaces = {}
    for sigma in sigma_list:
        for r in r_list:
            V, S_grid, t_grid = generate_price_surface(
                S_min=S_min, S_max=S_max, n_S=n_S, n_t=n_t,
                K=K, r=r, sigma=sigma, T=T, option_type=option_type
            )
            surfaces[(sigma, r)] = (V, S_grid, t_grid)
    return surfaces


def merton_call_price(S, K, r, sigma, tau, lam=0.1, mu_J=-0.05, sigma_J=0.1, N_max=50):
    """
    Merton jump-diffusion call price via truncated series expansion.

    V = sum_{n=0}^{N_max} [exp(-lam'*tau) * (lam'*tau)^n / n!] * BS(S, K, r_n, sigma_n, tau)

    where:
        lam' = lam * (1 + mu_J)
        r_n = r - lam*mu_J + n*log(1+mu_J)/tau
        sigma_n = sqrt(sigma^2 + n*sigma_J^2/tau)

    Parameters
    ----------
    S : float or ndarray
        Stock price(s).
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Diffusion volatility.
    tau : float or ndarray
        Time to maturity (T - t).
    lam : float
        Jump intensity (average number of jumps per year).
    mu_J : float
        Mean of log-jump size.
    sigma_J : float
        Std deviation of log-jump size.
    N_max : int
        Number of terms in the series expansion.

    Returns
    -------
    ndarray
        Merton jump-diffusion call prices.
    """
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)

    # Handle tau near zero: return intrinsic value
    near_zero = tau < 1e-12
    result = np.maximum(S - K, 0.0)

    if np.all(near_zero):
        return result

    lam_prime = lam * (1.0 + mu_J)

    price = np.zeros_like(S * tau, dtype=np.float64)

    for n in range(N_max + 1):
        # Poisson weight: exp(-lam'*tau) * (lam'*tau)^n / n!
        log_weight = -lam_prime * tau + n * np.log(np.maximum(lam_prime * tau, 1e-300))
        log_weight = log_weight - np.log(float(factorial(n, exact=True)))
        weight = np.exp(log_weight)

        # Adjusted parameters for term n
        # Guard against tau=0 in the division by using safe_tau
        safe_tau = np.where(near_zero, 1.0, tau)
        r_n = r - lam * mu_J + n * np.log(1.0 + mu_J) / safe_tau
        sigma_n = np.sqrt(sigma ** 2 + n * sigma_J ** 2 / safe_tau)

        bs_price = bs_call_price(S, K, r_n, sigma_n, tau)
        price = price + weight * bs_price

    result = np.where(near_zero, result, price)
    return result


def generate_merton_surface(S_min=50, S_max=150, n_S=100, t_min=0.0, n_t=100,
                            K=100, r=0.05, sigma=0.2, T=1.0,
                            lam=0.1, mu_J=-0.05, sigma_J=0.1):
    """
    Generate Merton jump-diffusion call price surface.

    Creates the same grid as generate_price_surface but uses merton_call_price
    instead of the standard Black-Scholes formula.

    Parameters
    ----------
    S_min, S_max : float
        Stock price range.
    n_S : int
        Number of stock price grid points.
    t_min : float
        Start of calendar time grid.
    n_t : int
        Number of time grid points.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Diffusion volatility.
    T : float
        Option maturity.
    lam : float
        Jump intensity.
    mu_J : float
        Mean of log-jump size.
    sigma_J : float
        Std deviation of log-jump size.

    Returns
    -------
    V_merton : ndarray, shape (n_S, n_t)
        Merton jump-diffusion call prices.
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    """
    _validate_inputs(sigma=sigma)

    t_max = T - 0.01  # Avoid maturity singularity

    S_grid = np.linspace(S_min, S_max, n_S)
    t_grid = np.linspace(t_min, t_max, n_t)

    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')
    tau_mesh = T - t_mesh  # time to maturity

    V_merton = merton_call_price(S_mesh, K, r, sigma, tau_mesh,
                                 lam=lam, mu_J=mu_J, sigma_J=sigma_J)

    return V_merton, S_grid, t_grid
