"""
Misspecification diagnostic taxonomy.

Systematic study of what SINDy with a Black-Scholes library discovers when
the underlying data violates BS assumptions in different ways:

    (A) Jump intensity sweep — Merton jump-diffusion, varying lambda.
    (B) Jump size sweep      — Merton jump-diffusion, varying mu_J.
    (C) Stochastic volatility sweep — Heston, varying vol-of-vol sigma_v at
        a fixed initial variance v0 (matched to BS sigma = sqrt(v0)).

The library is the standard 5-term BS library:
    [V, dV/dS, d2V/dS2, S*dV/dS, S^2*d2V/dS2]

For each parameter setting we record all five discovered coefficients plus
the noise-free R^2.  This expands the single-anecdote Merton finding into a
taxonomy that can (or cannot) distinguish jump misspecification from
stochastic-volatility misspecification.

Heston prices are computed via the Lewis (2001) integral over the
characteristic function — a CPU-only, scipy.quad-based implementation.
At sigma_v = 0 the Heston dynamics reduce to constant-volatility BS, which
the implementation handles as a degenerate special case (we just call the
BS pricer).
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.integrate import quad

from src.utils import set_all_seeds, setup_logging, Timer
from src.sindy_discovery import discover_pde, TERM_NAMES
from src.data_generation import generate_merton_surface, bs_call_price

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Heston (1993) semi-analytic call pricer
# ---------------------------------------------------------------------------

def _heston_char_func(phi, S, K, r, tau, v0, kappa, theta, sigma_v, rho, j):
    """
    Heston characteristic function (Heston 1993, formulation as in Albrecher
    et al. 2007 "The little Heston trap" — branch-cut-stable form).

    j = 1 or 2 selects the two probability measures P1 / P2.
    """
    x = np.log(S)
    a = kappa * theta
    if j == 1:
        u = 0.5
        b = kappa - rho * sigma_v
    else:
        u = -0.5
        b = kappa

    d = np.sqrt((rho * sigma_v * 1j * phi - b) ** 2
                - sigma_v ** 2 * (2 * u * 1j * phi - phi ** 2))
    g2 = (b - rho * sigma_v * 1j * phi - d) / (b - rho * sigma_v * 1j * phi + d)

    exp_dt = np.exp(-d * tau)
    C = (r * 1j * phi * tau
         + (a / sigma_v ** 2)
         * ((b - rho * sigma_v * 1j * phi - d) * tau
            - 2.0 * np.log((1.0 - g2 * exp_dt) / (1.0 - g2))))
    D = ((b - rho * sigma_v * 1j * phi - d) / sigma_v ** 2) \
        * ((1.0 - exp_dt) / (1.0 - g2 * exp_dt))

    return np.exp(C + D * v0 + 1j * phi * x)


def _heston_P(S, K, r, tau, v0, kappa, theta, sigma_v, rho, j, upper=200.0):
    """Probability P_j via the standard Heston integral (real part of e^{-i*phi*ln K} cf / (i phi))."""
    def integrand(phi):
        cf = _heston_char_func(phi, S, K, r, tau, v0, kappa, theta, sigma_v, rho, j)
        val = np.exp(-1j * phi * np.log(K)) * cf / (1j * phi)
        return val.real

    # Tight tolerances + finite upper bound; the integrand decays fast for
    # reasonable parameters.
    val, _ = quad(integrand, 1e-8, upper, limit=200, epsabs=1e-8, epsrel=1e-6)
    return 0.5 + val / np.pi


def heston_call_price_scalar(S, K, r, tau, v0, kappa, theta, sigma_v, rho):
    """
    Heston (1993) European call price for scalar inputs.

    Parameters
    ----------
    S, K, r, tau : float
    v0 : float
        Initial variance.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-run variance.
    sigma_v : float
        Vol-of-vol (the parameter we sweep).
    rho : float
        Correlation between asset and variance Brownian motions.

    Returns
    -------
    float
    """
    if tau < 1e-12:
        return max(S - K, 0.0)

    # Degenerate case: sigma_v == 0 reduces to constant-variance BS with
    # variance = mean reverting around theta from v0.  When kappa is small or
    # we take theta == v0 the variance stays constant and the price equals
    # BS at sigma = sqrt(v0). We use that closed form directly.
    if sigma_v < 1e-10:
        if abs(theta - v0) < 1e-12 or kappa < 1e-10:
            sigma_bs = np.sqrt(max(v0, 1e-12))
            return float(bs_call_price(np.array([S]), K, r, sigma_bs, tau)[0])
        # Otherwise variance is deterministic ODE: v(s) = theta + (v0-theta)*e^{-kappa*s}.
        # Mean-variance over [0, tau] gives effective sigma^2; fall through
        # to numerical integration via the characteristic function with a
        # small floor on sigma_v.
        sigma_v = 1e-8

    P1 = _heston_P(S, K, r, tau, v0, kappa, theta, sigma_v, rho, j=1)
    P2 = _heston_P(S, K, r, tau, v0, kappa, theta, sigma_v, rho, j=2)
    return float(S * P1 - K * np.exp(-r * tau) * P2)


def generate_heston_surface(S_min=50, S_max=150, n_S=100, t_min=0.0, n_t=100,
                            K=100, r=0.05, T=1.0,
                            v0=0.04, kappa=2.0, theta=0.04,
                            sigma_v=0.3, rho=-0.5):
    """
    Generate a Heston call price surface.

    Defaults give a fixed-variance slice at v0 = theta = 0.04 (matching
    BS sigma = 0.20).  Only sigma_v is varied across the taxonomy sweep.

    Returns
    -------
    V : ndarray, shape (n_S, n_t)
    S_grid : ndarray, shape (n_S,)
    t_grid : ndarray, shape (n_t,)
    """
    S_grid = np.linspace(S_min, S_max, n_S)
    # Pad t_max away from T so that tau = T - t > 0 for the last column
    t_max = T - 1e-3
    t_grid = np.linspace(t_min, t_max, n_t)

    V = np.zeros((n_S, n_t))
    for i, S in enumerate(S_grid):
        for j, t in enumerate(t_grid):
            tau = T - t
            V[i, j] = heston_call_price_scalar(
                S, K, r, tau, v0, kappa, theta, sigma_v, rho
            )

    return V, S_grid, t_grid


# ---------------------------------------------------------------------------
# SINDy on a surface — wrapper returning the 5 coefficients + R^2
# ---------------------------------------------------------------------------

def _run_bs_sindy_on_surface(V, S_grid, t_grid, K=100, T=1.0, sigma_for_qc=0.2,
                             r=0.05):
    """
    Run SINDy with the standard 5-term BS library on a given surface.
    Returns (coeffs (np.array len 5), r2).
    """
    set_all_seeds(42)
    result = discover_pde(
        V, S_grid, t_grid,
        true_sigma=sigma_for_qc, true_r=r,
        K=K, T=T, option_type='call',
    )
    coeffs = np.asarray(result['discovered_coefficients'], dtype=float)
    r2 = float(result['r2_score'])
    return coeffs, r2


# ---------------------------------------------------------------------------
# Sweeps (A), (B), (C)
# ---------------------------------------------------------------------------

def jump_intensity_sweep(lambdas=(0.0, 0.01, 0.05, 0.10, 0.20, 0.50),
                         mu_J=-0.10, sigma_J=0.15,
                         sigma=0.2, r=0.05, K=100, T=1.0):
    """
    Experiment A: vary Merton jump intensity lambda; record all 5 BS-library
    coefficients and R^2.

    Returns
    -------
    pandas.DataFrame
        One row per lambda.  Columns: experiment, parameter_name,
        parameter_value, coef_V, coef_dVdS, coef_d2VdS2, coef_S_dVdS,
        coef_S2_d2VdS2, R2.
    """
    rows = []
    for lam in lambdas:
        with Timer(f"Merton surface lam={lam}"):
            V, S_grid, t_grid = generate_merton_surface(
                S_min=50, S_max=150, n_S=100, t_min=0.0, n_t=100,
                K=K, r=r, sigma=sigma, T=T,
                lam=lam, mu_J=mu_J, sigma_J=sigma_J,
            )
        coeffs, r2 = _run_bs_sindy_on_surface(
            V, S_grid, t_grid, K=K, T=T, sigma_for_qc=sigma, r=r
        )
        rows.append({
            'experiment': 'jump_intensity',
            'parameter_name': 'lambda',
            'parameter_value': float(lam),
            'coef_V':           float(coeffs[0]),
            'coef_dVdS':        float(coeffs[1]),
            'coef_d2VdS2':      float(coeffs[2]),
            'coef_S_dVdS':      float(coeffs[3]),
            'coef_S2_d2VdS2':   float(coeffs[4]),
            'R2':               r2,
        })
        logger.info(
            f"lam={lam:.4f}  coef[d2V/dS2]={coeffs[2]:.4f}  "
            f"coef[S2 d2V/dS2]={coeffs[4]:.4f}  R2={r2:.6f}"
        )

    return pd.DataFrame(rows)


def jump_size_sweep(mu_Js=(-0.20, -0.10, -0.05, 0.0, 0.05, 0.10),
                    lam=0.10, sigma_J=0.15,
                    sigma=0.2, r=0.05, K=100, T=1.0):
    """
    Experiment B: vary Merton mean jump size mu_J at fixed lambda.
    """
    rows = []
    for mu_J in mu_Js:
        with Timer(f"Merton surface mu_J={mu_J}"):
            V, S_grid, t_grid = generate_merton_surface(
                S_min=50, S_max=150, n_S=100, t_min=0.0, n_t=100,
                K=K, r=r, sigma=sigma, T=T,
                lam=lam, mu_J=mu_J, sigma_J=sigma_J,
            )
        coeffs, r2 = _run_bs_sindy_on_surface(
            V, S_grid, t_grid, K=K, T=T, sigma_for_qc=sigma, r=r
        )
        rows.append({
            'experiment': 'jump_size',
            'parameter_name': 'mu_J',
            'parameter_value': float(mu_J),
            'coef_V':           float(coeffs[0]),
            'coef_dVdS':        float(coeffs[1]),
            'coef_d2VdS2':      float(coeffs[2]),
            'coef_S_dVdS':      float(coeffs[3]),
            'coef_S2_d2VdS2':   float(coeffs[4]),
            'R2':               r2,
        })
        logger.info(
            f"mu_J={mu_J:+.3f}  coef[d2V/dS2]={coeffs[2]:.4f}  "
            f"coef[V]={coeffs[0]:.4f}  R2={r2:.6f}"
        )

    return pd.DataFrame(rows)


def stochvol_sweep(sigma_vs=(0.0, 0.1, 0.2, 0.3, 0.5),
                   v0=0.04, kappa=2.0, theta=0.04, rho=-0.5,
                   r=0.05, K=100, T=1.0,
                   n_S=60, n_t=60):
    """
    Experiment C: vary Heston vol-of-vol sigma_v at fixed initial variance
    v0 = theta = 0.04 (matches BS sigma = 0.20).

    Smaller grid (60 x 60) is used because the semi-analytic Heston call
    requires ~3600 numerical integrations per surface; 100 x 100 would
    exceed the runtime budget.
    """
    rows = []
    for sv in sigma_vs:
        with Timer(f"Heston surface sigma_v={sv}"):
            V, S_grid, t_grid = generate_heston_surface(
                S_min=50, S_max=150, n_S=n_S, t_min=0.0, n_t=n_t,
                K=K, r=r, T=T,
                v0=v0, kappa=kappa, theta=theta, sigma_v=sv, rho=rho,
            )
        coeffs, r2 = _run_bs_sindy_on_surface(
            V, S_grid, t_grid, K=K, T=T,
            sigma_for_qc=float(np.sqrt(v0)), r=r,
        )
        rows.append({
            'experiment': 'stochvol',
            'parameter_name': 'sigma_v',
            'parameter_value': float(sv),
            'coef_V':           float(coeffs[0]),
            'coef_dVdS':        float(coeffs[1]),
            'coef_d2VdS2':      float(coeffs[2]),
            'coef_S_dVdS':      float(coeffs[3]),
            'coef_S2_d2VdS2':   float(coeffs[4]),
            'R2':               r2,
        })
        logger.info(
            f"sigma_v={sv:.3f}  coef[d2V/dS2]={coeffs[2]:.4f}  "
            f"coef[S2 d2V/dS2]={coeffs[4]:.4f}  R2={r2:.6f}"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _output_dirs():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tables_dir = os.path.join(project_root, 'outputs', 'tables')
    paper_fig_dir = os.path.join(project_root, 'outputs', 'figures', 'paper')
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(paper_fig_dir, exist_ok=True)
    return tables_dir, paper_fig_dir


def _savefig_paper(fig, base_name):
    """Save fig as both PNG and PDF at 300 DPI under outputs/figures/paper/."""
    import matplotlib.pyplot as plt
    _, paper_dir = _output_dirs()
    png_path = os.path.join(paper_dir, f"{base_name}.png")
    pdf_path = os.path.join(paper_dir, f"{base_name}.pdf")
    try:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    finally:
        plt.close(fig)
    logger.info(f"Saved paper figure: {base_name} (PNG+PDF)")
    return png_path


def _paper_rc():
    return {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times', 'serif'],
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.grid': False,
        'savefig.dpi': 300,
    }


# Okabe-Ito colorblind-safe palette
_CB_BLUE   = '#0072B2'
_CB_ORANGE = '#E69F00'
_CB_GREEN  = '#009E73'
_CB_PURPLE = '#CC79A7'
_CB_YELLOW = '#F0E442'


def plot_jump_sweep(df_jump):
    """
    Plot the spurious d^2V/dS^2 coefficient and R^2 as functions of jump
    intensity lambda.  Two y-axes: left = coefficient (blue), right = R^2
    (orange).
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    df = df_jump.sort_values('parameter_value').reset_index(drop=True)

    with mpl.rc_context(_paper_rc()):
        fig, ax_left = plt.subplots(figsize=(6.4, 4.2))
        ax_right = ax_left.twinx()

        ln1, = ax_left.plot(
            df['parameter_value'], df['coef_d2VdS2'],
            marker='o', color=_CB_BLUE, linewidth=1.8,
            label=r'Spurious $\partial^2 V/\partial S^2$ coefficient',
        )
        ax_left.set_xlabel(r'Jump intensity $\lambda$')
        ax_left.set_ylabel(r'Discovered $\partial^2 V/\partial S^2$ coefficient',
                           color=_CB_BLUE)
        ax_left.tick_params(axis='y', labelcolor=_CB_BLUE)
        ax_left.axhline(0.0, color='gray', linestyle=':', linewidth=0.8)

        ln2, = ax_right.plot(
            df['parameter_value'], df['R2'],
            marker='s', color=_CB_ORANGE, linewidth=1.8,
            linestyle='--', label=r'SINDy $R^2$ (clean surface)',
        )
        ax_right.set_ylabel(r'$R^2$', color=_CB_ORANGE)
        ax_right.tick_params(axis='y', labelcolor=_CB_ORANGE)

        ax_left.set_title(
            r'Jump misspecification: spurious term vs jump intensity'
        )
        ax_left.legend(handles=[ln1, ln2], loc='center right', frameon=True)

        fig.tight_layout()
        _savefig_paper(fig, 'misspec_jump_sweep')


def plot_stochvol_pattern(df_jump, df_stoch,
                          lam_match=0.20, sigma_v_match=0.30):
    """
    Compare the full 5-coefficient pattern for jumps (lambda=lam_match)
    versus stochastic volatility (sigma_v=sigma_v_match) at matched
    "severity".  Grouped bar chart.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    coef_cols = ['coef_V', 'coef_dVdS', 'coef_d2VdS2',
                 'coef_S_dVdS', 'coef_S2_d2VdS2']
    pretty_labels = [r'$V$', r'$\partial V/\partial S$',
                     r'$\partial^2 V/\partial S^2$',
                     r'$S\,\partial V/\partial S$',
                     r'$S^2\,\partial^2 V/\partial S^2$']

    def _pick(df, key, val):
        idx = (df['parameter_value'] - val).abs().idxmin()
        return df.loc[idx, coef_cols].values.astype(float)

    jump_vec  = _pick(df_jump, 'lambda', lam_match)
    stoch_vec = _pick(df_stoch, 'sigma_v', sigma_v_match)
    bs_truth  = np.array([0.05, 0.0, 0.0, -0.05, -0.5 * 0.2 ** 2])

    x = np.arange(5)
    width = 0.27

    with mpl.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.bar(x - width, jump_vec, width, color=_CB_BLUE,
               label=fr'Jumps ($\lambda={lam_match:.2f}$)')
        ax.bar(x,         stoch_vec, width, color=_CB_ORANGE,
               label=fr'Stoch vol ($\sigma_v={sigma_v_match:.2f}$)')
        ax.bar(x + width, bs_truth, width, color=_CB_GREEN,
               label=r'True BS coefficients ($\sigma=0.20$)')

        ax.axhline(0.0, color='gray', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(pretty_labels)
        ax.set_ylabel('Discovered coefficient')
        ax.set_title(
            'Misspecification signatures: jumps vs stochastic volatility'
        )
        ax.legend(loc='best', frameon=True)
        fig.tight_layout()
        _savefig_paper(fig, 'misspec_stochvol_pattern')


# ---------------------------------------------------------------------------
# End-to-end driver
# ---------------------------------------------------------------------------

def run_misspec_taxonomy(save=True):
    """
    Run all three sweeps, combine into one DataFrame, save CSV and figures.

    Returns
    -------
    dict with keys 'jump_intensity', 'jump_size', 'stochvol', 'combined'.
    """
    set_all_seeds(42)
    tables_dir, _ = _output_dirs()

    df_jump  = jump_intensity_sweep()
    df_size  = jump_size_sweep()
    df_stoch = stochvol_sweep()

    combined = pd.concat([df_jump, df_size, df_stoch], ignore_index=True)

    if save:
        out_csv = os.path.join(tables_dir, 'misspec_taxonomy.csv')
        combined.to_csv(out_csv, index=False)
        logger.info(f"Saved {out_csv}")

        plot_jump_sweep(df_jump)
        plot_stochvol_pattern(df_jump, df_stoch,
                              lam_match=0.20, sigma_v_match=0.30)

    return {
        'jump_intensity': df_jump,
        'jump_size': df_size,
        'stochvol': df_stoch,
        'combined': combined,
    }


if __name__ == '__main__':
    run_misspec_taxonomy()
