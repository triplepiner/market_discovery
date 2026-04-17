"""
Ablation study: test SINDy discovery when the candidate library is wrong.

Experiments include:
- Library expansion: adding distractor terms that are not in the true PDE
- Library reduction: removing terms that *are* in the true PDE
- Expansion under noise: distractor terms combined with noisy data

These experiments characterise the robustness of the STLSQ sparse regression
pipeline and inform how sensitive the discovery is to modelling choices.
"""

import numpy as np
from src.utils import set_all_seeds, setup_logging, Timer, safe_relative_error
from src.sindy_discovery import (
    compute_derivatives, stlsq_sweep, format_pde_string, TERM_NAMES,
)
from src.data_generation import generate_price_surface, add_noise

logger = setup_logging(__name__)

# Indices of the three true Black-Scholes terms in the standard 5-term library
_TRUE_INDICES = [0, 3, 4]  # V, S*dV/dS, S2*d2V/dS2
_TRUE_TERM_NAMES = ['V', 'S*dV/dS', 'S2*d2V/dS2']


# ---------------------------------------------------------------------------
# Expanded library builder
# ---------------------------------------------------------------------------

def build_expanded_library(derivs, level='B'):
    """Build expanded candidate libraries with distractor terms.

    Level A (5 terms - standard): V, dV/dS, d2V/dS2, S*dV/dS, S^2*d2V/dS2
    Level B (8 terms): A + S^3*d2V/dS2, V^2, S*V
    Level C (11 terms): B + sin(S/100), exp(-S/100), sqrt(S)*dV/dS
    Level D (14 terms): C + log(S)*d2V/dS2, (dV/dS)^2, S^2*V

    Parameters
    ----------
    derivs : dict
        Output of ``compute_derivatives``.  Must contain keys
        ``'V'``, ``'dVdS'``, ``'d2VdS2'``, ``'S_mesh'``.
    level : str
        Library complexity level: ``'A'``, ``'B'``, ``'C'``, or ``'D'``.

    Returns
    -------
    library : ndarray, shape (n_points, n_terms)
    term_names : list of str
    """
    V = derivs['V']
    dVdS = derivs['dVdS']
    d2VdS2 = derivs['d2VdS2']
    S_mesh = derivs['S_mesh']

    # Standard 5-column library (Level A)
    columns = [
        V.ravel(),
        dVdS.ravel(),
        d2VdS2.ravel(),
        (S_mesh * dVdS).ravel(),
        (S_mesh ** 2 * d2VdS2).ravel(),
    ]
    names = list(TERM_NAMES)  # copy

    level = level.upper()

    if level in ('B', 'C', 'D'):
        columns.append((S_mesh ** 3 * d2VdS2).ravel())
        names.append('S3*d2V/dS2')
        columns.append((V ** 2).ravel())
        names.append('V2')
        columns.append((S_mesh * V).ravel())
        names.append('S*V')

    if level in ('C', 'D'):
        columns.append(np.sin(S_mesh / 100.0).ravel())
        names.append('sin(S/100)')
        columns.append(np.exp(-S_mesh / 100.0).ravel())
        names.append('exp(-S/100)')
        columns.append((np.sqrt(S_mesh) * dVdS).ravel())
        names.append('sqrt(S)*dV/dS')

    if level == 'D':
        columns.append((np.log(S_mesh) * d2VdS2).ravel())
        names.append('log(S)*d2V/dS2')
        columns.append((dVdS ** 2).ravel())
        names.append('(dV/dS)2')
        columns.append((S_mesh ** 2 * V).ravel())
        names.append('S2*V')

    library = np.column_stack(columns)

    cond = np.linalg.cond(library)
    logger.info(
        f"Expanded library (level {level}): {library.shape[1]} terms, "
        f"condition number = {cond:.2e}"
    )

    return library, names


# ---------------------------------------------------------------------------
# Experiment 1 – library expansion
# ---------------------------------------------------------------------------

def run_library_expansion_experiment(K=100, r=0.05, sigma=0.2, T=1.0):
    """Test SINDy with progressively larger libraries on clean BS data.

    Generates a clean Black-Scholes call price surface, computes derivatives
    once, then builds candidate libraries at levels A through D.  For each
    level the STLSQ sweep is run and the results are compared against the
    known ground truth.

    Parameters
    ----------
    K, r, sigma, T : float
        Black-Scholes parameters.

    Returns
    -------
    list of dict
        One entry per library level with keys:
        ``level``, ``n_terms``, ``term_names``, ``coefficients``, ``r2``,
        ``active_mask``, ``n_active``, ``true_term_coefficients``,
        ``true_term_active``, ``false_positives``, ``false_negatives``,
        ``condition_number``.
    """
    logger.info("=== Library expansion experiment ===")

    V, S_grid, t_grid = generate_price_surface(K=K, r=r, sigma=sigma, T=T)
    derivs = compute_derivatives(V, S_grid, t_grid)
    target = derivs['dVdt'].ravel()

    results = []
    for level in ['A', 'B', 'C', 'D']:
        logger.info(f"--- Level {level} ---")
        library, names = build_expanded_library(derivs, level=level)
        best, _ = stlsq_sweep(library, target)

        coeffs = best['coefficients']
        active_mask = best['active_mask']
        n_terms = len(names)

        # Condition number of the library matrix
        condition_number = np.linalg.cond(library)

        # True-term analysis (indices 0, 3, 4 are the same across all levels)
        true_term_coefficients = {
            name: float(coeffs[idx])
            for name, idx in zip(_TRUE_TERM_NAMES, _TRUE_INDICES)
        }
        true_term_active = all(active_mask[idx] for idx in _TRUE_INDICES)

        # False positives: activated terms that are not among the three true terms
        false_positives = [
            names[i] for i in range(n_terms)
            if active_mask[i] and i not in _TRUE_INDICES
        ]
        # False negatives: true terms that were zeroed out
        false_negatives = [
            _TRUE_TERM_NAMES[j] for j, idx in enumerate(_TRUE_INDICES)
            if not active_mask[idx]
        ]

        pde_str = format_pde_string(coeffs, names)
        logger.info(f"Discovered PDE: {pde_str}")
        logger.info(
            f"True terms active: {true_term_active}, "
            f"FP: {false_positives}, FN: {false_negatives}"
        )

        results.append({
            'level': level,
            'n_terms': n_terms,
            'term_names': names,
            'coefficients': coeffs,
            'r2': best['r2'],
            'active_mask': active_mask,
            'n_active': best['n_active'],
            'true_term_coefficients': true_term_coefficients,
            'true_term_active': true_term_active,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'condition_number': condition_number,
        })

    return results


# ---------------------------------------------------------------------------
# Experiment 2 – library reduction
# ---------------------------------------------------------------------------

def run_library_reduction_experiment(K=100, r=0.05, sigma=0.2, T=1.0):
    """Test SINDy with missing true terms.

    Builds three reduced libraries (E, F, G) each missing one of the three
    true Black-Scholes terms and reports how the remaining coefficients and
    R^2 change.

    Parameters
    ----------
    K, r, sigma, T : float
        Black-Scholes parameters.

    Returns
    -------
    list of dict
        One entry per reduced library (E, F, G) with keys:
        ``label``, ``missing_term``, ``n_terms``, ``term_names``,
        ``coefficients``, ``r2``, ``active_mask``, ``n_active``,
        ``r2_drop``.
    """
    logger.info("=== Library reduction experiment ===")

    V, S_grid, t_grid = generate_price_surface(K=K, r=r, sigma=sigma, T=T)
    derivs = compute_derivatives(V, S_grid, t_grid)
    target = derivs['dVdt'].ravel()

    V_flat = derivs['V'].ravel()
    dVdS_flat = derivs['dVdS'].ravel()
    d2VdS2_flat = derivs['d2VdS2'].ravel()
    S_flat = derivs['S_mesh'].ravel()

    # Full-library baseline R^2 (level A)
    full_library, full_names = build_expanded_library(derivs, level='A')
    full_best, _ = stlsq_sweep(full_library, target)
    full_r2 = full_best['r2']
    logger.info(f"Full library R^2 = {full_r2:.6f}")

    # Define reduced libraries
    experiments = [
        {
            'label': 'E',
            'missing_term': 'V',
            'columns': [
                dVdS_flat,
                d2VdS2_flat,
                (S_flat * dVdS_flat),
                (S_flat ** 2 * d2VdS2_flat),
            ],
            'names': ['dV/dS', 'd2V/dS2', 'S*dV/dS', 'S2*d2V/dS2'],
        },
        {
            'label': 'F',
            'missing_term': 'S2*d2V/dS2',
            'columns': [
                V_flat,
                dVdS_flat,
                d2VdS2_flat,
                (S_flat * dVdS_flat),
            ],
            'names': ['V', 'dV/dS', 'd2V/dS2', 'S*dV/dS'],
        },
        {
            'label': 'G',
            'missing_term': 'S*dV/dS',
            'columns': [
                V_flat,
                dVdS_flat,
                d2VdS2_flat,
                (S_flat ** 2 * d2VdS2_flat),
            ],
            'names': ['V', 'dV/dS', 'd2V/dS2', 'S2*d2V/dS2'],
        },
    ]

    results = []
    for exp in experiments:
        logger.info(f"--- Library {exp['label']}: missing {exp['missing_term']} ---")
        library = np.column_stack(exp['columns'])
        best, _ = stlsq_sweep(library, target)

        coeffs = best['coefficients']
        active_mask = best['active_mask']
        r2 = best['r2']
        r2_drop = full_r2 - r2

        pde_str = format_pde_string(coeffs, exp['names'])
        logger.info(f"Discovered PDE: {pde_str}")
        logger.info(f"R^2 = {r2:.6f}, drop = {r2_drop:.6f}")

        results.append({
            'label': exp['label'],
            'missing_term': exp['missing_term'],
            'n_terms': len(exp['names']),
            'term_names': exp['names'],
            'coefficients': coeffs,
            'r2': r2,
            'active_mask': active_mask,
            'n_active': best['n_active'],
            'r2_drop': r2_drop,
        })

    return results


# ---------------------------------------------------------------------------
# Experiment 3 – expanded library under noise
# ---------------------------------------------------------------------------

def run_expansion_noise_experiment(noise_pct=0.05, K=100, r=0.05, sigma=0.2,
                                   T=1.0):
    """Test library C (11 terms) with 5 % noise.

    Adds Gaussian noise to the clean price surface, then uses Savitzky-Golay
    smoothing before differentiation.  The 11-term level-C library is
    constructed and STLSQ is run.  Metrics quantify how noise degrades the
    ability to reject distractor terms.

    Parameters
    ----------
    noise_pct : float
        Noise level as a fraction of the surface's standard deviation.
    K, r, sigma, T : float
        Black-Scholes parameters.

    Returns
    -------
    dict
        Keys: ``n_terms``, ``term_names``, ``coefficients``, ``r2``,
        ``active_mask``, ``n_active``, ``true_term_coefficients``,
        ``true_term_active``, ``false_positives``, ``false_negatives``,
        ``false_positive_rate``, ``false_negative_rate``,
        ``coefficient_errors``, ``condition_number``.
    """
    logger.info(f"=== Expansion + noise experiment (noise={noise_pct:.1%}) ===")

    V_clean, S_grid, t_grid = generate_price_surface(K=K, r=r, sigma=sigma, T=T)
    V_noisy = add_noise(V_clean, noise_pct)
    derivs = compute_derivatives(
        V_noisy, S_grid, t_grid, smooth=True, savgol_window=7,
    )
    target = derivs['dVdt'].ravel()

    library, names = build_expanded_library(derivs, level='C')
    best, _ = stlsq_sweep(library, target)

    coeffs = best['coefficients']
    active_mask = best['active_mask']
    n_terms = len(names)

    condition_number = np.linalg.cond(library)

    # True-term analysis
    true_coeffs_expected = {
        'V': -r,             # note: sign convention in BS PDE dV/dt = -rV + ...
        'S*dV/dS': r,        # placeholder — actual values depend on the regression
        'S2*d2V/dS2': 0.5 * sigma ** 2,
    }
    # Use the discovered values
    true_term_coefficients = {
        name: float(coeffs[idx])
        for name, idx in zip(_TRUE_TERM_NAMES, _TRUE_INDICES)
    }
    true_term_active = all(active_mask[idx] for idx in _TRUE_INDICES)

    # Coefficient errors against known BS coefficients
    # True PDE: dV/dt = -rV + rS*dV/dS + 0.5*sigma^2*S^2*d2V/dS2
    true_coeff_values = np.array([-r, r, 0.5 * sigma ** 2])
    discovered_true = np.array([coeffs[idx] for idx in _TRUE_INDICES])
    coefficient_errors = safe_relative_error(discovered_true, true_coeff_values)

    # False positives / negatives
    false_positives = [
        names[i] for i in range(n_terms)
        if active_mask[i] and i not in _TRUE_INDICES
    ]
    false_negatives = [
        _TRUE_TERM_NAMES[j] for j, idx in enumerate(_TRUE_INDICES)
        if not active_mask[idx]
    ]

    # Rates
    n_distractor = n_terms - len(_TRUE_INDICES)
    n_fp = len(false_positives)
    n_fn = len(false_negatives)
    false_positive_rate = n_fp / max(n_distractor, 1)
    false_negative_rate = n_fn / max(len(_TRUE_INDICES), 1)

    pde_str = format_pde_string(coeffs, names)
    logger.info(f"Discovered PDE: {pde_str}")
    logger.info(
        f"FP rate: {false_positive_rate:.2f}, FN rate: {false_negative_rate:.2f}"
    )

    return {
        'n_terms': n_terms,
        'term_names': names,
        'coefficients': coeffs,
        'r2': best['r2'],
        'active_mask': active_mask,
        'n_active': best['n_active'],
        'true_term_coefficients': true_term_coefficients,
        'true_term_active': true_term_active,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'coefficient_errors': coefficient_errors,
        'condition_number': condition_number,
    }


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def run_all_ablation_experiments(K=100, r=0.05, sigma=0.2, T=1.0):
    """Run all ablation experiments and return combined results.

    Seeds all random number generators for reproducibility, then executes:
    1. Library expansion (levels A-D on clean data)
    2. Library reduction (missing true terms)
    3. Expansion under noise (level C with 5 % noise)

    Parameters
    ----------
    K, r, sigma, T : float
        Black-Scholes parameters.

    Returns
    -------
    dict
        Keys: ``'expansion'``, ``'reduction'``, ``'expansion_noise'``.
    """
    set_all_seeds(42)

    with Timer("Library expansion experiment"):
        expansion = run_library_expansion_experiment(K=K, r=r, sigma=sigma, T=T)

    with Timer("Library reduction experiment"):
        reduction = run_library_reduction_experiment(K=K, r=r, sigma=sigma, T=T)

    with Timer("Expansion + noise experiment"):
        expansion_noise = run_expansion_noise_experiment(
            K=K, r=r, sigma=sigma, T=T,
        )

    logger.info("All ablation experiments complete.")

    return {
        'expansion': expansion,
        'reduction': reduction,
        'expansion_noise': expansion_noise,
    }
