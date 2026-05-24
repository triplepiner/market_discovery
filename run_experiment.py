"""Single-experiment CLI runner for the SINDy-KAN Black-Scholes project.

A reviewer who does not want to wait 30+ minutes for ``run_pipeline.py`` can
reproduce any single experiment in 2-3 minutes with::

    python run_experiment.py --experiment noise_comparison
    python run_experiment.py --experiment real_data_sindy
    python run_experiment.py --experiment kan_dupire
    python run_experiment.py --experiment misspec_taxonomy
    python run_experiment.py --experiment ablation
    python run_experiment.py --experiment discovery_baselines
    python run_experiment.py --experiment generalization
    python run_experiment.py --experiment activation_stability
    python run_experiment.py --experiment transfer
    python run_experiment.py --experiment all

Each experiment reads its parameters from ``config.yaml`` at the project
root. CPU only. Seed = 42 (first entry of ``seeds``).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any, Callable

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str = 'config.yaml') -> dict[str, Any]:
    """Load the YAML config from ``path`` (relative to the project root)."""
    cfg_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - yaml is in requirements
        raise RuntimeError(
            "PyYAML is required to load config.yaml. Install it via "
            "`pip install pyyaml`."
        ) from exc
    with open(cfg_path, 'r') as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

EXPERIMENTS: dict[str, str] = {
    'noise_comparison':    'Derivative-method noise sweep (FD/SavGol/Neural/Weak).',
    'real_data_sindy':     'Run v5 SINDy (windowed/quadratic/per-expiration) on 4 tickers.',
    'kan_dupire':          'Run [2,1] KAN-Dupire on all tickers.',
    'misspec_taxonomy':    'Jump-intensity, jump-size and stoch-vol misspec sweeps.',
    'ablation':            'Run the 6-config ablation chain on SPY.',
    'discovery_baselines': 'Weak-form / Ridge / Dupire discovery baselines + 5-fold CV.',
    'generalization':      'Per-expiration LOO + ATM->OTM regime transfer.',
    'activation_stability':'Multi-seed KAN activation stability sweep on SPY.',
    'transfer':            'Cross-ticker / temporal / mild-maturity transfer.',
    'all':                 'Run every experiment above sequentially.',
}


# ---------------------------------------------------------------------------
# Experiment implementations
# ---------------------------------------------------------------------------

def _ensure_outdirs(cfg: dict[str, Any]) -> None:
    paths = cfg.get('paths', {})
    for key in ('tables', 'figures_paper'):
        p = paths.get(key)
        if p:
            os.makedirs(os.path.join(PROJECT_ROOT, p), exist_ok=True)


def _seed(cfg: dict[str, Any]) -> int:
    seeds = cfg.get('seeds') or [42]
    return int(seeds[0])


def _outputs_table(cfg: dict[str, Any], name: str) -> str:
    p = cfg.get('paths', {}).get('tables', 'outputs/tables')
    return os.path.join(PROJECT_ROOT, p, name)


def exp_noise_comparison(cfg: dict[str, Any]) -> list[str]:
    """Re-run step 24 of run_pipeline (all-methods noise comparison).

    Uses the existing helpers (`discover_pde`, `sindy_with_neural_derivatives`,
    `weak_sindy_discover`, `adaptive_sindy_discover`) so the output schema
    matches what run_pipeline produces.
    """
    import numpy as np
    import pandas as pd
    from src.utils import set_all_seeds
    from src.data_generation import generate_price_surface, add_noise
    from src.sindy_discovery import (
        discover_pde, compute_r2_clean, compute_coefficient_metrics,
    )
    from src.neural_derivatives import sindy_with_neural_derivatives
    from src.weak_sindy import weak_sindy_discover

    seed = _seed(cfg)
    set_all_seeds(seed)
    syn = cfg['synthetic']
    nls = list(cfg['noise_levels'])
    K = float(syn['K']); R = float(syn['r']); SIGMA = float(syn['sigma']); T = float(syn['T'])
    n_S = int(syn['n_S']); n_t = int(syn['n_t'])

    V_clean, S_grid, t_grid = generate_price_surface(
        n_S=n_S, n_t=n_t, K=K, r=R, sigma=SIGMA, T=T,
    )
    methods = ['fd', 'savgol', 'neural', 'weak']
    rows: list[dict[str, Any]] = []
    for nl in nls:
        V = add_noise(V_clean, nl, seed=seed) if nl > 0 else V_clean.copy()
        for method in methods:
            try:
                if method == 'fd':
                    result = discover_pde(V, S_grid, t_grid,
                                          true_sigma=SIGMA, true_r=R,
                                          smooth=False, K=K, T=T)
                elif method == 'savgol':
                    result = discover_pde(V, S_grid, t_grid,
                                          true_sigma=SIGMA, true_r=R,
                                          smooth=True, savgol_window=21,
                                          savgol_poly=5, K=K, T=T)
                elif method == 'neural':
                    result = sindy_with_neural_derivatives(
                        V, S_grid, t_grid, true_sigma=SIGMA, true_r=R,
                        fit_epochs=200, seed=seed)
                else:  # weak
                    result = weak_sindy_discover(
                        V, S_grid, t_grid, n_test_functions=100,
                        true_sigma=SIGMA, true_r=R, seed=seed)
            except Exception as exc:
                print(f"  [{method} @ noise={nl}] inner failure: {exc}")
                continue
            coeffs = result['discovered_coefficients']
            r2_clean = compute_r2_clean(coeffs, S_grid, t_grid,
                                        K=K, r=R, sigma=SIGMA, T=T)
            cm = compute_coefficient_metrics(coeffs, R, SIGMA)
            rows.append({
                'noise_pct': nl, 'method': method,
                'r2_noisy': result['r2_score'], 'r2_clean': r2_clean,
                'n_active': result['n_active'],
                'coeff_V': cm['coeff_V'],
                'coeff_SdVdS': cm['coeff_SdVdS'],
                'coeff_S2d2VdS2': cm['coeff_S2d2VdS2'],
                'max_rel_err': cm['max_coeff_rel_error'],
                'correct_structure': cm['correct_structure'],
            })
    out = _outputs_table(cfg, 'all_methods_noise_comparison_v2.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    return [out]


def exp_real_data_sindy(cfg: dict[str, Any]) -> list[str]:
    import pandas as pd
    from src.utils import set_all_seeds
    from src.sindy_kan import _load_spy_option_data
    from src.real_data_v5 import run_v5_experiments_on_ticker

    seed = _seed(cfg)
    set_all_seeds(seed)
    tickers = list(cfg.get('real_data', {}).get('tickers') or ['SPY'])
    summary: list[dict[str, Any]] = []
    for tk in tickers:
        try:
            od = _load_spy_option_data(tk)
        except FileNotFoundError as exc:
            print(f"  [{tk}] skipped: {exc}")
            summary.append({'ticker': tk, 'status': 'missing_chain'})
            continue
        try:
            res = run_v5_experiments_on_ticker(od, tk, seed=seed)
            errors = res.get('errors') or {}
            summary.append({'ticker': tk,
                            'status': 'ok' if not errors else 'partial',
                            'errors': ';'.join(errors.keys())})
        except Exception as exc:
            summary.append({'ticker': tk, 'status': f'failed:{exc}'})
    out = _outputs_table(cfg, 'real_data_sindy_summary.csv')
    pd.DataFrame(summary).to_csv(out, index=False)
    return [out]


def exp_kan_dupire(cfg: dict[str, Any]) -> list[str]:
    import pandas as pd
    from src.utils import set_all_seeds
    from src.sindy_kan import (
        sindy_kan_dupire_all_tickers, _load_spy_option_data,
    )

    seed = _seed(cfg)
    set_all_seeds(seed)
    tickers = list(cfg.get('real_data', {}).get('tickers') or ['SPY'])
    per_ticker: dict[str, Any] = {}
    for tk in tickers:
        try:
            per_ticker[tk] = {'option_data': _load_spy_option_data(tk)}
        except FileNotFoundError as exc:
            print(f"  [{tk}] skipped: {exc}")
    if not per_ticker:
        raise RuntimeError("No cached chains available for KAN-Dupire run.")
    results = sindy_kan_dupire_all_tickers(per_ticker, seed=seed)
    rows = []
    for tk, r in results.items():
        rows.append({
            'ticker': tk,
            'r2_train': r.get('r2_train'),
            'r2_test': r.get('r2_test'),
            'sigma_loc_median': r.get('sigma_loc_median'),
            'error': r.get('error'),
        })
    out = _outputs_table(cfg, 'kan_dupire_summary.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    return [out]


def exp_misspec_taxonomy(cfg: dict[str, Any]) -> list[str]:
    from src.utils import set_all_seeds
    from src.misspec_taxonomy import run_misspec_taxonomy
    set_all_seeds(_seed(cfg))
    run_misspec_taxonomy(save=True)
    return [_outputs_table(cfg, 'misspec_taxonomy.csv')]


def exp_ablation(cfg: dict[str, Any]) -> list[str]:
    from src.utils import set_all_seeds
    from src.ablation_chain import run_ablation_chain
    set_all_seeds(_seed(cfg))
    out = _outputs_table(cfg, 'ablation_chain.csv')
    run_ablation_chain(ticker='SPY', save_csv=out)
    return [out]


def exp_discovery_baselines(cfg: dict[str, Any]) -> list[str]:
    """Delegates to scripts/run_discovery_baselines_cv.py:main()."""
    from src.utils import set_all_seeds
    set_all_seeds(_seed(cfg))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
    try:
        import importlib
        mod = importlib.import_module('run_discovery_baselines_cv')
        rc = mod.main()
        if rc not in (None, 0):
            raise RuntimeError(f"discovery_baselines returned {rc}")
    finally:
        sys.path.pop(0)
    return [_outputs_table(cfg, 'discovery_method_comparison.csv'),
            _outputs_table(cfg, 'cv_results.csv')]


def exp_generalization(cfg: dict[str, Any]) -> list[str]:
    from src.utils import set_all_seeds
    set_all_seeds(_seed(cfg))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
    try:
        import importlib
        mod = importlib.import_module('run_generalization_analysis')
        rc = mod.main()
        if rc not in (None, 0):
            raise RuntimeError(f"generalization returned {rc}")
    finally:
        sys.path.pop(0)
    return [_outputs_table(cfg, 'generalization_analysis.csv'),
            _outputs_table(cfg, 'per_expiration_coefficients.csv')]


def exp_activation_stability(cfg: dict[str, Any]) -> list[str]:
    from src.utils import set_all_seeds
    from src.sindy_kan import activation_stability_sweep
    seed = _seed(cfg)
    set_all_seeds(seed)
    seeds = tuple(int(s) for s in (cfg.get('seeds') or [42]))
    out = _outputs_table(cfg, 'activation_stability.csv')
    activation_stability_sweep(ticker='SPY', seeds=seeds, save_csv=out)
    return [out]


def exp_transfer(cfg: dict[str, Any]) -> list[str]:
    from src.utils import set_all_seeds
    from src.transfer_experiments import run_all_transfer_experiments
    set_all_seeds(_seed(cfg))
    run_all_transfer_experiments(seed=_seed(cfg))
    return [_outputs_table(cfg, 'transfer_experiments.csv'),
            _outputs_table(cfg, 'per_expiration_loo.csv')]


HANDLERS: dict[str, Callable[[dict[str, Any]], list[str]]] = {
    'noise_comparison':     exp_noise_comparison,
    'real_data_sindy':      exp_real_data_sindy,
    'kan_dupire':           exp_kan_dupire,
    'misspec_taxonomy':     exp_misspec_taxonomy,
    'ablation':             exp_ablation,
    'discovery_baselines':  exp_discovery_baselines,
    'generalization':       exp_generalization,
    'activation_stability': exp_activation_stability,
    'transfer':             exp_transfer,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_single(name: str, cfg: dict[str, Any]) -> bool:
    """Run one experiment by name. Returns True on success."""
    handler = HANDLERS.get(name)
    if handler is None:
        print(f"[{name}] FAILED: unknown experiment")
        return False
    print(f"[{name}] starting (seed={_seed(cfg)})")
    try:
        outputs = handler(cfg) or []
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=2)
        print(f"[{name}] FAILED: {exc}\n{tb}")
        return False
    rel = [os.path.relpath(p, PROJECT_ROOT) for p in outputs]
    print(f"[{name}] done (wrote {rel})")
    return True


def run_all(cfg: dict[str, Any]) -> int:
    failures = 0
    for name in HANDLERS:
        ok = run_single(name, cfg)
        if not ok:
            failures += 1
    return failures


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Run a single reproducibility experiment from config.yaml.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Available experiments:\n  ' + '\n  '.join(
            f'{k:<22s} {v}' for k, v in EXPERIMENTS.items()),
    )
    p.add_argument(
        '--experiment', '-e', required=True,
        choices=list(EXPERIMENTS.keys()),
        help='Which experiment to run.',
    )
    p.add_argument(
        '--config', '-c', default='config.yaml',
        help='Path to the YAML config (default: config.yaml).',
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _ensure_outdirs(cfg)
    if args.experiment == 'all':
        return run_all(cfg)
    return 0 if run_single(args.experiment, cfg) else 1


if __name__ == '__main__':
    sys.exit(main())
