"""Tests for Improvement 4: reproducibility config + CLI runner.

These tests only verify wiring — they do NOT execute any experiment, since
the experiments themselves take 2-3 minutes each and are exercised by their
own dedicated test modules (test_misspec_taxonomy, test_sindy_kan, etc.).
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.yaml')
RUNNER_PATH = os.path.join(PROJECT_ROOT, 'run_experiment.py')


REQUIRED_KEYS = {
    'paths', 'seeds', 'synthetic', 'noise_levels',
    'real_data', 'gp', 'kan', 'pinn',
}


def test_config_loads():
    """config.yaml parses with yaml.safe_load and has all required top-level keys."""
    yaml = pytest.importorskip('yaml')
    assert os.path.isfile(CONFIG_PATH), f"missing config.yaml at {CONFIG_PATH}"
    with open(CONFIG_PATH, 'r') as fh:
        cfg = yaml.safe_load(fh)
    assert isinstance(cfg, dict), 'config.yaml did not parse to a dict'
    missing = REQUIRED_KEYS - set(cfg.keys())
    assert not missing, f"config.yaml missing keys: {missing}"

    # Sanity-check a few values from the PRD.
    assert cfg['synthetic']['K'] == 100
    assert cfg['synthetic']['sigma'] == 0.20
    assert cfg['kan']['architecture'] == [2, 1]
    assert cfg['seeds'][0] == 42
    assert 0.05 in cfg['noise_levels']


def test_experiment_runner_imports():
    """run_experiment.py can be imported without executing experiments.

    The script guards its main body with ``if __name__ == "__main__":`` so a
    plain import must not trigger argparse or any heavy work.
    """
    assert os.path.isfile(RUNNER_PATH), f"missing run_experiment.py at {RUNNER_PATH}"
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    # Drop any cached copy so we get a fresh import.
    sys.modules.pop('run_experiment', None)
    mod = importlib.import_module('run_experiment')
    assert hasattr(mod, 'HANDLERS') and isinstance(mod.HANDLERS, dict)
    # Verify each PRD-mandated experiment is wired up.
    for name in (
        'noise_comparison', 'real_data_sindy', 'kan_dupire',
        'misspec_taxonomy', 'ablation', 'discovery_baselines',
        'generalization', 'activation_stability', 'transfer',
    ):
        assert name in mod.HANDLERS, f"experiment {name!r} not registered"
    # And 'all' is exposed via the CLI choices but routed separately.
    assert 'all' in mod.EXPERIMENTS
    # Parser builds without error.
    parser = mod.build_parser()
    assert parser is not None
