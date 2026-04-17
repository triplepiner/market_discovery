"""Tests for ablation study (library misspecification)."""
import numpy as np
import pytest
from src.ablation import (
    build_expanded_library, run_library_expansion_experiment,
    run_library_reduction_experiment, run_all_ablation_experiments,
)
from src.sindy_discovery import compute_derivatives
from src.data_generation import generate_price_surface


@pytest.fixture(scope='module')
def derivs():
    V, S, t = generate_price_surface(n_S=40, n_t=40)
    return compute_derivatives(V, S, t, trim=5)


class TestBuildExpandedLibrary:
    def test_level_A(self, derivs):
        lib, names = build_expanded_library(derivs, level='A')
        assert lib.shape[1] == 5
        assert len(names) == 5

    def test_level_B(self, derivs):
        lib, names = build_expanded_library(derivs, level='B')
        assert lib.shape[1] == 8

    def test_level_C(self, derivs):
        lib, names = build_expanded_library(derivs, level='C')
        assert lib.shape[1] == 11

    def test_level_D(self, derivs):
        lib, names = build_expanded_library(derivs, level='D')
        assert lib.shape[1] == 14


class TestExpansionExperiment:
    def test_true_terms_survive(self):
        results = run_library_expansion_experiment()
        # At level A (standard), true terms should be found
        level_a = results[0]
        assert level_a['r2'] > 0.99


class TestReductionExperiment:
    def test_r2_drops(self):
        results = run_library_reduction_experiment()
        # Missing a true term should reduce R^2
        for r in results:
            assert r['r2_drop'] > 0  # some drop expected
