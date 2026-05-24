"""Run Improvements 2 + 3 of the Remaining-Feedback PRD.

Writes:
  outputs/tables/discovery_method_comparison.csv  (5 rows)
  outputs/tables/cv_results.csv                   (5 folds x 2 models)
  outputs/tables/cv_kan_activations_summary.csv   (per-fold edge sweeps)

CPU only. Seed = 42 throughout.
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from src.discovery_baselines_cv import (
    build_spy_gp_dupire_dataset,
    run_discovery_method_comparison,
    run_cv,
)


def main() -> int:
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 80)

    t0 = time.perf_counter()
    dataset = build_spy_gp_dupire_dataset(seed=42)
    print(f"[setup]   GP-Dupire SPY dataset ready in {time.perf_counter()-t0:.1f}s "
          f"snapshot={dataset.get('snapshot_path')!s}")

    t1 = time.perf_counter()
    comp_df = run_discovery_method_comparison(dataset=dataset)
    print(f"\n[Part A]  Discovery method comparison "
          f"(elapsed {time.perf_counter()-t1:.1f}s)")
    print(comp_df.to_string(index=False))

    t2 = time.perf_counter()
    cv_out = run_cv(dataset=dataset)
    print(f"\n[Part B]  5-fold CV results "
          f"(elapsed {time.perf_counter()-t2:.1f}s)")
    print(cv_out['combined'].to_string(index=False))

    lin_mean = cv_out['linear']['mean_R2_test'].iloc[0] \
        if len(cv_out['linear']) > 0 else float('nan')
    lin_std = cv_out['linear']['std_R2_test'].iloc[0] \
        if len(cv_out['linear']) > 0 else float('nan')
    kan_mean = cv_out['kan']['mean_R2_test'].iloc[0] \
        if len(cv_out['kan']) > 0 else float('nan')
    kan_std = cv_out['kan']['std_R2_test'].iloc[0] \
        if len(cv_out['kan']) > 0 else float('nan')
    print("\n[Part B]  summary:")
    print(f"  Linear Dupire CV R^2: {lin_mean:.4f} +/- {lin_std:.4f}")
    print(f"  KAN [2,1]    CV R^2: {kan_mean:.4f} +/- {kan_std:.4f}")
    print(f"\n[done]    total runtime {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
