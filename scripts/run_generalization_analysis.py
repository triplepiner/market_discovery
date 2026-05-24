"""Run Improvement 5: per-expiration coefficients + regime transfer (SPY).

Writes:
  outputs/tables/per_expiration_coefficients.csv
  outputs/tables/generalization_analysis.csv
  outputs/figures/paper/coefficient_vs_maturity.{png,pdf}
  outputs/figures/paper/transfer_heatmap.{png,pdf}

CPU only, seed=42. Runtime budget ~5 min.
"""
from __future__ import annotations

import os
import sys

# Make the repo root importable when invoked from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.transfer_experiments import run_generalization_analysis


def main():
    os.chdir(ROOT)
    res = run_generalization_analysis(
        ticker='SPY', date='20260329', n_epochs=1500, seed=42)
    coef_df = res['coef_df']
    regime_df = res['regime_df']

    print("\n=== Per-expiration coefficients (SPY 20260329) ===")
    print(coef_df.to_string(index=False))

    print("\n=== Regime transfer (SPY) ===")
    print(regime_df.to_string(index=False))

    # Quick correlation between coef_diffusion and market_avg_iv.
    import numpy as np
    a = coef_df['coef_diffusion'].values.astype(float)
    b = coef_df['market_avg_iv'].values.astype(float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() >= 3:
        # Also compute correlation with iv^2 since coef is ~0.5 * sigma^2.
        c = b[ok] ** 2
        corr_iv = float(np.corrcoef(a[ok], b[ok])[0, 1])
        corr_iv2 = float(np.corrcoef(a[ok], c)[0, 1])
        print(f"\nCorr(coef_diffusion, market_avg_iv)  = {corr_iv:+.3f}")
        print(f"Corr(coef_diffusion, market_avg_iv^2)= {corr_iv2:+.3f}")
        print(f"n = {int(ok.sum())}")

    print("\nFiles written:")
    print(f"  {res['coef_csv']}")
    print(f"  {res['generalization_csv']}")
    print(f"  {res['coef_figure']}")
    print(f"  {res['heatmap_figure']}")


if __name__ == '__main__':
    main()
