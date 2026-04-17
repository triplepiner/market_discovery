# Black-Scholes PDE Discovery from Market Data

## Overview

This project discovers the Black-Scholes partial differential equation from synthetic option price data using the SINDy (Sparse Identification of Nonlinear Dynamics) algorithm, then validates the discovered PDE by solving it with a Physics-Informed Neural Network (PINN) and comparing against closed-form analytical solutions. The pipeline extends well beyond the core discovery loop: it includes baseline comparisons (Dense OLS, LassoCV, Ridge+Threshold, symbolic regression), extended model experiments (Merton jump-diffusion, Heston stochastic volatility), library ablation studies, noise-smoothing ablation, and a real market data pipeline using Yahoo Finance.

On clean data, SINDy recovers the three key PDE coefficients with < 0.65% relative error and R^2 = 0.999998. A reduced 3-term candidate library eliminates multicollinearity (condition number: 44.2 vs 6.78e+04) and achieves perfect PDE structure recovery. The PINN trained on the discovered PDE achieves 1.01% relative L2 pricing error for call options. Under 5% noise, all methods collapse -- our systematic noise-smoothing ablation (16 combinations) confirms that smoothing alone cannot recover PDE structure, motivating integral-form methods.

## Motivation

The Black-Scholes equation is the foundation of quantitative finance, but it rests on strong assumptions (constant volatility, no transaction costs, continuous hedging). When these assumptions fail, practitioners need to discover the actual governing PDE from observed market data. Recent advances in data-driven methods, particularly SINDy (Brunton et al., 2016), offer a principled way to identify PDEs directly from data without assuming a model form.

This project uses the Black-Scholes equation as a controlled testbed: we generate synthetic option prices from the known analytical formula, then "forget" the equation and attempt to rediscover it. This closed-loop approach gives us ground truth to measure accuracy against. We then stress-test with noise, alternative models (Merton, Heston), library misspecification, baseline comparisons, and real market data to understand the method's strengths and limitations.

## Pipeline Architecture

```
Synthetic BS Data --> Numerical Derivatives --> SINDy Sparse Regression --> Discovered PDE
                                                                                |
Analytical Solution <-- Compare <-- PINN Solution <-- PINN Training <-- PDE Coefficients
                                                                                |
                    Baselines / Extended Models / Ablation / Real Data Experiments
```

1. **Data Generation**: Generate call/put price surfaces from closed-form Black-Scholes formulas on a 100x100 grid. Supports Merton jump-diffusion pricing and noise injection.
2. **Numerical Differentiation**: Compute dV/dt, dV/dS, d2V/dS2 via central finite differences with boundary trimming.
3. **SINDy Discovery**: Build both a full 5-term and reduced 3-term candidate library, then use Sequential Thresholded Least Squares (STLSQ) to identify the active terms and their coefficients. Post-processing and correlation analysis diagnose multicollinearity.
4. **PINN Validation**: Train a neural network to solve the discovered PDE, enforcing it as a soft constraint alongside boundary conditions and observed data. Regional error analysis decomposes accuracy into ATM/ITM/OTM regions.
5. **Greeks Computation**: Compute Delta and Gamma from both the PINN and analytical formulas; compare on the interior grid.
6. **Baseline Comparisons**: Dense OLS, LassoCV, Ridge+Threshold, and gplearn symbolic regression on the same candidate library.
7. **Extended Models**: Merton jump-diffusion discovery and Heston variance-slicing linearity test.
8. **Ablation Study**: Library expansion (5, 8, 11, 14 terms), leave-one-out reduction, and noise interaction.
9. **Noise-Smoothing Ablation**: Savitzky-Golay smoothing ablation, grid resolution vs noise, and 4x4 noise-smoothing matrix.
10. **Real Market Data**: Pipeline for 4 tickers (SPY, QQQ, AAPL, MSFT) via yfinance with date-stamped caching and mock fallback.

## Key Results

### Core SINDy Discovery

| Metric | Value |
|--------|-------|
| Full library R^2 | 0.999998 |
| Full library condition number | 6.78e+04 |
| Full library active terms | 5 (2 false positives) |
| Reduced library R^2 | 0.999998 |
| Reduced library condition number | 44.2 |
| Reduced library active terms | 3 (perfect structure recovery) |

**Recovered coefficients (reduced library):**

| Term | Discovered | True | Relative Error |
|------|-----------|------|---------------|
| V | 0.0497 | 0.0500 | 0.65% |
| S*dV/dS | -0.0499 | -0.0500 | 0.20% |
| S^2*d2V/dS^2 | -0.0200 | -0.0200 | 0.13% |

**Multicollinearity diagnosis:** Post-processing with threshold 0.1*max(|coeff|) cannot fix the full library -- the spurious d2V/dS2 term (coefficient 0.555) is the *largest*, so thresholding eliminates the true terms instead. Pairwise correlations: corr(dV/dS, S*dV/dS) = 0.986, corr(d2V/dS2, S^2*d2V/dS2) = 0.969.

### PINN Validation

| Metric | Call | Put |
|--------|------|-----|
| Relative L2 error | 1.01% | 47.9% (4.53% ATM) |
| R^2 | 0.9998 | 0.601 |
| Training epochs | 5,000 | 10,000 |
| Training time | 487s | 986s |

**Put PINN regional breakdown:**

| Region | Relative L2 | MAE |
|--------|------------|-----|
| Full grid | 46.1% | $2.81 |
| ATM (0.8K-1.2K) | **4.53%** | $0.52 |
| ITM (< 0.8K) | 62.1% | $5.11 |
| OTM (> 1.2K) | 28.3% | $0.98 |

| Greek | MAE |
|-------|-----|
| Delta | 0.0125 |
| Gamma | 0.0020 |

### Noise Robustness

Under 5% noise, all methods collapse (R^2 < 0.001). Systematic investigation:

- **Smoothing ablation**: Best Savitzky-Golay at 5% noise: (21,5) -> R^2 = 0.053, correct structure = No
- **Grid resolution**: 30x30 to 200x200 at 5% noise all yield R^2 < 0.001
- **Noise-smoothing matrix**: 16 combinations tested; no setting recovers structure above 1% noise

### Baseline Comparisons

**Clean data:**

| Method | R^2 | Active Terms |
|--------|-----|-------------|
| Dense OLS | 0.999998 | 5 |
| LassoCV | 0.999929 | 3 |
| Ridge+Threshold | 0.999997 | 5 |
| gplearn Symbolic | 0.804 | -- |

**5% noise:** All baselines collapse (R^2 < 0.001), confirming derivative estimation is the bottleneck.

### Merton Jump-Diffusion

| Metric | Value |
|--------|-------|
| R^2 | 0.999992 |
| V coefficient | 0.053 (vs BS 0.050) |
| Extra dV/dS term | -0.067 |
| Extra d2V/dS2 term | +1.905 |

SINDy correctly detects model misspecification: the activated spurious terms absorb jump-diffusion dynamics.

### Heston Stochastic Volatility

| Metric | Value |
|--------|-------|
| Linearity R^2 | 0.999999 |
| Slope | -0.5001 (true: -0.500) |
| Variance levels tested | 5 |

### Ablation Study

**Library expansion:**

| Library Size | True Terms Survive | False Positives |
|-------------|-------------------|-----------------|
| 5 terms | Yes | 2 |
| 8 terms | Yes | 2 |
| 11 terms | Yes | 5 |
| 14 terms | Yes | 7 |

**Library reduction:** S^2*d2V/dS2 causes the largest R^2 drop (0.0174) when removed.

### Real Market Data (Yahoo Finance)

| Ticker | Source | Options | Avg IV | R^2 |
|--------|--------|---------|--------|-----|
| SPY | live | 777 | 21.8% | 0.485 |
| QQQ | live | 511 | 25.3% | 0.671 |
| AAPL | live | 136 | 31.4% | 0.290 |
| MSFT | live | 212 | 35.5% | -0.076 |

Real-market R^2 values reflect volatility smile, discrete strikes, bid-ask noise, and non-constant parameters -- all systematic violations of constant-sigma Black-Scholes. The pipeline uses date-stamped caching (`real_chain_{ticker}_{YYYYMMDD}.csv`) with 24-hour expiry and falls back to mock data when yfinance is unavailable.

### Computation Costs

| Stage | Runtime |
|-------|---------|
| Full pipeline | 2,516s (42 min) |
| PINN call (5K epochs) | 487s |
| PINN put (10K epochs) | 986s |
| PINN v2 put (10K epochs) | 967s |
| Real data experiment | 24s |
| Baselines (clean + noisy) | 30s |
| SINDy discovery (call) | 0.03s |
| Noise-smoothing experiments | 4s |
| Ablation study | 0.6s |

## Installation

```bash
cd bs-pde-discovery
pip install -r requirements.txt
```

## Usage

```bash
# Run full 22-step pipeline (approx. 42 minutes)
python run_pipeline.py

# Run all 71 tests
pytest tests/ -v
```

## Project Structure

```
bs-pde-discovery/
  src/
    __init__.py                 # Package init
    utils.py                    # Seeds, logging, Timer with timing registry
    data_generation.py          # Analytical BS surfaces + noise + Merton pricing
    sindy_discovery.py          # PDE discovery, post-processing, correlation analysis
    pinn_validation.py          # PINN, PINNv2 (relative loss), regional error analysis
    greeks.py                   # Analytical and PINN-based Greeks
    robustness.py               # Noise, parameter, smoothing, grid resolution robustness
    diagnostics.py              # Overfitting, leakage, stability checks
    visualization.py            # All plotting functions (25+ functions)
    baselines.py                # Dense, Lasso, Ridge, symbolic regression
    extended_models.py          # Merton jump-diffusion, Heston variance slicing
    ablation.py                 # Library misspecification experiments
    real_data.py                # Yahoo Finance data with caching + mock fallback
  tests/
    test_data_generation.py     # 21 tests for BS formulas
    test_sindy.py               # 7 tests for SINDy discovery
    test_pinn.py                # 5 tests for PINN training
    test_greeks.py              # 4 tests for Greeks computation
    test_diagnostics.py         # 4 tests for diagnostic checks
    test_integration.py         # 5 tests (1 E2E + 4 import smoke tests)
    test_baselines.py           # 7 tests for baseline methods
    test_extended_models.py     # 7 tests for Merton/Heston
    test_ablation.py            # 6 tests for library ablation
    test_real_data.py           # 5 tests for real data pipeline
  outputs/
    figures/                    # 27 publication-quality PNG plots
    tables/                     # 23 CSV result tables
  run_pipeline.py               # Full 22-step experiment pipeline
  requirements.txt              # Python dependencies
  README.md                     # This file
  REPORT.md                     # Detailed technical report (workshop paper)
```

## Methodology

### SINDy (Sparse Identification of Nonlinear Dynamics)

SINDy works by computing numerical partial derivatives of observed data, building a library of candidate PDE terms, and using sparse regression to find which terms are active. The Sequential Thresholded Least Squares (STLSQ) algorithm iteratively removes small coefficients and re-solves until a sparse solution is found.

The full library contains 5 candidate terms: {V, dV/dS, d2V/dS2, S*dV/dS, S^2*d2V/dS^2}. The reduced library retains only the 3 terms that appear in the true Black-Scholes equation: {V, S*dV/dS, S^2*d2V/dS^2}. This reduction drops the condition number from 6.78e+04 to 44.2, eliminating multicollinearity while preserving R^2 = 0.999998.

Post-processing with a secondary threshold (0.1 * max(|coeff|)) cannot fix the full-library multicollinearity because the spurious d2V/dS2 coefficient (0.555) is the largest term -- thresholding removes the true terms instead. This demonstrates that domain-knowledge-based library design (the reduced library) is necessary, not just post-hoc filtering.

### Physics-Informed Neural Networks (PINNs)

PINNs are neural networks trained to satisfy both observed data and a PDE simultaneously. The loss function has three components: PDE residual loss, boundary condition loss, and data fitting loss on a training subset.

The PINN achieves strong results for calls (1.01% L2 error) and good ATM accuracy for puts (4.53% L2 error in the 0.8K-1.2K region), though the aggregate put error (47.9%) is dominated by the deep ITM region. A v2 PINN with relative loss and elevated boundary weight performed slightly worse (50.3%), indicating that the loss landscape conflict requires architectural solutions (domain decomposition, curriculum learning) rather than simple loss reweighting.

### Noise-Smoothing Analysis

The noise bottleneck is characterized through three complementary experiments:
- **Smoothing ablation**: 7 Savitzky-Golay configurations at 5% noise (best R^2 = 0.053)
- **Grid resolution vs noise**: 4 grid sizes from 30x30 to 200x200 (no improvement under noise)
- **Noise-smoothing matrix**: 4x4 cross of noise levels and smoothing settings (structure never recovered above 1% noise)

These results motivate integral-form methods (Messenger and Bortz, 2021; Fasel et al., 2022) that avoid explicit differentiation.

## Diagnostics and Safeguards

- **Data leakage prevention**: 60/20/20 train/val/test split with verified disjointness
- **Overfitting detection**: Validation loss monitoring with early stopping
- **Gradient pathology monitoring**: Per-component gradient norm tracking
- **SINDy bootstrap stability**: 20 bootstrap resamples confirm coefficient consistency
- **Numerical derivative quality**: Comparison against analytical derivatives (all < 0.11% error)
- **PINN extrapolation testing**: Error on extended domain S in [30, 170] (3.98x ratio)
- **Regional PINN error analysis**: ATM/ITM/OTM decomposition for put option accuracy
- **Post-processing diagnosis**: Secondary thresholding demonstrates multicollinearity severity
- **Library correlation matrix**: Pairwise correlations quantify multicollinearity structure
- **Put-call parity**: Verify C - P = S - K*exp(-r*tau) across the grid
- **PDE residual distribution**: Statistics at 50,000 random points
- **Multicollinearity diagnostics**: Condition number and pairwise correlation monitoring
- **Cross-ticker consistency check**: SINDy coefficients compared across 4 real market tickers
- **Timing registry**: All experiment runtimes recorded and exported

## Figures

All 27 figures are saved in `outputs/figures/`:

| Figure | Description |
|--------|-------------|
| `price_surfaces_3d.png` | 3D call and put price surfaces |
| `sindy_threshold_sweep.png` | R^2 and sparsity vs STLSQ threshold |
| `sindy_coefficient_comparison.png` | Discovered vs true coefficients |
| `pinn_vs_analytical_call.png` | PINN vs analytical call prices and error |
| `pinn_vs_analytical_put.png` | PINN vs analytical put prices and error |
| `pinn_training_loss.png` | Training loss curves with all components |
| `pinn_call_error_analysis.png` | Call PINN error heatmap with regional annotations |
| `pinn_put_error_analysis.png` | Put PINN error heatmap with ATM/ITM/OTM regions |
| `greeks_comparison.png` | PINN vs analytical Delta and Gamma heatmaps |
| `greeks_error_heatmap.png` | Greek absolute error distributions |
| `noise_robustness.png` | SINDy performance vs noise level |
| `noise_smoothing_matrix.png` | R^2 heatmap across noise x smoothing settings |
| `grid_resolution_vs_noise.png` | R^2 vs grid size (clean vs noisy) |
| `smoothing_bias_variance.png` | Smoothing bias and R^2 vs filter configuration |
| `parameter_generalization.png` | Coefficient errors across sigma/r combinations |
| `reduced_vs_full_library.png` | Full vs reduced library comparison |
| `data_split_visualization.png` | Train/val/test point distribution |
| `baseline_coefficient_comparison.png` | Baseline method coefficient comparison |
| `baseline_lasso_path.png` | Lasso regularization path |
| `heston_variance_slicing.png` | Heston variance-slicing linearity plot |
| `ablation_library_heatmap.png` | Library expansion coefficient heatmap |
| `ablation_condition_numbers.png` | Condition number vs library size |
| `real_iv_surface_spy.png` | SPY implied volatility surface |
| `real_iv_surface_qqq.png` | QQQ implied volatility surface |
| `real_iv_surface_aapl.png` | AAPL implied volatility surface |
| `real_iv_surface_msft.png` | MSFT implied volatility surface |
| `real_sindy_comparison.png` | Cross-ticker SINDy coefficient comparison |

## Tables

All 23 CSV tables are saved in `outputs/tables/`.

## Output Summary

| Category | Count |
|----------|-------|
| PNG figures | 27 |
| CSV tables | 23 |
| Tests (all passing) | 71 |
| Pipeline runtime | 2,516 seconds (42 min) |

## References

- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637-654.
- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1-2), 125-144.
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies*, 6(2), 327-343.
- Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *PNAS*, 113(15), 3932-3937.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686-707.
- de Silva, B., et al. (2020). PySINDy: A Python package for the sparse identification of nonlinear dynamical systems from data. *JOSS*, 5(49), 2104.
- Messenger, D. A. & Bortz, D. M. (2021). Weak SINDy: Galerkin-based data-driven model selection. *Multiscale Modeling and Simulation*, 19(3), 1474-1497.
- Messenger, D. A., et al. (2022). Learning mean-field equations from particle data using WSINDy. *Physica D*, 439, 133406.
- Fasel, U., et al. (2022). Ensemble-SINDy: Robust sparse model discovery in the low-data, high-noise limit. *Proc. Royal Society A*, 478(2260), 20210904.
- Dhiman, N. & Hu, J. (2023). Physics-informed neural networks for solving option pricing problems. *Quantitative Finance*, 23(7-8), 1079-1094.
- Hainaut, D. & Casas, I. (2024). Neural network methods for option pricing under stochastic volatility. *Journal of Computational Finance*, 27(3), 1-38.
- Sharma, A. & Verma, R. (2025). Machine learning in option pricing: A comprehensive survey. *Annual Review of Financial Economics*, 17, forthcoming.

## License

MIT
