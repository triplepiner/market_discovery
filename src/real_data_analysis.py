"""
Deep analysis of SINDy results from real market data.

Extracts financial meaning from discovered PDE coefficients: effective
volatility, drift rate, dividend yields, and compares coefficient patterns
against Merton jump-diffusion synthetic data to assess whether real
markets exhibit jump-like dynamics.
"""

import numpy as np
import pandas as pd
import warnings
import logging

from src.utils import setup_logging
from src.sindy_discovery import TERM_NAMES

logger = setup_logging(__name__)

# Known approximate dividend yields for common tickers (as of 2024-2025)
_KNOWN_DIVIDEND_YIELDS = {
    'SPY': 0.013,
    'QQQ': 0.005,
    'AAPL': 0.005,
    'MSFT': 0.007,
}


# ===================================================================
# Improvement 1 — Analyze discovered PDE from real data
# ===================================================================

def analyze_discovered_pde(sindy_result, S0, r, avg_iv, ticker):
    """
    Extract financial meaning from SINDy-discovered PDE on real data.

    Interprets the discovered coefficients through the lens of the
    Black-Scholes PDE::

        dV/dt = r*V + (-r)*S*dV/dS + (-0.5*sigma^2)*S^2*d2V/dS^2

    For real data, deviations from this form reveal:
    - Effective volatility (from the diffusion term S^2*d2V/dS^2)
    - Effective drift / dividend yield (from the convection term S*dV/dS)
    - Jump signatures (from spurious bare derivative terms dV/dS, d2V/dS2)

    Parameters
    ----------
    sindy_result : dict
        Output from ``discover_pde()``, must contain ``'discovered_coefficients'``.
    S0 : float
        Current stock price.
    r : float
        Risk-free rate.
    avg_iv : float
        Average implied volatility across the option chain.
    ticker : str
        Ticker symbol for labeling.

    Returns
    -------
    dict
        Keys include: ``ticker``, ``sigma_discovered``, ``sigma_ratio``,
        ``r_discovered``, ``q_implied``, ``jump_signature``,
        ``term_comparison`` (list of dicts), and plausibility flags.
    """
    coeffs = np.asarray(sindy_result['discovered_coefficients'], dtype=float)

    # Coefficient indices: [V, dV/dS, d2V/dS2, S*dV/dS, S2*d2V/dS2]
    c_V = coeffs[0]
    c_dVdS = coeffs[1]
    c_d2VdS2 = coeffs[2]
    c_SdVdS = coeffs[3]
    c_S2d2VdS2 = coeffs[4]

    # 1. EFFECTIVE VOLATILITY
    # BS: S^2*d2V/dS2 coefficient = -0.5*sigma^2, so sigma = sqrt(-2*coeff)
    sigma_discovered = np.nan
    if c_S2d2VdS2 < -1e-10:
        sigma_discovered = float(np.sqrt(-2.0 * c_S2d2VdS2))

    sigma_ratio = np.nan
    if not np.isnan(sigma_discovered) and avg_iv > 0:
        sigma_ratio = sigma_discovered / avg_iv

    # 2. EFFECTIVE DRIFT
    # BS: S*dV/dS coefficient = -r (no dividends) or -(r-q) (with dividends)
    r_discovered = -c_SdVdS
    q_implied = r - r_discovered  # dividend yield = r_fetched - r_discovered

    # Plausibility checks
    r_plausible = (-0.5 < r_discovered < 1.0)
    q_plausible = (-0.10 < q_implied < 0.20) and r_plausible
    sigma_plausible = (not np.isnan(sigma_discovered) and 0.01 < sigma_discovered < 2.0)

    # 3. SPURIOUS TERM ANALYSIS (jump signature)
    # In pure BS: bare dV/dS and d2V/dS2 are zero.
    # In Merton: d2V/dS2 is large and positive (~1.9).
    # Jump signature = ratio of |bare d2V/dS2| to |S2*d2V/dS2|.
    jump_signature = np.nan
    if abs(c_S2d2VdS2) > 1e-10:
        jump_signature = abs(c_d2VdS2) / abs(c_S2d2VdS2)

    # 4. TERM-BY-TERM COMPARISON TABLE
    bs_theory = np.array([r, 0.0, 0.0, -r, -0.5 * avg_iv ** 2])

    interpretations = [
        'Discounting rate',
        'Jump/skew signature (bare)',
        'Jump diffusion (bare)',
        'Drift (r or r-q)',
        'Diffusion (volatility)',
    ]

    term_comparison = []
    for i, (name, interp) in enumerate(zip(TERM_NAMES, interpretations)):
        term_comparison.append({
            'term': name,
            'bs_theory': float(bs_theory[i]),
            'real_discovered': float(coeffs[i]),
            'interpretation': interp,
        })

    result = {
        'ticker': ticker,
        'S0': S0,
        'r_fetched': r,
        'avg_iv': avg_iv,
        'discovered_coefficients': coeffs.tolist(),
        'sigma_discovered': float(sigma_discovered) if not np.isnan(sigma_discovered) else None,
        'sigma_ratio': float(sigma_ratio) if not np.isnan(sigma_ratio) else None,
        'r_discovered': float(r_discovered),
        'q_implied': float(q_implied),
        'r_plausible': r_plausible,
        'q_plausible': q_plausible,
        'sigma_plausible': sigma_plausible,
        'jump_signature': float(jump_signature) if not np.isnan(jump_signature) else None,
        'bs_theory_coefficients': bs_theory.tolist(),
        'term_comparison': term_comparison,
    }

    logger.info(
        "%s: sigma_disc=%s, r_disc=%.4f, q_impl=%.4f, "
        "jump_sig=%s, sigma_plaus=%s, q_plaus=%s",
        ticker,
        f"{sigma_discovered:.4f}" if not np.isnan(sigma_discovered) else "N/A",
        r_discovered, q_implied,
        f"{jump_signature:.2f}" if not np.isnan(jump_signature) else "N/A",
        sigma_plausible, q_plausible,
    )

    return result


# ===================================================================
# Dividend yield discovery
# ===================================================================

def dividend_yield_discovery(sindy_result, r_fetched, ticker):
    """
    Attempt to discover the dividend yield from PDE coefficients.

    In BS with continuous dividends::

        dV/dt + 0.5*sigma^2*S^2*d2V/dS^2 + (r-q)*S*dV/dS - r*V = 0

    So the S*dV/dS coefficient = ``-(r-q)``, meaning
    ``q = r_fetched - (-coefficient) = r_fetched + coefficient``.

    Parameters
    ----------
    sindy_result : dict
        Must contain ``'discovered_coefficients'``.
    r_fetched : float
        Risk-free rate from treasury data.
    ticker : str
        Used to look up known dividend yield for comparison.

    Returns
    -------
    dict
        Keys: ``q_implied``, ``q_actual``, ``agreement``, ``plausible``.
    """
    coeffs = np.asarray(sindy_result['discovered_coefficients'], dtype=float)
    c_SdVdS = coeffs[3]  # S*dV/dS coefficient

    # In dividend-adjusted BS: S*dV/dS coefficient = -(r-q)
    # So r_discovered = -c_SdVdS = r - q, hence q = r - r_discovered.
    r_discovered = -c_SdVdS
    q_implied = r_fetched - r_discovered

    # Try to get known dividend yield
    q_actual = None
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        q_raw = info.get('dividendYield', None)
        if q_raw is not None and 0 < q_raw < 0.20:
            q_actual = float(q_raw)
    except Exception:
        pass

    # Fallback to known approximations
    if q_actual is None:
        q_actual = _KNOWN_DIVIDEND_YIELDS.get(ticker, None)

    # Check agreement (within 1%)
    agreement = False
    plausible = (-0.05 < q_implied < 0.15)
    if q_actual is not None and plausible:
        agreement = abs(q_implied - q_actual) < 0.01

    result = {
        'ticker': ticker,
        'r_fetched': r_fetched,
        'r_discovered': float(r_discovered),
        'q_implied': float(q_implied),
        'q_actual': float(q_actual) if q_actual is not None else None,
        'agreement': agreement,
        'plausible': plausible,
    }

    logger.info(
        "%s dividend: q_implied=%.4f, q_actual=%s, agreement=%s, plausible=%s",
        ticker, q_implied,
        f"{q_actual:.4f}" if q_actual is not None else "N/A",
        agreement, plausible,
    )

    return result


# ===================================================================
# Cross-ticker correlation
# ===================================================================

def compute_vix_correlation(ticker_analyses):
    """
    Check whether BS deviation correlates with implied volatility level.

    Higher-IV tickers might show more deviation from BS. This tests
    whether the BS model fit degrades systematically with volatility.

    Parameters
    ----------
    ticker_analyses : dict
        Maps ticker -> analysis dict from :func:`analyze_discovered_pde`.

    Returns
    -------
    dict
        Spearman correlation, sigma ratio statistics, details.
    """
    tickers = []
    avg_ivs = []
    deviations = []
    sigma_ratios = []

    for ticker, res in ticker_analyses.items():
        tickers.append(ticker)
        avg_ivs.append(res['avg_iv'])

        disc = np.array(res['discovered_coefficients'])
        bs = np.array(res['bs_theory_coefficients'])
        deviations.append(float(np.linalg.norm(disc - bs)))

        sr = res.get('sigma_ratio')
        sigma_ratios.append(sr if sr is not None else np.nan)

    result = {
        'tickers': tickers,
        'avg_ivs': avg_ivs,
        'deviations': deviations,
        'sigma_ratios': sigma_ratios,
        'spearman_corr': None,
        'spearman_pvalue': None,
        'sigma_ratio_std': None,
        'sigma_ratio_varies': None,
    }

    if len(tickers) >= 3:
        try:
            from scipy import stats
            corr, pval = stats.spearmanr(avg_ivs, deviations)
            result['spearman_corr'] = float(corr)
            result['spearman_pvalue'] = float(pval)
        except Exception:
            pass

    valid_ratios = [sr for sr in sigma_ratios if sr is not None and not np.isnan(sr)]
    if len(valid_ratios) >= 2:
        result['sigma_ratio_std'] = float(np.std(valid_ratios))
        # If std > 0.1, ratios vary across tickers (different effective dynamics)
        result['sigma_ratio_varies'] = float(np.std(valid_ratios)) > 0.1

    return result


# ===================================================================
# Improvement 2 — Merton bridge
# ===================================================================

def _cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _normalize_fingerprint(coeffs):
    """Normalize coefficient vector so max |value| = 1."""
    coeffs = np.asarray(coeffs, dtype=float)
    max_abs = np.max(np.abs(coeffs))
    if max_abs < 1e-10:
        return coeffs
    return coeffs / max_abs


def merton_real_data_bridge(merton_result, real_results_dict):
    """
    Compare SINDy coefficient fingerprints across BS, Merton, and real data.

    Computes cosine similarity between each ticker's discovered coefficient
    vector and the Merton / BS reference vectors. This provides suggestive
    (not conclusive) evidence about whether real markets exhibit jump-like
    dynamics similar to the Merton model.

    Parameters
    ----------
    merton_result : dict
        From :func:`run_merton_experiment`, containing
        ``'discovered_coefficients'`` and ``'true_bs_coefficients'``.
    real_results_dict : dict
        Maps ticker -> result dict (from ``run_real_data_experiment``),
        each containing ``'sindy_result'`` with ``'discovered_coefficients'``.

    Returns
    -------
    dict
        Keys: ``bridge_df`` (DataFrame), ``summary`` (dict).
    """
    merton_coeffs = np.array(merton_result['discovered_coefficients'])
    bs_coeffs = np.array(merton_result['true_bs_coefficients'])

    merton_fp = _normalize_fingerprint(merton_coeffs)
    bs_fp = _normalize_fingerprint(bs_coeffs)

    # Merton reference values
    merton_lambda = merton_result.get('params', {}).get('lam', 0.1)
    merton_d2VdS2 = merton_coeffs[2]  # bare d2V/dS2 from Merton synthetic

    rows = []
    for ticker, res in real_results_dict.items():
        sindy = res.get('sindy_result', {})
        disc_coeffs = np.asarray(
            sindy.get('discovered_coefficients', np.zeros(5)), dtype=float
        )

        real_fp = _normalize_fingerprint(disc_coeffs)

        cos_bs = _cosine_similarity(real_fp, bs_fp)
        cos_merton = _cosine_similarity(real_fp, merton_fp)
        closer_to = 'Merton' if cos_merton > cos_bs else 'BS'

        # Rough jump intensity estimation
        real_d2VdS2 = disc_coeffs[2]
        est_jump_intensity = np.nan
        if abs(merton_d2VdS2) > 1e-10:
            est_jump_intensity = abs(real_d2VdS2) / abs(merton_d2VdS2) * merton_lambda

        rows.append({
            'ticker': ticker,
            'cos_sim_bs': cos_bs,
            'cos_sim_merton': cos_merton,
            'closer_to': closer_to,
            'jump_intensity_est': est_jump_intensity,
            'real_d2VdS2': real_d2VdS2,
            'merton_d2VdS2': merton_d2VdS2,
        })

    bridge_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    # Aggregate findings
    n_closer_merton = sum(1 for r in rows if r['closer_to'] == 'Merton')
    n_closer_bs = sum(1 for r in rows if r['closer_to'] == 'BS')

    # Check: do index options show more jump risk than single stocks?
    index_tickers = {'SPY', 'QQQ'}
    index_jump = [
        r['jump_intensity_est'] for r in rows
        if r['ticker'] in index_tickers and not np.isnan(r['jump_intensity_est'])
    ]
    stock_jump = [
        r['jump_intensity_est'] for r in rows
        if r['ticker'] not in index_tickers and not np.isnan(r['jump_intensity_est'])
    ]

    index_more_jumpy = None
    if index_jump and stock_jump:
        index_more_jumpy = float(np.mean(index_jump)) > float(np.mean(stock_jump))

    summary = {
        'n_closer_merton': n_closer_merton,
        'n_closer_bs': n_closer_bs,
        'index_more_jumpy': index_more_jumpy,
        'merton_fingerprint': merton_fp.tolist(),
        'bs_fingerprint': bs_fp.tolist(),
    }

    logger.info(
        "Merton bridge: %d tickers closer to Merton, %d to BS",
        n_closer_merton, n_closer_bs,
    )

    return {'bridge_df': bridge_df, 'summary': summary}


# ===================================================================
# Improvement 3 — IV regime analysis
# ===================================================================

def iv_regime_analysis(option_data, S0, r, ticker):
    """
    Split the option chain by moneyness and maturity, run SINDy on each slice.

    Checks whether the discovered effective volatility varies across
    regimes — revealing volatility term structure and smile/skew effects
    from PDE coefficients without assuming any smile model.

    If a slice has fewer than 30 options or fewer than 3 expirations,
    it is skipped and flagged as "low confidence."

    Parameters
    ----------
    option_data : dict
        From :func:`fetch_option_data`, must include ``'option_df'``.
    S0 : float
        Current stock price.
    r : float
        Risk-free rate.
    ticker : str
        Ticker symbol.

    Returns
    -------
    dict
        Keys: ``maturity_regimes``, ``moneyness_regimes``,
        ``skew_detected``, ``term_structure_shape``, ``data_source``.
    """
    from src.real_data import (
        construct_smooth_surface, run_sindy_on_real_data, _dataframe_to_result,
    )

    df = option_data['option_df'].copy()
    df['moneyness'] = df['strike'] / S0  # K/S0 convention

    MIN_OPTIONS = 30
    MIN_EXPIRATIONS = 3

    def _analyze_slice(slice_df, regime_name):
        """Run SINDy on a slice and extract sigma."""
        n_opt = len(slice_df)

        if n_opt < MIN_OPTIONS:
            logger.warning(
                "%s %s: only %d options (need %d), skipping",
                ticker, regime_name, n_opt, MIN_OPTIONS,
            )
            avg_iv_slice = _safe_avg_iv(slice_df)
            return _skip_result(regime_name, n_opt, avg_iv_slice,
                                f'insufficient data ({n_opt} < {MIN_OPTIONS})')

        n_exps = slice_df['expiration'].nunique()
        if n_exps < MIN_EXPIRATIONS:
            logger.warning(
                "%s %s: only %d expirations (need %d), skipping",
                ticker, regime_name, n_exps, MIN_EXPIRATIONS,
            )
            avg_iv_slice = _safe_avg_iv(slice_df)
            return _skip_result(regime_name, n_opt, avg_iv_slice,
                                f'insufficient expirations ({n_exps} < {MIN_EXPIRATIONS})')

        # Build option_data-like dict for this slice
        try:
            slice_odata = _dataframe_to_result(
                slice_df, ticker,
                data_source=option_data.get('data_source', 'unknown'),
            )
            surface = construct_smooth_surface(slice_odata, n_K=30)
            sindy_out = run_sindy_on_real_data(surface, slice_odata)

            sigma_disc = sindy_out.get('sigma_effective', np.nan)
            r2 = sindy_out['sindy_result'].get('r2_score', np.nan)
            avg_iv_slice = _safe_avg_iv(slice_df)

            ratio = np.nan
            if (not np.isnan(sigma_disc) and
                    not np.isnan(avg_iv_slice) and avg_iv_slice > 0):
                ratio = sigma_disc / avg_iv_slice

            return {
                'regime': regime_name,
                'n_options': n_opt,
                'sigma_discovered': _nan_to_none(sigma_disc),
                'sigma_market': _nan_to_none(avg_iv_slice),
                'ratio': _nan_to_none(ratio),
                'r2': _nan_to_none(r2),
                'skipped': False,
                'reason': None,
            }
        except Exception as e:
            logger.warning("%s %s: analysis failed: %s", ticker, regime_name, e)
            avg_iv_slice = _safe_avg_iv(slice_df)
            return _skip_result(regime_name, n_opt, avg_iv_slice, str(e))

    # SPLIT 1: By maturity
    maturity_regimes = []
    mat_bins = [
        ('Short (<2mo)', df['tau'] < 0.15),
        ('Medium (2-6mo)', (df['tau'] >= 0.15) & (df['tau'] < 0.5)),
        ('Long (>6mo)', df['tau'] >= 0.5),
    ]
    for name, mask in mat_bins:
        maturity_regimes.append(_analyze_slice(df[mask].copy(), name))

    # SPLIT 2: By moneyness (K/S0)
    moneyness_regimes = []
    mon_bins = [
        ('OTM puts (<0.95)', df['moneyness'] < 0.95),
        ('ATM (0.95-1.05)', (df['moneyness'] >= 0.95) & (df['moneyness'] <= 1.05)),
        ('OTM calls (>1.05)', df['moneyness'] > 1.05),
    ]
    for name, mask in mon_bins:
        moneyness_regimes.append(_analyze_slice(df[mask].copy(), name))

    # Detect skew (OTM put sigma > ATM sigma)
    skew_detected = _detect_skew(moneyness_regimes)

    # Detect term structure shape
    term_structure_shape = _detect_term_structure(maturity_regimes)

    logger.info(
        "%s regime analysis: mat_slices=%d, mon_slices=%d, skew=%s, ts=%s",
        ticker,
        sum(1 for r in maturity_regimes if not r['skipped']),
        sum(1 for r in moneyness_regimes if not r['skipped']),
        skew_detected, term_structure_shape,
    )

    return {
        'ticker': ticker,
        'maturity_regimes': maturity_regimes,
        'moneyness_regimes': moneyness_regimes,
        'skew_detected': skew_detected,
        'term_structure_shape': term_structure_shape,
        'data_source': option_data.get('data_source', 'unknown'),
    }


# ===================================================================
# Internal helpers
# ===================================================================

def _safe_avg_iv(df):
    """Compute average IV from a DataFrame slice, ignoring bad values."""
    ivs = df['implied_vol'].dropna()
    ivs = ivs[(ivs > 0) & (ivs < 3.0)]
    return float(ivs.mean()) if len(ivs) > 0 else np.nan


def _nan_to_none(v):
    """Convert NaN to None for clean JSON serialization."""
    if v is None:
        return None
    try:
        if np.isnan(v):
            return None
    except (TypeError, ValueError):
        pass
    return float(v)


def _skip_result(regime_name, n_options, sigma_market, reason):
    """Return a standardized skip result dict."""
    return {
        'regime': regime_name,
        'n_options': n_options,
        'sigma_discovered': None,
        'sigma_market': _nan_to_none(sigma_market),
        'ratio': None,
        'r2': None,
        'skipped': True,
        'reason': reason,
    }


def _detect_skew(moneyness_regimes):
    """Check if OTM put implied vol > ATM implied vol (volatility skew)."""
    otm_put = next(
        (r for r in moneyness_regimes if 'OTM puts' in r['regime'] and not r['skipped']),
        None,
    )
    atm = next(
        (r for r in moneyness_regimes if 'ATM' in r['regime'] and not r['skipped']),
        None,
    )
    if otm_put and atm:
        otm_sigma = otm_put.get('sigma_discovered') or otm_put.get('sigma_market')
        atm_sigma = atm.get('sigma_discovered') or atm.get('sigma_market')
        if otm_sigma is not None and atm_sigma is not None:
            return otm_sigma > atm_sigma
    return None


def _detect_term_structure(maturity_regimes):
    """Determine term structure shape from maturity regime sigmas."""
    active = [r for r in maturity_regimes if not r['skipped']]
    if len(active) < 2:
        return None

    sigmas = []
    for r in active:
        s = r.get('sigma_discovered') or r.get('sigma_market')
        if s is not None:
            sigmas.append(s)

    if len(sigmas) < 2:
        return None

    if sigmas[-1] > sigmas[0] * 1.05:
        return 'upward'
    elif sigmas[-1] < sigmas[0] * 0.95:
        return 'downward'
    else:
        return 'flat'
