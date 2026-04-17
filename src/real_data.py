"""
Fetch real option data from Yahoo Finance and run SINDy PDE discovery.

This module bridges market data and the PDE discovery pipeline. It fetches
live option chains, constructs smooth implied-volatility surfaces, and
applies SINDy to discover the governing PDE from real market observations.

If network access is unavailable, all operations fall back gracefully to
synthetic mock data that mimics realistic market characteristics.
"""

import numpy as np
import pandas as pd
import os
import warnings
import logging
from datetime import datetime, timezone

from src.utils import set_all_seeds, setup_logging, Timer
from src.sindy_discovery import discover_pde, TERM_NAMES
from src.data_generation import bs_call_price

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Default approximate spot prices and volatilities for common tickers
# ---------------------------------------------------------------------------
_TICKER_DEFAULTS = {
    'SPY':  {'S0': 520.0, 'sigma': 0.18},
    'QQQ':  {'S0': 480.0, 'sigma': 0.22},
    'AAPL': {'S0': 210.0, 'sigma': 0.28},
    'MSFT': {'S0': 430.0, 'sigma': 0.20},
}

_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'outputs', 'tables',
)


# ===================================================================
# Misspecification diagnostics
# ===================================================================

def compute_bs_deviation_score(discovered_coefficients, r=0.05, sigma=0.2):
    """Compute normalised distance between discovered and theoretical BS coefficients.

    The Black-Scholes PDE for dV/dt has the form::

        dV/dt = r*V + 0*dV/dS + 0*d2V/dS2 + (-r)*S*dV/dS + (-0.5*sigma^2)*S2*d2V/dS2

    Parameters
    ----------
    discovered_coefficients : array-like, shape (5,)
        Coefficients in the order of ``TERM_NAMES``:
        ``[V, dV/dS, d2V/dS2, S*dV/dS, S2*d2V/dS2]``.
    r : float
        Risk-free rate used for the theoretical benchmark.
    sigma : float
        Volatility used for the theoretical benchmark.

    Returns
    -------
    float
        Relative L2 deviation score.  Zero means exact agreement with BS.
    """
    disc = np.asarray(discovered_coefficients, dtype=float)
    true = np.array([r, 0.0, 0.0, -r, -0.5 * sigma ** 2])
    return float(np.linalg.norm(disc - true) / (np.linalg.norm(true) + 1e-10))


def run_cross_method_real_data(V, S_grid, t_grid, ticker, sigma_eff, r=0.05):
    """Run standard, neural, and weak SINDy on a real-data surface.

    Each method is executed independently; failures are caught so that the
    remaining methods still produce results.

    Parameters
    ----------
    V : ndarray, shape (n_S, n_t)
        Price surface (space axis 0, time axis 1, time ascending).
    S_grid : ndarray
        Spatial (strike) grid.
    t_grid : ndarray
        Calendar-time grid (ascending).
    ticker : str
        Ticker symbol (used only for logging).
    sigma_eff : float
        Effective volatility estimate, passed as ``true_sigma``.
    r : float
        Risk-free rate, passed as ``true_r``.

    Returns
    -------
    dict
        Keys ``'standard'``, ``'neural'``, ``'weak'``, each containing a
        SINDy result dict or *None* if that method failed.
    """
    results = {'standard': None, 'neural': None, 'weak': None}

    T_max = float(t_grid.max())
    common_kw = dict(
        true_sigma=sigma_eff,
        true_r=r,
        smooth=True,
        K=float(np.median(S_grid)),
        T=T_max,
        option_type='call',
    )

    # --- standard SINDy (already imported at module level) ---
    try:
        results['standard'] = discover_pde(V, S_grid, t_grid, **common_kw)
        logger.info("%s cross-method standard: R2=%.6f", ticker,
                    results['standard']['r2_score'])
    except Exception as exc:
        logger.warning("%s cross-method standard failed: %s", ticker, exc)

    # --- neural SINDy (lazy import) ---
    try:
        from src.neural_derivatives import discover_pde_neural  # noqa: F811
        results['neural'] = discover_pde_neural(V, S_grid, t_grid, **common_kw)
        logger.info("%s cross-method neural: R2=%.6f", ticker,
                    results['neural']['r2_score'])
    except ImportError:
        logger.info("%s cross-method neural: module not available", ticker)
    except Exception as exc:
        logger.warning("%s cross-method neural failed: %s", ticker, exc)

    # --- weak SINDy (lazy import) ---
    try:
        from src.weak_sindy import discover_pde_weak  # noqa: F811
        results['weak'] = discover_pde_weak(V, S_grid, t_grid, **common_kw)
        logger.info("%s cross-method weak: R2=%.6f", ticker,
                    results['weak']['r2_score'])
    except ImportError:
        logger.info("%s cross-method weak: module not available", ticker)
    except Exception as exc:
        logger.warning("%s cross-method weak failed: %s", ticker, exc)

    return results


# ===================================================================
# Public API
# ===================================================================

def fetch_option_data(ticker_symbol='SPY', cache_dir=None):
    """Fetch option chain data from Yahoo Finance.

    Retrieves calls for all available expirations, filters for liquid
    contracts near the money, and returns a standardised dictionary.

    Parameters
    ----------
    ticker_symbol : str
        Equity ticker (e.g. ``'SPY'``).
    cache_dir : str or None
        Directory for CSV cache.  Defaults to ``outputs/tables/`` inside
        the project root.

    Returns
    -------
    dict
        Keys: ``ticker``, ``S0``, ``r``, ``strikes``, ``expirations``,
        ``tau``, ``mid_prices``, ``implied_vols``, ``option_df``,
        ``data_source``, ``n_options``, ``bid_ask_spread_pct``,
        ``n_expirations_raw``, ``n_expirations_filtered``,
        ``n_contracts_raw``, ``n_contracts_filtered``,
        ``strike_range``, ``tau_range``, ``iv_range``.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR

    # ------------------------------------------------------------------
    # 1. Try loading from cache (date-stamped, < 24 hours old)
    # ------------------------------------------------------------------
    today_str = datetime.now(timezone.utc).strftime('%Y%m%d')
    cache_path = os.path.join(
        cache_dir, f'real_chain_{ticker_symbol}_{today_str}.csv'
    )
    if os.path.isfile(cache_path):
        try:
            mtime = os.path.getmtime(cache_path)
            age_hours = (datetime.now(timezone.utc).timestamp() - mtime) / 3600.0
            if age_hours < 24:
                df = pd.read_csv(cache_path)
                if len(df) >= 5:
                    logger.info(
                        "Loaded cached data for %s from %s (%d rows, %.1fh old)",
                        ticker_symbol, cache_path, len(df), age_hours,
                    )
                    return _dataframe_to_result(
                        df, ticker_symbol, data_source='cached'
                    )
        except Exception as exc:
            logger.warning("Cache read failed for %s: %s", ticker_symbol, exc)

    # ------------------------------------------------------------------
    # 2. Try live fetch via yfinance
    # ------------------------------------------------------------------
    try:
        import yfinance as yf  # noqa: F811
    except ImportError:
        logger.warning("yfinance not installed -- falling back to mock data.")
        return _generate_mock_data(ticker_symbol)

    try:
        ticker = yf.Ticker(ticker_symbol)

        # Current stock price
        hist = ticker.history(period='1d')
        if hist.empty:
            raise ValueError(f"No price history for {ticker_symbol}")
        S0 = float(hist['Close'].iloc[-1])

        # Risk-free rate: try ^IRX first, then ^TNX, then fallback
        r = _fetch_risk_free_rate(yf)

        # Expiration dates
        expirations = ticker.options
        if not expirations or len(expirations) == 0:
            raise ValueError(f"No option expirations for {ticker_symbol}")

        n_expirations_raw = len(expirations)

        rows = []
        n_contracts_raw = 0
        expirations_with_data = set()
        today = pd.Timestamp.now().normalize()

        for exp_date in expirations:
            try:
                chain = ticker.option_chain(exp_date)
            except Exception:
                continue
            calls = chain.calls
            if calls is None or calls.empty:
                continue

            n_contracts_raw += len(calls)

            exp_ts = pd.Timestamp(exp_date)
            tau_years = max((exp_ts - today).days, 1) / 365.25

            # Filter tau range: >= 7/365.25 and <= 2.0
            if tau_years < 7 / 365.25 or tau_years > 2.0:
                continue

            # Liquidity filters
            sub = calls.copy()
            sub = sub[sub['bid'].fillna(0) > 0]
            sub = sub[sub['ask'].fillna(0) > 0]
            sub = sub[sub['volume'].fillna(0) >= 50]
            sub = sub[sub['openInterest'].fillna(0) >= 100]

            # Moneyness filter
            moneyness = S0 / sub['strike']
            sub = sub[(moneyness >= 0.8) & (moneyness <= 1.2)]

            for _, row in sub.iterrows():
                mid = (row['bid'] + row['ask']) / 2.0
                iv = row.get('impliedVolatility', np.nan)

                # IV filter: not NaN, > 0, < 3.0
                if pd.isna(iv) or iv <= 0 or iv >= 3.0:
                    continue

                expirations_with_data.add(exp_date)
                rows.append({
                    'strike': row['strike'],
                    'expiration': exp_date,
                    'tau': tau_years,
                    'bid': row['bid'],
                    'ask': row['ask'],
                    'mid_price': mid,
                    'implied_vol': iv,
                    'volume': row.get('volume', 0),
                    'openInterest': row.get('openInterest', 0),
                    'S0': S0,
                    'r': r,
                })

        if len(rows) < 5:
            raise ValueError(
                f"Only {len(rows)} options passed filters for {ticker_symbol}"
            )

        df = pd.DataFrame(rows)

        n_contracts_filtered = len(df)
        n_expirations_filtered = len(expirations_with_data)

        # Cache to disk
        try:
            os.makedirs(cache_dir, exist_ok=True)
            df.to_csv(cache_path, index=False)
            logger.info("Cached %d options to %s", len(df), cache_path)
        except Exception as exc_cache:
            logger.warning("Failed to cache data: %s", exc_cache)

        logger.info(
            "Fetched %d options for %s (S0=%.2f, r=%.4f)",
            len(df), ticker_symbol, S0, r,
        )

        result = _dataframe_to_result(df, ticker_symbol, data_source='live')
        result['n_expirations_raw'] = n_expirations_raw
        result['n_expirations_filtered'] = n_expirations_filtered
        result['n_contracts_raw'] = n_contracts_raw
        result['n_contracts_filtered'] = n_contracts_filtered
        return result

    except Exception as exc:
        logger.warning(
            "Live fetch failed for %s (%s) -- falling back to mock data.",
            ticker_symbol, exc,
        )
        return _generate_mock_data(ticker_symbol)


def _fetch_risk_free_rate(yf):
    """Attempt to fetch the risk-free rate from treasury tickers.

    Tries ^IRX (13-week T-bill) first, then ^TNX (10-year note),
    with a final fallback to 0.045.
    """
    # Try ^IRX (13-week T-bill, quoted in percent)
    try:
        irx = yf.Ticker("^IRX").history(period='1d')
        if not irx.empty:
            r = float(irx['Close'].iloc[-1]) / 100.0
            if 0 < r <= 0.20:
                return r
            logger.warning("IRX rate %.4f looks suspicious", r)
    except Exception as exc_irx:
        logger.warning("IRX fetch failed (%s)", exc_irx)

    # Try ^TNX (10-year Treasury note, quoted in percent)
    try:
        tnx = yf.Ticker("^TNX").history(period='1d')
        if not tnx.empty:
            r = float(tnx['Close'].iloc[-1]) / 100.0
            if 0 < r <= 0.20:
                logger.info("Using ^TNX rate: %.4f", r)
                return r
            logger.warning("TNX rate %.4f looks suspicious", r)
    except Exception as exc_tnx:
        logger.warning("TNX fetch failed (%s)", exc_tnx)

    logger.warning("All rate fetches failed -- using r=0.045")
    return 0.045


def _generate_mock_data(ticker_symbol='SPY'):
    """Generate synthetic mock data that mimics real market characteristics.

    Produces a realistic-looking option chain with multiple expirations,
    a volatility smile, and noisy bid/ask spreads.

    Parameters
    ----------
    ticker_symbol : str
        Ticker symbol (used to select approximate spot price / vol).

    Returns
    -------
    dict
        Same structure as :func:`fetch_option_data` with
        ``data_source='mock'``.
    """
    set_all_seeds(42)
    logger.warning("Using MOCK data for %s", ticker_symbol)

    defaults = _TICKER_DEFAULTS.get(
        ticker_symbol, {'S0': 400.0, 'sigma': 0.22}
    )
    S0 = defaults['S0']
    base_sigma = defaults['sigma']
    r = 0.045

    # 8 expirations: ~2 weeks to ~6 months
    tau_days = np.array([14, 28, 42, 60, 90, 120, 150, 180])
    taus = tau_days / 365.25

    rows = []
    for tau_val, days in zip(taus, tau_days):
        n_strikes = np.random.randint(15, 21)
        strikes = np.linspace(S0 * 0.8, S0 * 1.2, n_strikes)

        for K in strikes:
            moneyness = K / S0
            # Simple parabolic smile
            iv = base_sigma + 0.03 * (moneyness - 1.0) ** 2
            price = float(bs_call_price(S0, K, r, iv, tau_val))

            # Add 2-3% Gaussian noise
            noise_frac = np.random.uniform(0.02, 0.03)
            price *= (1.0 + noise_frac * np.random.randn())
            price = max(price, 0.01)

            # Random bid-ask spread: 1-5% of mid
            spread_frac = np.random.uniform(0.01, 0.05)
            half_spread = price * spread_frac / 2.0
            bid = max(price - half_spread, 0.01)
            ask = price + half_spread

            exp_date = (
                pd.Timestamp.now().normalize()
                + pd.Timedelta(days=int(days))
            ).strftime('%Y-%m-%d')

            rows.append({
                'strike': K,
                'expiration': exp_date,
                'tau': tau_val,
                'bid': bid,
                'ask': ask,
                'mid_price': price,
                'implied_vol': iv,
                'volume': np.random.randint(100, 5000),
                'openInterest': np.random.randint(500, 20000),
                'S0': S0,
                'r': r,
            })

    df = pd.DataFrame(rows)
    logger.info(
        "Generated %d mock options for %s (S0=%.2f, sigma=%.2f)",
        len(df), ticker_symbol, S0, base_sigma,
    )
    return _dataframe_to_result(df, ticker_symbol, data_source='mock')


def construct_smooth_surface(option_data, n_K=40, n_tau=None):
    """Construct a smooth price surface from discrete option data.

    Interpolates implied volatilities onto a regular grid, then
    recomputes Black-Scholes prices so the resulting surface is smooth
    enough for numerical differentiation.

    Parameters
    ----------
    option_data : dict
        Output of :func:`fetch_option_data`.
    n_K : int
        Number of strike grid points.
    n_tau : int or None
        Number of maturity grid points.  When *None* the unique
        expirations in the data are used directly (minimum 3 required).

    Returns
    -------
    dict
        Keys: ``V_surface``, ``K_grid``, ``tau_grid``, ``iv_surface``,
        ``S0``, ``r``, ``n_valid_points``.

    Raises
    ------
    ValueError
        If the data contains fewer than 3 usable expirations or strikes.
    """
    from scipy.interpolate import griddata

    S0 = option_data['S0']
    r = option_data['r']
    df = option_data['option_df'].copy()

    # ------------------------------------------------------------------
    # Clean implied vols
    # ------------------------------------------------------------------
    iv = df['implied_vol'].values.astype(float)
    valid = np.isfinite(iv) & (iv > 0) & (iv <= 2.0)
    df = df.loc[valid].copy()

    if len(df) < 10:
        raise ValueError(
            f"Too few valid data points ({len(df)}) after IV filtering."
        )

    strikes = df['strike'].values
    taus = df['tau'].values
    ivs = df['implied_vol'].values

    # ------------------------------------------------------------------
    # Build regular grid
    # ------------------------------------------------------------------
    K_min, K_max = strikes.min(), strikes.max()
    if K_max - K_min < 1e-6:
        raise ValueError("Strike range too narrow to build a surface.")
    K_grid = np.linspace(K_min, K_max, n_K)

    unique_taus = np.sort(np.unique(np.round(taus, 6)))
    if len(unique_taus) < 3:
        raise ValueError(
            f"Need at least 3 distinct expirations, got {len(unique_taus)}."
        )
    if n_tau is not None and n_tau >= 3:
        tau_grid = np.linspace(unique_taus.min(), unique_taus.max(), n_tau)
    else:
        tau_grid = unique_taus

    KK, TT = np.meshgrid(K_grid, tau_grid, indexing='ij')  # (n_K, n_tau)

    # ------------------------------------------------------------------
    # Interpolate IV -- use linear for robustness with sparse real data
    # ------------------------------------------------------------------
    points = np.column_stack([strikes, taus])
    try:
        iv_surface = griddata(points, ivs, (KK, TT), method='linear')
    except Exception:
        logger.warning("Linear interpolation failed -- falling back to nearest.")
        iv_surface = griddata(points, ivs, (KK, TT), method='nearest')

    # Fill remaining NaN with nearest neighbour
    nan_mask = np.isnan(iv_surface)
    if np.any(nan_mask):
        iv_nearest = griddata(points, ivs, (KK, TT), method='nearest')
        iv_surface[nan_mask] = iv_nearest[nan_mask]

    # Clamp to sensible range
    iv_surface = np.clip(iv_surface, 0.01, 2.0)

    n_valid = int(np.sum(np.isfinite(iv_surface)))

    # ------------------------------------------------------------------
    # Recompute BS prices on the regular grid
    # ------------------------------------------------------------------
    V_surface = np.zeros_like(KK)
    for i in range(KK.shape[0]):
        for j in range(KK.shape[1]):
            K_val = KK[i, j]
            tau_val = TT[i, j]
            sigma_val = iv_surface[i, j]
            if tau_val <= 0 or sigma_val <= 0:
                V_surface[i, j] = max(S0 - K_val, 0.0)
            else:
                V_surface[i, j] = float(
                    bs_call_price(S0, K_val, r, sigma_val, tau_val)
                )

    logger.info(
        "Smooth surface: %d x %d (K x tau), %d valid IV points",
        n_K, len(tau_grid), n_valid,
    )

    return {
        'V_surface': V_surface,
        'K_grid': K_grid,
        'tau_grid': tau_grid,
        'iv_surface': iv_surface,
        'S0': S0,
        'r': r,
        'n_valid_points': n_valid,
    }


def run_sindy_on_real_data(surface_data, option_data):
    """Run SINDy PDE discovery on the constructed real-data surface.

    Uses stronger smoothing and wider boundary trimming than the
    synthetic-data pipeline to cope with market noise.

    Parameters
    ----------
    surface_data : dict
        Output of :func:`construct_smooth_surface`.
    option_data : dict
        Output of :func:`fetch_option_data`.

    Returns
    -------
    dict
        Keys: ``sindy_result``, ``sigma_effective``, ``avg_implied_vol``,
        ``ticker``, ``data_source``.
    """
    V = surface_data['V_surface']
    K_grid = surface_data['K_grid']
    tau_grid = surface_data['tau_grid']
    r = option_data['r']

    avg_iv = float(np.nanmean(option_data['implied_vols']))
    if np.isnan(avg_iv) or avg_iv <= 0:
        avg_iv = 0.20
        logger.warning("Average implied vol invalid -- defaulting to 0.20")

    # Choose trim size based on grid dimensions
    min_dim = min(V.shape[0], V.shape[1])
    if min_dim > 25:
        trim = 10
    elif min_dim > 15:
        trim = 5
    else:
        trim = max(min_dim // 4, 1)

    # discover_pde expects (n_S, n_t) with S_grid and t_grid.
    # We pass K_grid as the spatial variable and tau_grid as the time
    # variable.  T is set to tau_grid.max() so that the internal
    # conversion t -> tau works correctly.
    T_max = float(tau_grid.max())

    # Convert tau_grid to calendar-time grid: t = T_max - tau
    # discover_pde differentiates with respect to t internally (axis=1).
    # V must be arranged so that axis-0 is "space" (K) and axis-1 is
    # "time" (t ascending).
    t_grid = T_max - tau_grid[::-1]  # ascending calendar time
    V_t = V[:, ::-1]                 # flip tau-axis to match t ascending

    with Timer(f"SINDy on real data ({option_data['ticker']})"):
        sindy_result = discover_pde(
            V_t,
            K_grid,
            t_grid,
            true_sigma=avg_iv,
            true_r=r,
            smooth=True,
            K=float(np.median(K_grid)),
            T=T_max,
            option_type='call',
            savgol_window=11,
            savgol_poly=5,
            trim=trim,
        )

    # Effective sigma implied by discovered diffusion coefficient
    # BS PDE: dV/dt = r*V - r*S*dV/dS - 0.5*sigma^2*S^2*d2V/dS2
    # So coeff of S^2*d2V/dS2 should be -0.5*sigma^2
    sigma_effective = np.nan
    coeffs = sindy_result['discovered_coefficients']
    idx_S2d2V = TERM_NAMES.index('S2*d2V/dS2')
    coeff_diffusion = coeffs[idx_S2d2V]
    if coeff_diffusion < 0:
        sigma_effective = float(np.sqrt(-2.0 * coeff_diffusion))

    logger.info(
        "%s: sigma_eff=%.4f, avg_iv=%.4f, R2=%.6f, active=%s",
        option_data['ticker'],
        sigma_effective,
        avg_iv,
        sindy_result['r2_score'],
        sindy_result['active_terms'],
    )

    return {
        'sindy_result': sindy_result,
        'sigma_effective': sigma_effective,
        'avg_implied_vol': avg_iv,
        'ticker': option_data['ticker'],
        'data_source': option_data['data_source'],
    }


def run_real_data_experiment(tickers=None, cache_dir=None):
    """Run the full real-data experiment across multiple tickers.

    For each ticker the pipeline fetches data, constructs a smooth
    surface, runs SINDy, and collects results.

    Parameters
    ----------
    tickers : list of str or None
        Defaults to ``['SPY', 'QQQ', 'AAPL', 'MSFT']``.
    cache_dir : str or None
        Passed through to :func:`fetch_option_data`.

    Returns
    -------
    dict
        Keys: ``per_ticker_results``, ``cross_ticker_consistency``,
        ``summary_df``.
    """
    if tickers is None:
        tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT']

    per_ticker = {}
    summary_rows = []

    for ticker_symbol in tickers:
        logger.info("=" * 60)
        logger.info("Processing ticker: %s", ticker_symbol)
        logger.info("=" * 60)

        try:
            # 1. Fetch data
            option_data = fetch_option_data(ticker_symbol, cache_dir=cache_dir)

            # 2. Construct surface
            try:
                surface = construct_smooth_surface(option_data)
            except ValueError as ve:
                logger.error(
                    "Surface construction failed for %s: %s", ticker_symbol, ve
                )
                continue

            # 3. Run SINDy
            result = run_sindy_on_real_data(surface, option_data)

            # Store option_data and surface_data alongside SINDy result
            result['option_data'] = option_data
            result['surface_data'] = surface

            # 4. BS deviation score (misspecification diagnostic)
            sigma_bench = result['avg_implied_vol']
            r_bench = option_data['r']
            result['bs_deviation_score'] = compute_bs_deviation_score(
                result['sindy_result']['discovered_coefficients'],
                r=r_bench,
                sigma=sigma_bench,
            )
            logger.info(
                "%s BS deviation score: %.4f",
                ticker_symbol, result['bs_deviation_score'],
            )

            # 5. Cross-method comparison
            V_t = surface['V_surface'][:, ::-1]
            T_max = float(surface['tau_grid'].max())
            t_grid_cm = T_max - surface['tau_grid'][::-1]
            result['cross_method'] = run_cross_method_real_data(
                V_t,
                surface['K_grid'],
                t_grid_cm,
                ticker=ticker_symbol,
                sigma_eff=result['avg_implied_vol'],
                r=r_bench,
            )

            per_ticker[ticker_symbol] = result

            sindy = result['sindy_result']
            summary_rows.append({
                'ticker': ticker_symbol,
                'data_source': result['data_source'],
                'n_options': option_data['n_options'],
                'avg_implied_vol': result['avg_implied_vol'],
                'sigma_effective': result['sigma_effective'],
                'r2_score': sindy['r2_score'],
                'bs_deviation_score': result['bs_deviation_score'],
                'n_active_terms': sindy['n_active'],
                'active_terms': ', '.join(sindy['active_terms']),
                'pde': sindy['human_readable_pde'],
            })

        except Exception as exc:
            logger.error(
                "Ticker %s failed completely: %s", ticker_symbol, exc,
                exc_info=True,
            )
            continue

    # ------------------------------------------------------------------
    # Cross-ticker consistency check
    # ------------------------------------------------------------------
    cross_consistent = False
    if len(per_ticker) >= 2:
        active_sets = [
            frozenset(res['sindy_result']['active_terms'])
            for res in per_ticker.values()
        ]
        cross_consistent = len(set(active_sets)) == 1
        if cross_consistent:
            logger.info(
                "Cross-ticker consistency: PASS -- all tickers share "
                "active terms %s",
                active_sets[0],
            )
        else:
            logger.warning(
                "Cross-ticker consistency: FAIL -- different active term "
                "sets across tickers."
            )

    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    if not summary_df.empty:
        logger.info("\n%s", summary_df.to_string(index=False))

    return {
        'per_ticker_results': per_ticker,
        'cross_ticker_consistency': cross_consistent,
        'summary_df': summary_df,
    }


# ===================================================================
# Internal helpers
# ===================================================================

def _dataframe_to_result(df, ticker_symbol, data_source):
    """Convert a DataFrame of option rows into the standard result dict."""
    # Ensure expected columns exist
    for col in ('strike', 'tau', 'mid_price', 'implied_vol', 'S0', 'r'):
        if col not in df.columns:
            raise KeyError(f"Missing expected column: {col}")

    S0 = float(df['S0'].iloc[0])
    r = float(df['r'].iloc[0])

    # Compute enriched statistics
    strikes = df['strike'].values.astype(float)
    taus = df['tau'].values.astype(float)
    ivs = df['implied_vol'].values.astype(float)
    valid_ivs = ivs[np.isfinite(ivs) & (ivs > 0)]

    # Bid-ask spread percentage
    if 'bid' in df.columns and 'ask' in df.columns:
        mid = df['mid_price'].values.astype(float)
        spread = (df['ask'].values.astype(float) - df['bid'].values.astype(float))
        with np.errstate(divide='ignore', invalid='ignore'):
            spread_pct = spread / mid * 100.0
        bid_ask_spread_pct = float(np.nanmean(spread_pct))
    else:
        bid_ask_spread_pct = np.nan

    # Count unique expirations
    unique_exps = sorted(df['expiration'].unique().tolist())
    n_expirations = len(unique_exps)

    return {
        'ticker': ticker_symbol,
        'S0': S0,
        'r': r,
        'strikes': strikes,
        'expirations': unique_exps,
        'tau': taus,
        'mid_prices': df['mid_price'].values.astype(float),
        'implied_vols': ivs,
        'option_df': df,
        'data_source': data_source,
        'n_options': len(df),
        'bid_ask_spread_pct': bid_ask_spread_pct,
        'n_expirations_raw': n_expirations,
        'n_expirations_filtered': n_expirations,
        'n_contracts_raw': len(df),
        'n_contracts_filtered': len(df),
        'strike_range': (float(strikes.min()), float(strikes.max())),
        'tau_range': (float(taus.min()), float(taus.max())),
        'iv_range': (
            (float(valid_ivs.min()), float(valid_ivs.max()))
            if len(valid_ivs) > 0
            else (np.nan, np.nan)
        ),
    }
