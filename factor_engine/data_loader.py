"""
Data fetching via yfinance: price history, fundamentals, and benchmark prices.

All three functions check the local cache first (cache.py) and only call
yfinance if the data is not already on disk.  This keeps repeated runs fast
during development without triggering Yahoo Finance rate limits.

Simplifying assumptions (all documented):
  - Survivorship bias: tickers passed in are today's S&P 500 members.
  - Fundamentals are point-in-time as of TODAY's fetch, not historical.
    A production system would use quarterly SEC filings for each date.
  - 'auto_adjust=True' means yfinance returns split/dividend-adjusted Close.
"""

import hashlib

import numpy as np
import pandas as pd

from factor_engine.cache import load_cache, save_cache


def _ticker_hash(tickers: list) -> str:
    """
    Stable 8-character hash of a ticker list, order-independent.

    Using len(tickers) as a cache key causes collisions whenever two
    different ticker universes have the same size (e.g., a 500-ticker
    S&P 500 run and a 500-ticker test run).  This hash is derived from
    the sorted, joined ticker symbols so that any change in composition
    or ordering produces a different key.
    """
    key_str = ",".join(sorted(str(t) for t in tickers))
    return hashlib.md5(key_str.encode()).hexdigest()[:8]


def fetch_price_history(
    tickers: list,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch daily adjusted close prices for all tickers.

    Returns a DataFrame (dates × tickers).  Tickers that fail to download
    are silently dropped by yfinance and absent from the output columns.
    The DataFrame is forward-filled for at most 5 days to handle short
    trading halts, then remaining NaN values are left as-is.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to download.
    start_date : str
        Inclusive start date, 'YYYY-MM-DD'.
    end_date : str
        Exclusive end date, 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        Adjusted close prices.  Index = dates, columns = ticker symbols.
    """
    cache_key = f"prices_{start_date}_{end_date}_{_ticker_hash(tickers)}"
    cached = load_cache(cache_key)
    if cached is not None:
        return cached

    try:
        import yfinance as yf

        raw = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        # yfinance returns MultiIndex columns when multiple tickers are passed
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            # Single ticker or older yfinance version
            prices = raw[["Close"]] if "Close" in raw.columns else raw

        prices = prices.ffill(limit=5)      # fill short trading halts
        prices = prices.dropna(how="all")   # drop dates with all-NaN

    except Exception as exc:
        print(f"  [data_loader] price fetch failed: {exc}")
        prices = pd.DataFrame()

    save_cache(cache_key, prices)
    return prices


def fetch_fundamentals(tickers: list) -> pd.DataFrame:
    """
    Fetch trailing PE ratio, ROE, and market cap for each ticker.

    Uses yfinance Ticker.info (one HTTP request per ticker).  Slow on first
    run but cached permanently afterward.

    Columns returned:
      pe_ratio  , trailing P/E ratio (yfinance: trailingPE)
      roe       , trailing return on equity (yfinance: returnOnEquity)
      market_cap, market capitalisation in USD (yfinance: marketCap)

    Simplifying assumption: these are today's values applied to the entire
    backtest period.  A point-in-time database would be needed for production.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols.

    Returns
    -------
    pd.DataFrame
        Columns: pe_ratio, roe, market_cap.  Index = ticker symbol.
    """
    cache_key = f"fundamentals_{_ticker_hash(tickers)}"
    cached = load_cache(cache_key)
    if cached is not None:
        return cached

    import yfinance as yf

    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            rows.append({
                "ticker": ticker,
                "pe_ratio": info.get("trailingPE", np.nan),
                "roe": info.get("returnOnEquity", np.nan),
                "market_cap": info.get("marketCap", np.nan),
            })
        except Exception:
            rows.append({
                "ticker": ticker,
                "pe_ratio": np.nan,
                "roe": np.nan,
                "market_cap": np.nan,
            })

    df = pd.DataFrame(rows).set_index("ticker")
    save_cache(cache_key, df)
    return df


def fetch_benchmark_prices(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """
    Fetch daily adjusted close prices for a single benchmark ticker (SPY).

    Returns a pd.Series indexed by date for easy period-return slicing
    in the backtester.

    Parameters
    ----------
    ticker : str
        Benchmark symbol, e.g. 'SPY'.
    start_date : str
        Inclusive start date, 'YYYY-MM-DD'.
    end_date : str
        Exclusive end date, 'YYYY-MM-DD'.

    Returns
    -------
    pd.Series
        Adjusted close prices indexed by date.
        Returns empty Series if the download fails.
    """
    cache_key = f"benchmark_{ticker}_{start_date}_{end_date}"
    cached = load_cache(cache_key)
    if cached is not None:
        return cached

    try:
        import yfinance as yf

        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
        # Single ticker may return MultiIndex or flat columns depending on version
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].iloc[:, 0]
        else:
            prices = raw["Close"]

        # Ensure Series type regardless of yfinance version
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]

        prices = prices.dropna()

    except Exception as exc:
        print(f"  [data_loader] benchmark fetch failed: {exc}")
        prices = pd.Series(dtype=float)

    save_cache(cache_key, prices)
    return prices
