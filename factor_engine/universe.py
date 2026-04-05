"""
S&P 500 universe definition and benchmark ticker.

Fetches the current S&P 500 constituent list from Wikipedia.
Raises RuntimeError if the network is unavailable or the page is unreachable,
rather than silently falling back to a small hardcoded subset.  A 20-ticker
fallback would produce statistically meaningless backtest results and give
no indication that something went wrong.

⚠ Survivorship bias: using today's index composition for a historical backtest
  means all stocks survived to the present. Bankrupt or acquired firms from
  2014–2026 are excluded, which overstates strategy returns vs a true
  point-in-time index. This is documented as a known simplifying assumption
  in analysis.md.
"""

import sys
from io import StringIO
from typing import List

import pandas as pd
import requests

# SPY is the most liquid and tightly tracking S&P 500 proxy.
# Used for regression (alpha, beta) and equity curve comparison.
BENCHMARK_TICKER = "SPY"


def get_sp500_tickers() -> List[str]:
    """
    Return the current S&P 500 constituent ticker symbols from Wikipedia.

    Source: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

    Ticker cleaning: replaces '.' with '-' so that e.g. 'BRK.B' becomes
    'BRK-B', which is the format yfinance accepts.

    Survivorship bias acknowledged: see module docstring.

    Returns
    -------
    list[str]
        ~500 ticker symbols as strings.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Wikipedia returns 403 for requests without a browser User-Agent.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text), header=0)
        df = tables[0]
        tickers = df["Symbol"].tolist()
        # Yahoo Finance uses '-' not '.' in multi-class tickers (BRK.B → BRK-B)
        tickers = [str(t).replace(".", "-") for t in tickers]
        print(f"  [universe] loaded {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
    except Exception as exc:
        # Do NOT silently fall back to a small hardcoded list — a 20-ticker
        # universe would produce garbage backtest results with no warning.
        # Raise so the caller can see the real failure and take action.
        print(
            f"\n[universe] ERROR: failed to fetch S&P 500 constituent list.\n"
            f"  Cause: {exc}\n"
            f"  Resolve: check your network connection or supply a local ticker list.\n",
            file=sys.stderr,
        )
        raise RuntimeError(
            f"Cannot fetch S&P 500 universe from Wikipedia: {exc}"
        ) from exc


def get_benchmark_ticker() -> str:
    """
    Return the benchmark ETF ticker symbol.

    SPY (SPDR S&P 500 ETF Trust) is used as the benchmark throughout
    all regression and performance attribution calculations.

    Returns
    -------
    str
        'SPY'
    """
    return BENCHMARK_TICKER
