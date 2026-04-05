"""
Local disk cache for yfinance API responses.

Avoids repeated network calls during development by persisting downloaded
DataFrames and Series as pickle files in data/cache/.

Cache is keyed by a human-readable string (e.g., 'prices_2014-01-01_2026-04-01_503').
No TTL: the cache is permanent until explicitly cleared with clear_cache().
Delete individual .pkl files in data/cache/ to force a partial re-fetch.

Design: functions not a class, keeping the interface simple and stateless.
"""

import pickle
from pathlib import Path
from typing import Any, Optional

# Cache directory: factor-backtest-engine/data/cache/
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


def _cache_path(key: str) -> Path:
    """
    Return the full path to the .pkl file for a given cache key.

    Sanitises the key so it is safe to use as a filename on all OS.
    Replaces characters that are invalid in filenames (/, \\, :, space).
    """
    safe_key = (
        key.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )
    return CACHE_DIR / f"{safe_key}.pkl"


def load_cache(key: str) -> Optional[Any]:
    """
    Load a previously cached object from disk.

    Parameters
    ----------
    key : str
        Cache identifier (e.g., 'prices_2014-01-01_2026-04-01_503').

    Returns
    -------
    Any or None
        Deserialised object if the cache file exists and is readable.
        None if the file does not exist or is corrupt (triggers re-fetch).
    """
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        # Corrupt or incompatible pickle — treat as cache miss
        return None


def save_cache(key: str, data: Any) -> None:
    """
    Persist an object to disk under the given cache key.

    Creates the cache directory if it does not yet exist.
    Write failures (disk full, permissions) are swallowed silently so
    a cache error never propagates to crash the pipeline.

    Parameters
    ----------
    key : str
        Cache identifier.
    data : Any
        Any picklable Python object (pd.DataFrame, pd.Series, dict, …).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(key)
    try:
        with open(path, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass  # Cache write failure is non-fatal


def clear_cache() -> None:
    """
    Delete all .pkl files in the cache directory.

    Call this when fresh data is required — e.g., after the backtest
    end_date has advanced and new price history is available on Yahoo Finance.
    """
    if not CACHE_DIR.exists():
        return
    for pkl_file in CACHE_DIR.glob("*.pkl"):
        try:
            pkl_file.unlink()
        except Exception:
            pass
