"""
cache.py
--------
Lightweight disk-caching helpers.

All cached artefacts live under config.DIR_CACHE (default: ./cache/).
Serialisation uses pickle for DataFrames / dicts; JSON for plain dicts
when requested explicitly.

Usage
-----
    from cache import load_cache, save_cache, cache_exists

    if cache_exists("transfermarkt_raw"):
        df = load_cache("transfermarkt_raw")
    else:
        df = download_expensive_data()
        save_cache("transfermarkt_raw", df)
"""

import os
import pickle
import json
from datetime import datetime

from config import DIR_CACHE


def _path(key: str, fmt: str = "pkl") -> str:
    return os.path.join(DIR_CACHE, f"{key}.{fmt}")


def cache_exists(key: str, fmt: str = "pkl") -> bool:
    """Return True if a cache file for *key* exists on disk."""
    return os.path.isfile(_path(key, fmt))


def save_cache(key: str, obj, fmt: str = "pkl") -> None:
    """
    Persist *obj* to disk under *key*.

    Parameters
    ----------
    key : str   Short identifier, used as the filename stem.
    obj         Any pickle-able Python object (DataFrame, dict, list …).
    fmt : str   "pkl" (default) or "json".
    """
    p = _path(key, fmt)
    if fmt == "json":
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
    else:
        with open(p, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[Cache] Saved  '{key}' → {p}  ({ts})")


def load_cache(key: str, fmt: str = "pkl"):
    """
    Load and return the cached object for *key*.

    Raises FileNotFoundError if the cache file does not exist
    (call cache_exists() first if you want to branch).
    """
    p = _path(key, fmt)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"No cache found for key '{key}' at {p}")
    if fmt == "json":
        with open(p, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
    else:
        with open(p, "rb") as fh:
            obj = pickle.load(fh)
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[Cache] Loaded '{key}' ← {p}  ({ts})")
    return obj


def clear_cache(key: str | None = None, fmt: str = "pkl") -> None:
    """
    Delete a specific cache entry (or all .pkl files if key is None).
    """
    if key is not None:
        p = _path(key, fmt)
        if os.path.isfile(p):
            os.remove(p)
            print(f"[Cache] Cleared '{key}'")
    else:
        removed = 0
        for fn in os.listdir(DIR_CACHE):
            if fn.endswith(f".{fmt}"):
                os.remove(os.path.join(DIR_CACHE, fn))
                removed += 1
        print(f"[Cache] Cleared {removed} cache file(s) in '{DIR_CACHE}/'")
