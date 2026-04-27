"""
data_loader.py
--------------
Loads raw data from two external sources:

  1. StatsBomb Open Data (via statsbombpy)
     → computes under_pressure % per event for each available league.
     → updates config.PRESSURE_INDEX and config.PRESSURE_SOURCE in-place.

  2. Transfermarkt transfers (ewenme/transfers GitHub repository)
     → downloads one flat CSV per league and concatenates.

Both loaders are wrapped with disk-caching (see cache.py):
  * On first run  → downloads / computes and writes to ./cache/
  * On later runs → reads directly from ./cache/ (no network calls)

Pass  force_refresh=True  to any loader to bypass the cache and re-download.
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from config import (
    SB_LEAGUE_MAP, TM_LEAGUE_MAP,
    PRESSURE_INDEX, PRESSURE_SOURCE,
    DIR_TABLES,
)
from cache import cache_exists, load_cache, save_cache

warnings.filterwarnings("ignore")

# ── Transfermarkt raw URL ─────────────────────────────────────────────────────
_TM_BASE    = "https://raw.githubusercontent.com/ewenme/transfers/master/data"
_TM_HEADERS = {"User-Agent": "Mozilla/5.0"}
_LOCAL_PRESSURE_TABLE = Path(DIR_TABLES) / "table0_statsbomb_pressure.csv"


def _apply_pressure_rows(sb_df: pd.DataFrame) -> pd.DataFrame:
    """Update the shared pressure dictionaries from a league-level table."""
    for _, row in sb_df.iterrows():
        PRESSURE_INDEX[row["league"]] = row["under_pressure_pct"]
        PRESSURE_SOURCE[row["league"]] = "statsbomb"
    return sb_df


def _load_local_pressure_snapshot() -> pd.DataFrame:
    """Load the checked-in pressure snapshot when live StatsBomb access is absent."""
    if not _LOCAL_PRESSURE_TABLE.is_file():
        return pd.DataFrame()

    try:
        sb_df = pd.read_csv(_LOCAL_PRESSURE_TABLE)
        if sb_df.empty:
            return sb_df
        _apply_pressure_rows(sb_df)
        print(f"[StatsBomb] Loaded {len(sb_df)} leagues from local snapshot.\n")
        return sb_df
    except Exception as e:
        print(f"[WARN] Local pressure snapshot unreadable: {e}\n")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 1. STATSBOMB — Pressure Metrics
# ══════════════════════════════════════════════════════════════════════════════

def load_statsbomb_pressure(force_refresh: bool = False) -> pd.DataFrame:
    """
    Compute under_pressure % and counterpress % from StatsBomb open data.

    Results are cached to  cache/statsbomb_pressure.pkl.
    On cache hit the global PRESSURE_INDEX / PRESSURE_SOURCE dicts are
    still updated from the cached values so downstream code is consistent.

    Parameters
    ----------
    force_refresh : bool
        If True, ignore any existing cache and re-download from StatsBomb.

    Returns
    -------
    sb_df : pd.DataFrame
        One row per league:  league | under_pressure_pct | counterpress_pct | n_matches
    """
    cache_key = "statsbomb_pressure"

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if not force_refresh and cache_exists(cache_key):
        sb_df = _apply_pressure_rows(load_cache(cache_key))
        print(f"[StatsBomb] Loaded {len(sb_df)} leagues from cache.\n")
        return sb_df

    if not force_refresh:
        local_snapshot = _load_local_pressure_snapshot()
        if not local_snapshot.empty:
            return local_snapshot

    # ── Cache miss — fetch from StatsBomb ─────────────────────────────────────
    try:
        from statsbombpy import sb
    except ImportError:
        print("[WARN] statsbombpy not installed - trying local pressure snapshot.\n")
        snapshot = _load_local_pressure_snapshot()
        return snapshot if not snapshot.empty else pd.DataFrame()

    print("[StatsBomb] Loading competition list …")
    try:
        comps = sb.competitions()
    except Exception as e:
        print(f"[WARN] StatsBomb competitions() failed: {e}\n")
        snapshot = _load_local_pressure_snapshot()
        return snapshot if not snapshot.empty else pd.DataFrame()

    comps["league_key"] = comps["competition_name"].map(SB_LEAGUE_MAP)
    comps = comps.dropna(subset=["league_key"])

    accum: dict[str, dict] = {}

    for _, row in comps.iterrows():
        lk        = row["league_key"]
        comp_id   = int(row["competition_id"])
        season_id = int(row["season_id"])
        try:
            matches   = sb.matches(competition_id=comp_id, season_id=season_id)
            match_ids = matches["match_id"].tolist()
            up, cp, ev = 0, 0, 0
            for mid in tqdm(match_ids,
                            desc=f"{lk} s{season_id}", leave=False):
                events = sb.events(match_id=mid)
                up += int(events["under_pressure"].fillna(False).sum())
                cp += int(events["counterpress"].fillna(False).sum())
                ev += len(events)
            if ev == 0:
                continue
            if lk not in accum:
                accum[lk] = {"up": 0, "cp": 0, "ev": 0, "n": 0}
            accum[lk]["up"] += up
            accum[lk]["cp"] += cp
            accum[lk]["ev"] += ev
            accum[lk]["n"]  += len(match_ids)
        except Exception as e:
            print(f"  [WARN] {lk} s{season_id}: {e}")

    rows = []
    for lk, acc in accum.items():
        if acc["ev"] > 0:
            up_pct = round(acc["up"] / acc["ev"] * 100, 3)
            cp_pct = round(acc["cp"] / acc["ev"] * 100, 3)
            rows.append({
                "league":             lk,
                "under_pressure_pct": up_pct,
                "counterpress_pct":   cp_pct,
                "n_matches":          acc["n"],
            })
            print(f"  [{lk}] under_pressure={up_pct}%  "
                  f"counterpress={cp_pct}%  (n={acc['n']} matches)")

    print(f"[StatsBomb] Updated {len(rows)} leagues from real event data.\n")
    sb_df = _apply_pressure_rows(pd.DataFrame(rows))

    # Persist immediately
    save_cache(cache_key, sb_df)
    sb_df.to_csv(f"{DIR_TABLES}/table0_statsbomb_pressure.csv", index=False)
    print(f"[✓] table0_statsbomb_pressure.csv")

    return sb_df


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRANSFERMARKT — Transfer Records
# ══════════════════════════════════════════════════════════════════════════════

def _parse_fee(val) -> float:
    """
    Convert Transfermarkt fee strings to float EUR.

    Examples
    --------
    '€45.5m'  → 45_500_000.0
    '€300k'   → 300_000.0
    'Free'    → 0.0
    'Loan'    → 0.0
    NaN       → 0.0
    """
    if pd.isna(val):
        return 0.0
    s = str(val).strip().lower()
    if s in ("free", "loan", "-", "", "?", "unknown"):
        return 0.0
    s = s.replace("€", "").replace(",", "").replace(" ", "")
    try:
        if "m" in s:
            return float(s.replace("m", "")) * 1_000_000
        if "k" in s:
            return float(s.replace("k", "")) * 1_000
        return float(s)
    except ValueError:
        return 0.0


def load_transfermarkt(force_refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Download Transfermarkt transfer CSVs and return cleaned 'in' transfers.

    Cache
    -----
    Raw concatenated DataFrame  → cache/transfermarkt_raw.pkl
    Club → league lookup dict   → cache/club_map.pkl

    Parameters
    ----------
    force_refresh : bool
        Bypass cache and re-download from GitHub.

    Returns
    -------
    df       : pd.DataFrame  All 'in' transfers with dest_league / origin_league.
    club_map : dict          {club_name: league_key}
    """
    cache_key_df  = "transfermarkt_raw"
    cache_key_cm  = "club_map"

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if (not force_refresh
            and cache_exists(cache_key_df)
            and cache_exists(cache_key_cm)):
        df       = load_cache(cache_key_df)
        club_map = load_cache(cache_key_cm)
        print(f"[TM] Loaded {len(df):,} 'in' transfers from cache.\n")
        return df, club_map

    # ── Cache miss — fetch from GitHub ────────────────────────────────────────
    print("[TM] Downloading transfer CSVs …")
    frames = []
    for slug, league_key in TM_LEAGUE_MAP.items():
        url = f"{_TM_BASE}/{slug}.csv"
        try:
            resp = requests.get(url, headers=_TM_HEADERS, timeout=20)
            resp.raise_for_status()
            chunk = pd.read_csv(
                pd.io.common.BytesIO(resp.content), low_memory=False
            )
            chunk["league_key"] = league_key
            frames.append(chunk)
            print(f"  [OK] {league_key}: {len(chunk):,} rows")
        except Exception as e:
            print(f"  [WARN] {league_key} ({slug}): {e}")

    if not frames:
        sys.exit(
            "\n[ERROR] Could not load any Transfermarkt CSV files.\n"
            "  Check internet connectivity and GitHub URL availability.\n"
        )

    raw = pd.concat(frames, ignore_index=True)
    print(f"[TM] Raw rows: {len(raw):,}")

    # ── Build club → league lookup from all inbound records ───────────────────
    in_all   = raw[raw["transfer_movement"] == "in"].copy()
    club_map: dict[str, str] = {}
    for _, row in in_all.iterrows():
        c = str(row.get("club_name", "")).strip()
        if c and c not in club_map:
            club_map[c] = row["league_key"]

    # ── Enrich with origin league ─────────────────────────────────────────────
    df = in_all.copy()
    df["dest_league"]   = df["league_key"]
    df["origin_league"] = (
        df["club_involved_name"]
          .fillna("Unknown")
          .str.strip()
          .map(club_map)
          .fillna("Unknown")
    )

    mapped = (df["origin_league"] != "Unknown").mean()
    print(f"[TM] 'in' transfers: {len(df):,}")
    print(f"[TM] Origin league mapped: {mapped:.1%}\n")

    # Persist immediately
    save_cache(cache_key_df, df)
    save_cache(cache_key_cm, club_map)

    return df, club_map
