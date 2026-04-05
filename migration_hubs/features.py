"""
features.py
-----------
Cleans raw Transfermarkt 'in'-transfer records and derives all analysis
features needed by the EDA, SNA, and statistical modules.

Schema produced
---------------
player_name          str
dest_club            str
origin_club          str
dest_league          str    internal league key
origin_league        str    internal league key  (or "Unknown")
transfer_fee         float  EUR (0.0 for Free/Loan)
log_transfer_fee     float  log(1 + transfer_fee)
season               str    e.g. "2017-18"
season_year          int    first year of the season
origin_pressure      float  under_pressure % (NaN if league unmapped)
dest_pressure        float
pressure_gap         float  dest − origin (NaN if either side unmapped)
origin_pressure_src  str    "statsbomb" | "estimated"
dest_pressure_src    str
high_pressure_treat  int    1 if origin_pressure ≥ median across mapped leagues
"""

import sys
import numpy as np
import pandas as pd

from config import PRESSURE_INDEX, PRESSURE_SOURCE


# World-record transfer cap (€222M — Neymar 2017).
# Values above this are data-entry errors in Transfermarkt.
_FEE_CAP = 222_000_000.0
_MIN_SEASON_YEAR = 2010   # pre-Bosman/FFP era excluded for consistency


def _parse_fee(val) -> float:
    """Convert Transfermarkt fee string / scalar → float EUR."""
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


def _extract_season_year(df: pd.DataFrame) -> pd.Series:
    """Return integer first-year from a 'season' column like '2017-18'."""
    if "season" in df.columns:
        return (
            pd.to_numeric(
                df["season"].astype(str).str[:4], errors="coerce"
            )
            .fillna(0)
            .astype(int)
        )
    if "year" in df.columns:
        return pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    return pd.Series(0, index=df.index)


def clean_and_engineer(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Transfermarkt 'in'-transfer records and build analysis features.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Output of data_loader.load_transfermarkt() — already restricted to
        transfer_movement == 'in'.

    Returns
    -------
    df : pd.DataFrame   Analysis-ready DataFrame (see module docstring).
    """
    df = df_raw.copy()

    # ── Rename to canonical schema ────────────────────────────────────────────
    df = df.rename(columns={
        "club_name":          "dest_club",
        "club_involved_name": "origin_club",
        "fee_cleaned":        "fee_raw",      # Transfermarkt column name
    })

    # ── Parse transfer fee ────────────────────────────────────────────────────
    fee_col = next(
        (c for c in ("fee_raw", "fee", "transfer_fee") if c in df.columns),
        None,
    )
    df["transfer_fee"] = (
        df[fee_col].apply(_parse_fee) if fee_col else 0.0
    )
    df["transfer_fee"] = df["transfer_fee"].clip(upper=_FEE_CAP)

    # ── Drop rows with missing player name ────────────────────────────────────
    df = df.dropna(subset=["player_name"])
    df["player_name"] = df["player_name"].str.strip()

    # ── Season year ───────────────────────────────────────────────────────────
    df["season_year"] = _extract_season_year(df)

    # ── Modern era filter ─────────────────────────────────────────────────────
    before = len(df)
    df = df[df["season_year"] >= _MIN_SEASON_YEAR].copy()
    print(
        f"[Features] season_year ≥ {_MIN_SEASON_YEAR}: "
        f"{before:,} → {len(df):,} records"
    )

    # ── Pressure metrics ──────────────────────────────────────────────────────
    df["origin_pressure"]    = df["origin_league"].map(PRESSURE_INDEX)
    df["dest_pressure"]      = df["dest_league"].map(PRESSURE_INDEX)
    df["pressure_gap"]       = df["dest_pressure"] - df["origin_pressure"]
    df["origin_pressure_src"] = df["origin_league"].map(PRESSURE_SOURCE)
    df["dest_pressure_src"]   = df["dest_league"].map(PRESSURE_SOURCE)

    # ── Outcome variable ──────────────────────────────────────────────────────
    df["log_transfer_fee"] = np.log1p(df["transfer_fee"])

    # ── PSM treatment: origin pressure ≥ median of all mapped leagues ─────────
    med = pd.Series(PRESSURE_INDEX).median()
    df["high_pressure_treat"] = (
        df["origin_pressure"].fillna(0) >= med
    ).astype(int)

    # ── Keep only transfers with a known destination league ───────────────────
    df = df[df["dest_league"].notna() & (df["dest_league"] != "Unknown")]

    # ── Select and order output columns ───────────────────────────────────────
    keep = [c for c in [
        "player_name", "dest_club", "origin_club",
        "dest_league", "origin_league",
        "transfer_fee", "log_transfer_fee",
        "season", "season_year",
        "origin_pressure", "dest_pressure", "pressure_gap",
        "origin_pressure_src", "dest_pressure_src",
        "high_pressure_treat",
    ] if c in df.columns]
    df = df[keep].reset_index(drop=True)

    # ── Validation ────────────────────────────────────────────────────────────
    print(f"[Features] {len(df):,} transfers after cleaning")
    print(f"           Dest leagues   : {df['dest_league'].nunique()}")
    print(f"           Seasons        : {df['season_year'].nunique()}")
    non_zero = (df["transfer_fee"] > 0)
    print(f"           Non-zero fees  : {non_zero.sum():,} ({non_zero.mean():.1%})")
    mapped   = df["origin_pressure"].notna()
    print(f"           Origin pressure: {mapped.sum():,} mapped ({mapped.mean():.1%})\n")

    if len(df) < 200:
        sys.exit(
            f"[ERROR] Only {len(df)} records after cleaning — too few for analysis. "
            "Check data_loader output."
        )

    return df
