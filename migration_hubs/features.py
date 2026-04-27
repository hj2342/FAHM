"""
features.py
-----------
Cleans raw Transfermarkt 'in'-transfer records and derives all analysis
features needed by the EDA, SNA, and statistical modules.

Schema produced
---------------
player_name          str
age                  float
position             str
position_group       str
club_country         str
player_nationality   str    optional merged metadata
player_birth_country str    optional merged metadata
player_birth_city    str    optional merged metadata
is_eu                bool   optional merged metadata
turkey_link_flag     bool   optional merged metadata
race_group           str    optional merged metadata
ethnicity_group      str    optional merged metadata
skin_tone_group      str    optional merged metadata
bias_focus_group     str    optional merged metadata
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
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

from config import DIR_INPUTS, PRESSURE_INDEX, PRESSURE_SOURCE


# World-record transfer cap (€222M — Neymar 2017).
# Values above this are data-entry errors in Transfermarkt.
_FEE_CAP = 222_000_000.0
_MIN_SEASON_YEAR = 2010   # pre-Bosman/FFP era excluded for consistency
_PLAYER_METADATA_PATH = Path(DIR_INPUTS) / "player_metadata.csv"


def _position_group(val: object) -> str:
    """Collapse detailed Transfermarkt positions into broad player groups."""
    if pd.isna(val):
        return "Unknown"

    s = str(val).strip().lower()
    if not s or s == "nan":
        return "Unknown"
    if "goalkeeper" in s:
        return "Goalkeeper"
    if ("back" in s) or ("defence" in s) or ("defender" in s) or ("sweeper" in s):
        return "Defender"
    if "midfield" in s:
        return "Midfielder"
    if ("winger" in s) or ("forward" in s) or ("striker" in s) or ("attack" in s):
        return "Attacker"
    return "Other"


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


def _normalize_name(val: object) -> str:
    """Normalize player names for lightweight metadata merging."""
    if pd.isna(val):
        return ""

    s = unicodedata.normalize("NFKD", str(val))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"\s*\(\d+\)$", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _load_player_metadata() -> pd.DataFrame:
    """
    Load optional player-level metadata supplied by the user.

    Expected file:
        inputs/player_metadata.csv

    Required column:
        player_name
    """
    if not _PLAYER_METADATA_PATH.is_file():
        return pd.DataFrame()

    meta = pd.read_csv(_PLAYER_METADATA_PATH)
    meta.columns = [str(c).strip() for c in meta.columns]
    if "player_name" not in meta.columns:
        print(
            "[Metadata] inputs/player_metadata.csv found but missing "
            "'player_name' column - skipping merge."
        )
        return pd.DataFrame()

    meta = meta.dropna(subset=["player_name"]).copy()
    meta["player_name"] = meta["player_name"].astype(str).str.strip()
    meta["player_name_key"] = meta["player_name"].map(_normalize_name)
    meta = meta[meta["player_name_key"].ne("")].copy()

    dupes = meta["player_name_key"].duplicated(keep=False).sum()
    if dupes:
        print(
            f"[Metadata] {dupes:,} duplicated name rows found; "
            "keeping the first occurrence per normalized player name."
        )
        meta = meta.drop_duplicates(subset=["player_name_key"], keep="first")

    print(f"[Metadata] Loaded {len(meta):,} player metadata rows.\n")
    return meta


def _merge_player_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Merge optional player-level metadata onto the engineered transfer records.

    Returns
    -------
    merged_df : pd.DataFrame
    metadata_cols : list[str]
        Metadata columns successfully added to the engineered dataset.
    """
    meta = _load_player_metadata()
    if meta.empty:
        return df, []

    merged = df.copy()
    merged["player_name_key"] = merged["player_name"].map(_normalize_name)

    meta_cols = [c for c in meta.columns if c not in {"player_name", "player_name_key"}]
    merged = merged.merge(
        meta[["player_name_key"] + meta_cols],
        how="left",
        on="player_name_key",
    )
    matched = merged["player_name_key"].isin(set(meta["player_name_key"]))
    usable_cols = [
        c for c in (
            "player_nationality",
            "player_birth_country",
            "player_birth_city",
            "is_eu",
        )
        if c in merged.columns
    ]
    if usable_cols:
        has_usable_metadata = (
            merged[usable_cols]
              .fillna("")
              .astype(str)
              .apply(lambda col: col.str.strip().ne(""))
              .any(axis=1)
        )
    else:
        has_usable_metadata = pd.Series(False, index=merged.index)
    print(
        f"[Metadata] Matched {matched.sum():,} of {len(merged):,} transfers "
        f"({matched.mean():.1%}) to player metadata; "
        f"{has_usable_metadata.sum():,} ({has_usable_metadata.mean():.1%}) "
        "have nonblank fetched nationality / birth / EU fields.\n"
    )
    return merged, meta_cols


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
        "country":            "club_country",
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

    # ── Observed player descriptors from Transfermarkt ────────────────────────
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if "position" in df.columns:
        df["position"] = (
            df["position"]
              .astype(str)
              .str.strip()
              .replace({"nan": np.nan})
        )
        df["position_group"] = df["position"].apply(_position_group)
    if "club_country" in df.columns:
        df["club_country"] = (
            df["club_country"]
              .astype(str)
              .str.strip()
              .replace({"nan": np.nan})
        )

    # ── Season year ───────────────────────────────────────────────────────────
    df["season_year"] = _extract_season_year(df)

    # ── Modern era filter ─────────────────────────────────────────────────────
    before = len(df)
    df = df[df["season_year"] >= _MIN_SEASON_YEAR].copy()
    print(
        f"[Features] season_year ≥ {_MIN_SEASON_YEAR}: "
        f"{before:,} → {len(df):,} records"
    )

    # Optional external player metadata
    df, metadata_cols = _merge_player_metadata(df)

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
        "player_name", "age", "position", "position_group", "club_country",
        "player_nationality", "player_birth_country", "player_birth_city",
        "is_eu", "turkey_link_flag",
        "race_group", "ethnicity_group", "skin_tone_group",
        "bias_focus_group", "metadata_source",
        "source_match_quality", "source_candidate_count",
        "source_candidate_count_post_birth_year",
        "source_player_names", "source_name_in_home_country",
        "source_main_position", "source_date_of_birth",
        "source_dest_leagues", "notes",
        "dest_club", "origin_club",
        "dest_league", "origin_league",
        "transfer_fee", "log_transfer_fee",
        "season", "season_year",
        "origin_pressure", "dest_pressure", "pressure_gap",
        "origin_pressure_src", "dest_pressure_src",
        "high_pressure_treat",
    ] if c in df.columns]
    keep += [c for c in metadata_cols if c not in keep and c in df.columns]
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
