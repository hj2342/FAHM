"""
build_player_metadata.py
------------------------
Build optional player metadata for the transfer sample from a public
Transfermarkt-derived player profiles table.

This script fetches nationality, birth-country, EU status, and a small set of
source fields that can support milestone-3 bias checks. It does not infer race
or ethnicity labels.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from urllib.request import urlopen

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

LOCAL_SITE_PACKAGES = Path(__file__).resolve().parent.parent / ".python_packages"
if LOCAL_SITE_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import pandas as pd

from config import DIR_INPUTS
from data_loader import load_transfermarkt


SOURCE_URL = (
    "https://github.com/salimt/football-datasets/raw/refs/heads/main/"
    "datalake/transfermarkt/player_profiles/player_profiles.csv"
)
SOURCE_PATH = Path(DIR_INPUTS) / "player_profiles_source.csv"
OUTPUT_PATH = Path(DIR_INPUTS) / "player_metadata.csv"
MIN_SEASON_YEAR = 2010


def _normalize_text(val: object) -> str:
    if pd.isna(val):
        return ""

    s = unicodedata.normalize("NFKD", str(val))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"\s*\(\d+\)$", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _collapse_unique(values) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for val in values:
        if pd.isna(val):
            continue
        s = str(val).strip()
        if not s or s.lower() == "nan":
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _single_value(values) -> str:
    collapsed = _collapse_unique(values)
    return collapsed[0] if len(collapsed) == 1 else ""


def _mode_birth_year(series: pd.Series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return pd.NA
    vals = vals.round().astype(int)
    return int(vals.mode().iloc[0])


def _country_has_turkey_link(*values: object) -> bool:
    tokens = [_normalize_text(v) for v in values]
    return any(
        (" turkey " in f" {token} ") or (" turkiye " in f" {token} ")
        for token in tokens
    )


def _download_source(force: bool = False) -> None:
    if SOURCE_PATH.is_file() and not force:
        print(f"[Metadata] Reusing existing source file: {SOURCE_PATH}")
        return

    print(f"[Metadata] Downloading public player profiles from:\n  {SOURCE_URL}\n")
    with urlopen(SOURCE_URL, timeout=120) as response:
        SOURCE_PATH.write_bytes(response.read())
    print(f"[Metadata] Saved source file to {SOURCE_PATH}\n")


def _load_transfer_players(dest_leagues: list[str] | None, limit: int | None) -> pd.DataFrame:
    raw, _ = load_transfermarkt(force_refresh=False)
    df = raw.copy()
    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["season_year"] = pd.to_numeric(
        df["season"].astype(str).str[:4], errors="coerce"
    ).fillna(0).astype(int)
    df["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    df = df[df["season_year"] >= MIN_SEASON_YEAR].copy()

    if dest_leagues:
        dest_set = {s.strip() for s in dest_leagues if s.strip()}
        df = df[df["dest_league"].isin(dest_set)].copy()

    df["player_name_key"] = df["player_name"].map(_normalize_text)
    df = df[df["player_name_key"].ne("")].copy()
    df["approx_birth_year"] = df["season_year"] - df["age"]

    players = (
        df.groupby(["player_name", "player_name_key"], as_index=False)
          .agg(
              transfer_rows=("player_name", "size"),
              approx_birth_year=("approx_birth_year", _mode_birth_year),
              source_dest_leagues=(
                  "dest_league",
                  lambda s: " | ".join(sorted(pd.Series(s.dropna().astype(str).unique()))),
              ),
          )
          .sort_values(["transfer_rows", "player_name"], ascending=[False, True])
          .reset_index(drop=True)
    )

    if limit is not None:
        players = players.head(limit).copy()

    print(
        f"[Metadata] Transfer sample players selected: {len(players):,} "
        f"(modern era, season_year >= {MIN_SEASON_YEAR})\n"
    )
    return players


def _load_source_profiles() -> dict[str, pd.DataFrame]:
    src = pd.read_csv(SOURCE_PATH, low_memory=False)
    src.columns = [str(c).strip() for c in src.columns]
    src["player_name_key"] = src["player_name"].map(_normalize_text)
    src["dob_year"] = pd.to_datetime(src["date_of_birth"], errors="coerce").dt.year
    src = src[src["player_name_key"].ne("")].copy()
    print(
        f"[Metadata] Loaded {len(src):,} public player-profile rows "
        f"across {src['player_name_key'].nunique():,} normalized names.\n"
    )
    return {key: grp.copy() for key, grp in src.groupby("player_name_key")}


def _build_metadata(players: pd.DataFrame, source_by_key: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for row in players.itertuples(index=False):
        candidates = source_by_key.get(row.player_name_key)

        if candidates is None or candidates.empty:
            rows.append({
                "player_name": row.player_name,
                "player_nationality": "",
                "player_birth_country": "",
                "player_birth_city": "",
                "is_eu": "",
                "turkey_link_flag": False,
                "bias_focus_group": "",
                "metadata_source": "salimt/football-datasets player_profiles",
                "source_match_quality": "no_name_match",
                "source_candidate_count": 0,
                "source_candidate_count_post_birth_year": 0,
                "source_player_names": "",
                "source_name_in_home_country": "",
                "source_main_position": "",
                "source_date_of_birth": "",
                "source_dest_leagues": row.source_dest_leagues,
                "notes": "No public profile matched this normalized player name.",
            })
            continue

        pre_birth_count = int(len(candidates))
        narrowed = candidates
        birth_year_applied = False
        if pd.notna(row.approx_birth_year):
            approx_year = int(row.approx_birth_year)
            birth_window = candidates["dob_year"].between(
                approx_year - 1,
                approx_year + 1,
                inclusive="both",
            )
            if birth_window.any():
                narrowed = candidates[birth_window].copy()
                birth_year_applied = len(narrowed) < len(candidates)

        post_birth_count = int(len(narrowed))
        player_nationality = _single_value(narrowed["citizenship"])
        player_birth_country = _single_value(narrowed["country_of_birth"])
        player_birth_city = _single_value(narrowed["place_of_birth"])
        source_name_in_home_country = _single_value(narrowed["name_in_home_country"])
        source_main_position = _single_value(narrowed["main_position"])
        source_date_of_birth = _single_value(narrowed["date_of_birth"])

        eu_vals = _collapse_unique(narrowed["is_eu"])
        is_eu = eu_vals[0] if len(eu_vals) == 1 else ""

        if pre_birth_count == 1:
            match_quality = "unique_name_match"
        elif birth_year_applied and post_birth_count == 1:
            match_quality = "birth_year_resolved"
        elif post_birth_count >= 1 and all([
            bool(player_nationality),
            bool(player_birth_country),
            len(_collapse_unique(narrowed["player_name"])) >= 1,
        ]):
            match_quality = "collapsed_duplicate_rows"
        else:
            match_quality = "ambiguous_multi_profile"

        turkey_link_flag = _country_has_turkey_link(
            player_nationality,
            player_birth_country,
        )
        bias_focus_group = (
            "Turkey-linked"
            if turkey_link_flag
            else ("Other" if match_quality != "no_name_match" else "")
        )

        note_bits = []
        if match_quality == "ambiguous_multi_profile":
            note_bits.append(
                "Multiple public profiles share this normalized name; unresolved fields are left blank."
            )
        if birth_year_applied:
            note_bits.append("Approximate birth year from transfer age/season was used to narrow candidates.")

        rows.append({
            "player_name": row.player_name,
            "player_nationality": player_nationality,
            "player_birth_country": player_birth_country,
            "player_birth_city": player_birth_city,
            "is_eu": is_eu,
            "turkey_link_flag": turkey_link_flag,
            "bias_focus_group": bias_focus_group,
            "metadata_source": "salimt/football-datasets player_profiles",
            "source_match_quality": match_quality,
            "source_candidate_count": pre_birth_count,
            "source_candidate_count_post_birth_year": post_birth_count,
            "source_player_names": " | ".join(_collapse_unique(narrowed["player_name"])),
            "source_name_in_home_country": source_name_in_home_country,
            "source_main_position": source_main_position,
            "source_date_of_birth": source_date_of_birth,
            "source_dest_leagues": row.source_dest_leagues,
            "notes": " ".join(note_bits).strip(),
        })

    meta = pd.DataFrame(rows).sort_values("player_name").reset_index(drop=True)
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build optional player metadata for the transfer sample."
    )
    parser.add_argument(
        "--refresh-source",
        action="store_true",
        help="Re-download the public player profiles source CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of transfer-sample players to process.",
    )
    parser.add_argument(
        "--dest-leagues",
        type=str,
        default="",
        help='Optional comma-separated subset, e.g. "Serie A,Bundesliga".',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dest_leagues = [s.strip() for s in args.dest_leagues.split(",") if s.strip()]

    _download_source(force=args.refresh_source)
    players = _load_transfer_players(dest_leagues=dest_leagues, limit=args.limit)
    source_by_key = _load_source_profiles()
    meta = _build_metadata(players, source_by_key)
    meta.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    nationality_ok = meta["player_nationality"].fillna("").astype(str).str.strip().ne("")
    birth_ok = meta["player_birth_country"].fillna("").astype(str).str.strip().ne("")
    eu_ok = meta["is_eu"].fillna("").astype(str).str.strip().ne("")
    turkey_ok = meta["turkey_link_flag"].fillna(False).astype(bool)

    print(f"[Metadata] Wrote {len(meta):,} rows to {OUTPUT_PATH}")
    print(
        f"[Metadata] Coverage - nationality: {int(nationality_ok.sum()):,} "
        f"({nationality_ok.mean():.1%})"
    )
    print(
        f"[Metadata] Coverage - birth country: {int(birth_ok.sum()):,} "
        f"({birth_ok.mean():.1%})"
    )
    print(
        f"[Metadata] Coverage - EU status: {int(eu_ok.sum()):,} "
        f"({eu_ok.mean():.1%})"
    )
    print(
        f"[Metadata] Turkey-linked players in metadata: {int(turkey_ok.sum()):,}\n"
    )


if __name__ == "__main__":
    main()
