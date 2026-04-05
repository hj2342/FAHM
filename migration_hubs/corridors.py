"""
corridors.py
------------
Extracts and summarises verified multi-step player career corridors.

A corridor is a chronological chain of consecutive inter-league transfers
by a single player, e.g.  Eredivisie → Bundesliga → EPL.

Only strict sequential chains are counted:
  - Each step must originate in the same league the player arrived at.
  - Intra-league club moves (same origin == destination) are skipped.

Table : table5_career_corridors.csv  — saved immediately on creation.
"""

import pandas as pd
import numpy as np

from config import PRESSURE_INDEX, DIR_TABLES


def build_player_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per player per verified sequential transfer chain.

    Parameters
    ----------
    df : pd.DataFrame   Cleaned analysis DataFrame from features.py

    Returns
    -------
    sequences : pd.DataFrame with columns
        player_name | career_path | n_steps | first_season | last_season
        pressure_seq | is_ascending
    """
    df_seq = df.dropna(subset=["origin_league", "dest_league"]).copy()
    df_seq = df_seq[
        (df_seq["origin_league"] != "Unknown") &
        (df_seq["dest_league"]   != "Unknown")
    ].sort_values(["player_name", "season_year"])

    records = []
    for player, grp in df_seq.groupby("player_name"):
        if len(grp) < 2:
            continue
        first = grp.iloc[0]
        path  = [first["origin_league"], first["dest_league"]]
        years = [first["season_year"]]

        for _, row in grp.iloc[1:].iterrows():
            # Only extend the chain when the next move starts from the
            # last known destination AND crosses a league boundary
            if (row["origin_league"] == path[-1] and
                    row["dest_league"] != path[-1]):
                path.append(row["dest_league"])
                years.append(row["season_year"])

        if len(path) < 3:   # need at least origin → hop → destination
            continue

        pressures = [PRESSURE_INDEX.get(lg) for lg in path]
        valid_p   = [p for p in pressures if p is not None]
        ascending = (
            len(valid_p) > 1 and
            all(valid_p[i] <= valid_p[i + 1] for i in range(len(valid_p) - 1))
        )
        records.append({
            "player_name":  player,
            "career_path":  tuple(path),
            "n_steps":      len(path) - 1,
            "first_season": min(years),
            "last_season":  max(years),
            "pressure_seq": pressures,
            "is_ascending": ascending,
        })

    sequences = pd.DataFrame(records)
    print(
        f"[Corridors] {len(sequences):,} players with "
        "≥2 verified sequential inter-league transfers"
    )
    return sequences


def summarise_corridors(sequences: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player sequences into corridor frequency table.
    Saves table5 immediately.

    Returns
    -------
    freq : pd.DataFrame  sorted by number of players descending
    """
    if sequences.empty:
        print("[NOTE] No multi-step corridors found in this dataset.\n")
        return pd.DataFrame()

    freq = (
        sequences.groupby("career_path")
                 .agg(
                     n_players   = ("player_name",  "count"),
                     is_ascending= ("is_ascending", "first"),
                 )
                 .reset_index()
                 .sort_values("n_players", ascending=False)
    )
    freq["path_str"]      = freq["career_path"].apply(
        lambda p: " → ".join(p)
    )
    freq["pressure_gain"] = freq["career_path"].apply(
        lambda p: (
            PRESSURE_INDEX.get(p[-1], 0) - PRESSURE_INDEX.get(p[0], 0)
        )
    )

    print("\n[Corridors] Top 10 verified corridors:")
    print(
        freq[["path_str", "n_players", "pressure_gain", "is_ascending"]]
        .head(10)
        .to_string(index=False)
    )

    csv_out = f"{DIR_TABLES}/table5_career_corridors.csv"
    freq.to_csv(csv_out, index=False)
    print(f"[✓] {csv_out}\n")
    return freq


# ── Public runner ─────────────────────────────────────────────────────────────

def run_corridors(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and summarise career corridors. Returns corridor freq table."""
    print("\n── CAREER CORRIDORS ─────────────────────────────────────────")
    sequences = build_player_sequences(df)
    freq      = summarise_corridors(sequences)
    return freq
