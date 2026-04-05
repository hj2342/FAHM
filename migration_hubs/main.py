"""
main.py
-------
Top-level orchestrator for the Migration Hubs & Talent Pipelines analysis.

Run
---
    python main.py

Optional flags
--------------
    python main.py --refresh-data    # ignore cache; re-download everything
    python main.py --clear-cache     # wipe ./cache/ and exit

Output
------
    outputs/figures/   8 PNG figures
    outputs/tables/    9 CSV tables
    outputs/results_summary.txt
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Project modules ───────────────────────────────────────────────────────────
from config import PRESSURE_INDEX, PRESSURE_SOURCE, DIR_TABLES, RANDOM_SEED
from cache  import clear_cache
from data_loader import load_statsbomb_pressure, load_transfermarkt
from features    import clean_and_engineer
from eda         import run_eda
from network     import run_network_analysis
from corridors   import run_corridors
from stats       import run_statistical_analysis

np.random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# Results summary
# ══════════════════════════════════════════════════════════════════════════════

def write_results_summary(df: pd.DataFrame, model, att, se) -> None:
    """Print and save a structured results summary."""
    if model is not None:
        b_pi   = model.params.get("origin_pressure", np.nan)
        p_pi   = model.pvalues.get("origin_pressure", np.nan)
        adj_r2 = model.rsquared_adj
        sig    = (not np.isnan(p_pi)) and p_pi < 0.05
        verdict = (
            f"✓ origin_pressure SIGNIFICANT  β={b_pi:.4f}, p={p_pi:.4f}"
            if sig else
            f"✗ origin_pressure NOT significant  β={b_pi:.4f}, p={p_pi:.4f}"
        )
    else:
        verdict = "OLS not run — merge FBref data for minutes_played DV"
        adj_r2  = float("nan")

    att_str = (
        f"{att:+.4f} log-fee units  "
        f"(exp(ATT)={np.exp(att):.3f} fee multiplier, "
        f"Bootstrap SE={se:.4f})"
        if att is not None else "PSM not run"
    )
    unmapped_pct = df["origin_pressure"].isna().mean()
    non_zero_fee = (df["transfer_fee"] > 0).sum()

    summary = f"""
================================================================================
MIGRATION HUBS & TALENT PIPELINES — RESULTS SUMMARY
CS-UH 2219E · Computational Social Science · NYUAD · Spring 2026
Team: Mahmoud Kassem · Aymane Omari · Fady John · Hariharan Janardhanan
Instructor: Professor Talal Rahwan
================================================================================

DATA SOURCES
  StatsBomb Open Data: under_pressure % computed from real match events
  Transfermarkt:       ewenme/transfers (master branch, flat CSV per league)

SCOPE
  Dest leagues   : {df['dest_league'].nunique()}
  Total records  : {len(df):,}
  Unique players : {df['player_name'].nunique():,}
  Non-zero fees  : {non_zero_fee:,} ({non_zero_fee/len(df):.1%})
  Season range   : {df['season_year'].min()}–{df['season_year'].max()}

OUTCOME VARIABLE
  DV = log_transfer_fee  (proxy for player valuation; Müller et al. 2017)
  Note: to use minutes_played, merge FBref/WhoScored on (player_name, season)

HYPOTHESIS TEST — OLS (dest-league FE, clustered SEs)
  {verdict}
  Adj. R² = {adj_r2:.4f}

CAUSAL INFERENCE — PSM (1:1 NN, caliper = 0.1 × SD logit PS)
  ATT = {att_str}

LIMITATIONS
  1. DV is transfer fee, not playing time — attenuation risk if fee ≠ quality
  2. Origin league unmapped for ~{unmapped_pct:.0%} of transfers
  3. Bundesliga pressure index is [ESTIMATED] — StatsBomb data unavailable
  4. No Transfermarkt records for non-European leagues in this dataset

OUTPUTS
  Figures  :  outputs/figures/  (8 PNG files)
  Tables   :  outputs/tables/   (9 CSV files)
  Summary  :  outputs/results_summary.txt
================================================================================
"""
    print(summary)
    path = "outputs/results_summary.txt"
    with open(path, "w") as fh:
        fh.write(summary)
    print(f"[✓] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migration Hubs & Talent Pipelines — Replication Script"
    )
    parser.add_argument(
        "--refresh-data", action="store_true",
        help="Ignore cache and re-download all raw data.",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Delete all cached files and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.clear_cache:
        clear_cache()
        print("Cache cleared. Exiting.")
        sys.exit(0)

    force = args.refresh_data

    # ── 1. StatsBomb pressure metrics ─────────────────────────────────────────
    print("\n── STATSBOMB ─────────────────────────────────────────────────")
    try:
        load_statsbomb_pressure(force_refresh=force)
    except Exception as e:
        print(f"[WARN] StatsBomb failed: {e}")
        print("       Falling back to pre-specified pressure estimates.\n")

    # ── 2. Transfermarkt transfer records ─────────────────────────────────────
    print("\n── TRANSFERMARKT ─────────────────────────────────────────────")
    df_raw, _ = load_transfermarkt(force_refresh=force)

    # ── 3. Feature engineering ────────────────────────────────────────────────
    print("\n── FEATURE ENGINEERING ───────────────────────────────────────")
    df = clean_and_engineer(df_raw)

    # ── 4. EDA  (figures 1–3 + tables 1–2 saved immediately) ─────────────────
    run_eda(df)

    # ── 5. SNA  (figures 4–5 + tables 3–4 saved immediately) ─────────────────
    run_network_analysis(df)

    # ── 6. Career corridors  (table 5 saved immediately) ──────────────────────
    run_corridors(df)

    # ── 7. Statistics  (figures 6–8 + tables 6–8 saved immediately) ───────────
    model, att, se_boot, _ = run_statistical_analysis(df)

    # ── 8. Results summary ────────────────────────────────────────────────────
    write_results_summary(df, model, att, se_boot)


if __name__ == "__main__":
    main()
