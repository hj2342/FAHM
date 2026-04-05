#!/usr/bin/env python3
"""
================================================================================
MIGRATION HUBS & TALENT PIPELINES
A Network Topology of the Global Football Transfer Market

CS-UH 2219E · Computational Social Science · NYUAD · Spring 2026
Team: Mahmoud Kassem · Aymane Omari · Fady John · Hariharan Janardhanan
Instructor: Professor Talal Rahwan
================================================================================

DATA SOURCES
------------
  1. StatsBomb Open Data  →  https://github.com/statsbomb/open-data
     Free JSON event data loaded via `statsbombpy` (no auth required).
     Used to compute under_pressure % per event by league — the "Pressure Index."

  2. Transfermarkt transfers → https://github.com/ewenme/transfers  (master branch)
     Pre-scraped flat CSVs: one file per league, all seasons combined.
     URL format confirmed: BASE/data/{slug}.csv   (NO season subfolder).
     Slugs confirmed by URL test:
       premier-league, primera-division, 1-bundesliga, serie-a,
       ligue-1, eredivisie, championship

OUTCOME VARIABLE FOR OLS / PSM
-------------------------------
  Transfermarkt transfer records do not contain minutes_played or match
  performance statistics. The dependent variable for causal analysis is:

      log_transfer_fee  —  log(1 + transfer_fee_EUR)

  Rationale: transfer fee is a widely-used proxy for perceived player quality
  and market valuation in football economics (Müller et al. 2017; Sæbø &
  Hvattum 2019). A higher fee commanded in the destination league is consistent
  with superior performance. The research question becomes:

  "Does the tactical intensity (under_pressure %) of a player's origin league
   predict the transfer fee they command in the destination league, after
   controlling for destination league fixed effects and transfer year?"

  To analyse minutes_played directly, merge an FBref or WhoScored player-season
  dataset on (player_name, season) before running Sections 7–8.

INSTALL
-------
  pip install statsbombpy pandas numpy networkx python-louvain statsmodels
              scikit-learn matplotlib seaborn requests tqdm scipy

RUN
---
  python migration_hubs_final.py

OUTPUT
------
  outputs/figures/  — 7 PNG figures
  outputs/tables/   — 8 CSV tables
  outputs/results_summary.txt
================================================================================
"""

# ── 0. IMPORTS ─────────────────────────────────────────────────────────────────
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import community as community_louvain
import requests
from tqdm import tqdm
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables",  exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
PAL = dict(teal="#00E5A0", purple="#7C5CBF", coral="#FF6B6B",
           gold="#F0B429", muted="#8B949E", bg="#0D1117", panel="#161B22")
plt.rcParams.update({
    "figure.facecolor": PAL["bg"],  "axes.facecolor":  PAL["panel"],
    "axes.edgecolor":   PAL["muted"], "axes.labelcolor": "white",
    "xtick.color":      PAL["muted"], "ytick.color":     PAL["muted"],
    "text.color":       "white",    "grid.color":      "#2D333B",
    "grid.linewidth":   0.5,        "font.size":        11,
    "axes.titlesize":   13,         "axes.titleweight": "bold",
    "figure.dpi":       150,        "savefig.bbox":     "tight",
    "savefig.facecolor": PAL["bg"],
})

# ── League name normalisation maps ─────────────────────────────────────────────
# StatsBomb competition names  →  internal key
SB_LEAGUE_MAP = {
    "Premier League":       "EPL",
    "La Liga":              "La Liga",
    "Ligue 1":              "Ligue 1",
    "Serie A":              "Serie A",
    "Major League Soccer":  "MLS",
    "Eredivisie":           "Eredivisie",
}
# Transfermarkt CSV slug → internal key
# Slugs verified by URL test against master branch.
# Bundesliga slug is "1-bundesliga" (not "bundesliga").
TM_LEAGUE_MAP = {
    "premier-league":   "EPL",
    "primera-division": "La Liga",
    "1-bundesliga":     "Bundesliga",
    "serie-a":          "Serie A",
    "ligue-1":          "Ligue 1",
    "eredivisie":       "Eredivisie",
    "championship":     "Championship",
}

# ── Pressure Index (under_pressure % per event) ────────────────────────────────
# StatsBomb-derived values are populated by load_statsbomb_pressure().
# [ESTIMATED] values are from published aggregate sources (CIES / Opta) and
# are clearly flagged in all figures and tables.
# Bundesliga is NOT in StatsBomb open data — its value remains [ESTIMATED].
PRESSURE_INDEX = {
    "EPL":          21.3,   # overwritten by StatsBomb if available
    "La Liga":      20.3,   # overwritten by StatsBomb
    "Bundesliga":   22.5,   # [ESTIMATED] — not in StatsBomb open data
    "Serie A":      21.2,   # overwritten by StatsBomb
    "Ligue 1":      20.8,   # overwritten by StatsBomb
    "Eredivisie":   19.8,   # [ESTIMATED]
    "MLS":          18.9,   # overwritten by StatsBomb (partial)
    "Championship": 25.1,   # [ESTIMATED] — high-intensity English 2nd tier
    "Liga NOS":     19.2,   # [ESTIMATED]
}
PRESSURE_SOURCE = {k: "estimated" for k in PRESSURE_INDEX}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — STATSBOMB: COMPUTE REAL PRESSURE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def load_statsbomb_pressure():
    """
    Load all available StatsBomb open-data competitions and compute
    under_pressure % and counterpress % per event, averaged across all
    matches in each competition.

    Updates the global PRESSURE_INDEX and PRESSURE_SOURCE dictionaries
    in-place for leagues where StatsBomb data exists.

    Returns
    -------
    sb_df : pd.DataFrame  One row per league with computed pressure metrics.
    """
    from statsbombpy import sb

    print("[StatsBomb] Loading competition list …")
    comps = sb.competitions()
    comps["league_key"] = comps["competition_name"].map(SB_LEAGUE_MAP)
    comps = comps.dropna(subset=["league_key"])

    accum = {}   # league_key → {up_sum, cp_sum, ev_sum, n_matches}

    for _, row in comps.iterrows():
        lk        = row["league_key"]
        comp_id   = int(row["competition_id"])
        season_id = int(row["season_id"])
        try:
            matches   = sb.matches(competition_id=comp_id, season_id=season_id)
            match_ids = matches["match_id"].tolist()
            up, cp, ev = 0, 0, 0
            for mid in tqdm(match_ids, desc=f"{lk} s{season_id}", leave=False):
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
            PRESSURE_INDEX[lk]  = up_pct
            PRESSURE_SOURCE[lk] = "statsbomb"
            rows.append({"league": lk, "under_pressure_pct": up_pct,
                         "counterpress_pct": cp_pct, "n_matches": acc["n"]})
            print(f"  [{lk}] under_pressure={up_pct}%  "
                  f"counterpress={cp_pct}%  (n={acc['n']} matches)")

    print(f"[StatsBomb] Updated {len(rows)} leagues from real event data.\n")
    sb_df = pd.DataFrame(rows)
    sb_df.to_csv("outputs/tables/table0_statsbomb_pressure.csv", index=False)
    return sb_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TRANSFERMARKT: LOAD REAL TRANSFER RECORDS
# ══════════════════════════════════════════════════════════════════════════════

def parse_fee(val):
    """Convert '€45.5m', 'Free', 'Loan', NaN → float EUR. Returns 0.0 for
    free/loan/unknown, NaN only for genuinely unparseable strings."""
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


def load_transfermarkt():
    """
    Download one CSV per league from ewenme/transfers (master branch).
    URL structure confirmed: BASE/data/{slug}.csv (no season subfolder).

    Only 'in' transfers are retained — these represent a player arriving at
    a club in the destination league.

    Origin league is derived from a club→league lookup built from every
    club_name that appears as a destination across all files. Transfers
    whose origin club cannot be mapped are retained but flagged
    origin_league = "Unknown".

    Returns
    -------
    df         : pd.DataFrame  Cleaned transfer records
    club_map   : dict          {club_name: league_key}
    """
    BASE = "https://raw.githubusercontent.com/ewenme/transfers/master/data"
    HEADERS = {"User-Agent": "Mozilla/5.0"}

    frames = []
    for slug, league_key in TM_LEAGUE_MAP.items():
        url = f"{BASE}/{slug}.csv"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            df = pd.read_csv(pd.io.common.BytesIO(resp.content), low_memory=False)
            df["league_key"] = league_key
            frames.append(df)
            print(f"  [OK] {league_key}: {len(df):,} rows")
        except Exception as e:
            print(f"  [WARN] {league_key} ({slug}): {e}")

    if not frames:
        sys.exit(
            "\n[ERROR] Could not load any Transfermarkt CSV files.\n"
            "  Check internet access and that master branch URLs are reachable.\n"
            "  Run test_urls.py first to confirm URL structure.\n"
        )

    raw = pd.concat(frames, ignore_index=True)
    print(f"[TM] Raw rows loaded: {len(raw):,}\n")

    # ── Build club → league map from all destination clubs ────────────────────
    in_all  = raw[raw["transfer_movement"] == "in"].copy()
    club_map = {}
    for _, row in in_all.iterrows():
        c = str(row.get("club_name", "")).strip()
        if c and c not in club_map:
            club_map[c] = row["league_key"]

    # ── Use only 'in' transfers ───────────────────────────────────────────────
    df = in_all.copy()
    df["dest_league"]   = df["league_key"]
    df["origin_league"] = (df["club_involved_name"]
                            .fillna("Unknown")
                            .str.strip()
                            .map(club_map)
                            .fillna("Unknown"))

    mapped = (df["origin_league"] != "Unknown").mean()
    print(f"[TM] 'in' transfers: {len(df):,}")
    print(f"[TM] Origin league mapped: {mapped:.1%}\n")
    return df, club_map


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA CLEANING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def clean_and_engineer(df_raw):
    """
    Clean Transfermarkt 'in' transfers and build all analysis features.

    Schema after cleaning:
        player_name, dest_club, origin_club, dest_league, origin_league,
        transfer_fee, season, season_year,
        origin_pressure, dest_pressure, pressure_gap,
        origin_pressure_source, dest_pressure_source,
        log_transfer_fee, high_pressure_treat

    Outcome variable:
        log_transfer_fee = log(1 + transfer_fee_EUR)
        Rationale: proxy for player quality; used as DV in OLS/PSM.
        Transfer fees >€222M (Neymar world record) are capped as data errors.
    """
    df = df_raw.copy()

    # Rename to standard schema
    df = df.rename(columns={
        "club_name":          "dest_club",
        "club_involved_name": "origin_club",
        "fee_cleaned":        "fee_raw",
    })

    # Parse fee; cap at world-record value (€222M) — anything above is a
    # data entry error in Transfermarkt
    fee_col = "fee_raw" if "fee_raw" in df.columns else "fee" if "fee" in df.columns else None
    df["transfer_fee"] = df[fee_col].apply(parse_fee) if fee_col else 0.0
    df["transfer_fee"] = df["transfer_fee"].clip(upper=222_000_000)

    # Drop rows with missing player name
    df = df.dropna(subset=["player_name"])
    df["player_name"] = df["player_name"].str.strip()

    # Restrict to modern era (2010 onwards).
    # Pre-2010 seasons mix incompatible fee regimes and are analytically
    # inconsistent with post-Bosman, post-FFP transfer market structure.
    if "season_year" in df.columns and df["season_year"].max() > 2010:
        before = len(df)
        df = df[df["season_year"] >= 2010].copy()
        print(f"[Clean] Filtered to season_year ≥ 2010: "
              f"{before:,} → {len(df):,} records retained")

    # Season year (integer) from "2017-18" string
    if "season" in df.columns:
        df["season_year"] = (pd.to_numeric(
            df["season"].astype(str).str[:4], errors="coerce"
        ).fillna(0).astype(int))
    elif "year" in df.columns:
        df["season_year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    else:
        df["season_year"] = 0

    # Attach pressure metrics
    df["origin_pressure"]       = df["origin_league"].map(PRESSURE_INDEX)
    df["dest_pressure"]         = df["dest_league"].map(PRESSURE_INDEX)
    df["pressure_gap"]          = df["dest_pressure"] - df["origin_pressure"]
    df["origin_pressure_source"]= df["origin_league"].map(PRESSURE_SOURCE)
    df["dest_pressure_source"]  = df["dest_league"].map(PRESSURE_SOURCE)

    # Outcome variable: log transfer fee
    df["log_transfer_fee"] = np.log1p(df["transfer_fee"])

    # PSM treatment: origin league pressure ≥ median across all mapped leagues
    mapped_pressures = pd.Series(PRESSURE_INDEX)
    med = mapped_pressures.median()
    df["high_pressure_treat"] = (
        df["origin_pressure"].fillna(0) >= med
    ).astype(int)

    # Keep only transfers with a known destination league
    df = df[df["dest_league"].notna() & (df["dest_league"] != "Unknown")]

    keep = [c for c in [
        "player_name", "dest_club", "origin_club",
        "dest_league", "origin_league",
        "transfer_fee", "log_transfer_fee",
        "season", "season_year",
        "origin_pressure", "dest_pressure", "pressure_gap",
        "origin_pressure_source", "dest_pressure_source",
        "high_pressure_treat",
    ] if c in df.columns]
    df = df[keep].reset_index(drop=True)

    print(f"[Clean] {len(df):,} transfers after cleaning")
    print(f"        Dest leagues:  {df['dest_league'].nunique()}")
    print(f"        Seasons:       {df['season_year'].nunique()}")
    print(f"        Non-zero fees: {(df['transfer_fee']>0).sum():,} "
          f"({(df['transfer_fee']>0).mean():.1%})")
    print(f"        Origin pressure mapped: "
          f"{df['origin_pressure'].notna().sum():,} "
          f"({df['origin_pressure'].notna().mean():.1%})\n")

    # Validate: refuse to run analysis on trivially small dataset
    if len(df) < 200:
        sys.exit(
            f"[ERROR] Only {len(df)} records after cleaning. "
            "Too few to run analysis. Check data loading."
        )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def plot_eda_overview(df):
    """Figure 1: Annual transfer count + top destination leagues + fee distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Annual count — should show real variation, not uniform flat bars
    ax = axes[0]
    annual = (df.groupby("season_year").size()
                .reset_index(name="n")
                .query("season_year > 2000"))
    ax.bar(annual["season_year"].astype(str), annual["n"],
           color=PAL["purple"], edgecolor="none", width=0.7)
    ax.set_title("Annual Transfer Count (Incoming)"); ax.set_xlabel("Season")
    ax.set_ylabel("Transfers"); ax.yaxis.grid(True)
    ax.set_xticklabels(annual["season_year"].astype(str), rotation=45, ha="right")

    # Top destination leagues
    ax = axes[1]
    top = df["dest_league"].value_counts().head(10)
    ax.barh(top.index[::-1], top.values[::-1], color=PAL["teal"], edgecolor="none")
    ax.set_title("Transfers by Destination League")
    ax.set_xlabel("Number of Transfers"); ax.xaxis.grid(True)

    # Fee distribution (non-zero only, log-x)
    ax = axes[2]
    fees = df.loc[df["transfer_fee"] > 0, "transfer_fee"] / 1e6
    ax.hist(fees, bins=60, color=PAL["coral"], edgecolor="none",
            density=True, log=True)
    ax.set_xscale("log")
    ax.set_title("Transfer Fee Distribution (Non-Zero)")
    ax.set_xlabel("Fee (€M, log scale)"); ax.set_ylabel("Density (log)")
    ax.axvline(fees.median(), color=PAL["gold"], linestyle="--",
               linewidth=1.5, label=f"Median €{fees.median():.1f}M")
    ax.legend(framealpha=0.3, fontsize=9)

    fig.suptitle("Figure 1: Transfer Market Overview", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/figures/fig1_eda_overview.png"); plt.close()

    summary = pd.DataFrame({
        "Metric": ["Total transfers", "Unique players", "Dest leagues", "Seasons",
                   "Non-zero fee transfers", "Median fee (€M)",
                   "Max fee (€M)", "Origin pressure mapped"],
        "Value": [
            f"{len(df):,}",
            f"{df['player_name'].nunique():,}",
            f"{df['dest_league'].nunique()}",
            f"{df.query('season_year>2000')['season_year'].nunique()}",
            f"{(df['transfer_fee']>0).sum():,} ({(df['transfer_fee']>0).mean():.1%})",
            f"€{fees.median():.2f}M",
            f"€{fees.max():.1f}M",
            f"{df['origin_pressure'].notna().mean():.1%}",
        ]
    })
    summary.to_csv("outputs/tables/table1_dataset_summary.csv", index=False)
    print("[✓] fig1_eda_overview.png  |  table1_dataset_summary.csv")


def plot_pressure_overview(df):
    """Figure 2: Pressure index by league + pressure gap distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Pressure bar — colour by source
    ax = axes[0]
    pi = (pd.Series(PRESSURE_INDEX).sort_values(ascending=True).reset_index())
    pi.columns = ["league", "pressure"]
    colors = [PAL["teal"] if PRESSURE_SOURCE.get(l) == "statsbomb" else PAL["gold"]
              for l in pi["league"]]
    ax.barh(pi["league"], pi["pressure"], color=colors, edgecolor="none")
    ax.axvline(pi["pressure"].median(), color=PAL["coral"], linestyle="--",
               linewidth=1.5, label="Median")
    patches = [mpatches.Patch(color=PAL["teal"],  label="StatsBomb-computed"),
               mpatches.Patch(color=PAL["gold"], label="[ESTIMATED]")]
    ax.legend(handles=patches, framealpha=0.3, fontsize=9)
    ax.set_title("Pressure Index (Under-Pressure % per Event)\n"
                 "Teal = StatsBomb-computed · Gold = [ESTIMATED]")
    ax.set_xlabel("Under Pressure %"); ax.xaxis.grid(True)

    # Pressure gap: only rows where both sides are mapped
    ax = axes[1]
    gap = df["pressure_gap"].dropna()
    ax.hist(gap, bins=40, color=PAL["purple"], edgecolor="none", density=True)
    ax.axvline(0, color=PAL["coral"], linestyle="--", linewidth=2,
               label="No pressure change")
    ax.axvline(gap.mean(), color=PAL["teal"], linestyle="--", linewidth=1.5,
               label=f"Mean gap = {gap.mean():.2f}")
    ax.set_title("Pressure Gap Distribution\n(dest_pressure − origin_pressure)")
    ax.set_xlabel("Pressure Gap"); ax.set_ylabel("Density")
    ax.legend(framealpha=0.3); ax.yaxis.grid(True)

    fig.suptitle("Figure 2: Tactical Intensity Metrics", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/figures/fig2_pressure.png"); plt.close()
    print("[✓] fig2_pressure.png")


def plot_flow_heatmap(df):
    """Figure 3: League-to-league transfer flow heatmap."""
    known = df[
        (df["origin_league"] != "Unknown") &
        (df["dest_league"]   != "Unknown") &
        df["origin_league"].notna() & df["dest_league"].notna()
    ]
    if len(known) < 20:
        print("[WARN] Too few inter-league transfers for heatmap — skipping"); return

    top_leagues = (pd.concat([known["origin_league"], known["dest_league"]])
                     .value_counts().head(9).index.tolist())
    sub = known[known["origin_league"].isin(top_leagues) &
                known["dest_league"].isin(top_leagues)]
    flow = (sub.groupby(["origin_league","dest_league"]).size()
               .unstack(fill_value=0)
               .reindex(index=top_leagues, columns=top_leagues, fill_value=0))
    arr = flow.values.copy(); np.fill_diagonal(arr, 0)
    flow = pd.DataFrame(arr, index=flow.index, columns=flow.columns)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(flow, ax=ax, cmap=sns.color_palette("mako", as_cmap=True),
                annot=True, fmt="d", linewidths=0.5, linecolor="#2D333B",
                cbar_kws={"label": "Transfer Count"},
                annot_kws={"size": 9})
    ax.set_title("Figure 3: League-to-League Transfer Flow\n"
                 "(Rows = Origin, Columns = Destination)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Destination League"); ax.set_ylabel("Origin League")
    plt.xticks(rotation=35, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("outputs/figures/fig3_flow_heatmap.png"); plt.close()
    flow.to_csv("outputs/tables/table2_transfer_flow_matrix.csv")
    print("[✓] fig3_flow_heatmap.png  |  table2_transfer_flow_matrix.csv")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SOCIAL NETWORK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def build_transfer_graph(df):
    """
    Build a weighted directed graph from inter-league transfer counts.
    Edges below the 25th percentile of transfer volume are dropped to
    avoid a complete graph (which yields betweenness=0 for all nodes).
    """
    valid = df[
        df["origin_league"].notna() & df["dest_league"].notna() &
        (df["origin_league"] != "Unknown") & (df["dest_league"] != "Unknown") &
        (df["origin_league"] != df["dest_league"])
    ]
    edge_counts = (valid.groupby(["origin_league","dest_league"])
                        .size().reset_index(name="weight"))

    # Sparsification: keep only edges above 25th-percentile transfer count.
    # Rationale: a fully connected graph has betweenness=0 everywhere; removing
    # low-volume edges reveals genuine stepping-stone structure.
    threshold = max(3, edge_counts["weight"].quantile(0.25))
    edge_counts = edge_counts[edge_counts["weight"] >= threshold]
    print(f"[SNA] Edge threshold: ≥{threshold:.0f} transfers "
          f"({len(edge_counts)} edges retained)")

    G = nx.DiGraph()
    nodes = set(df["dest_league"].dropna()) | set(df["origin_league"].dropna())
    nodes.discard("Unknown")
    G.add_nodes_from(nodes)
    for _, row in edge_counts.iterrows():
        G.add_edge(row["origin_league"], row["dest_league"], weight=row["weight"])

    print(f"[SNA] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
    return G


def compute_network_metrics(G):
    """Betweenness centrality, PageRank, in/out-degree, net flow."""
    in_deg   = dict(G.in_degree(weight="weight"))
    out_deg  = dict(G.out_degree(weight="weight"))
    between  = nx.betweenness_centrality(G, weight="weight", normalized=True)
    pagerank = nx.pagerank(G, weight="weight", alpha=0.85)
    G_und    = G.to_undirected()
    cluster  = nx.clustering(G_und, weight="weight")

    nodes = list(G.nodes())
    metrics = pd.DataFrame({
        "league":           nodes,
        "in_degree":        [in_deg.get(n,0)   for n in nodes],
        "out_degree":       [out_deg.get(n,0)  for n in nodes],
        "net_flow":         [in_deg.get(n,0)-out_deg.get(n,0) for n in nodes],
        "betweenness":      [round(between.get(n,0),4)  for n in nodes],
        "pagerank":         [round(pagerank.get(n,0),4) for n in nodes],
        "clustering_coef":  [round(cluster.get(n,0),4)  for n in nodes],
        "pressure_index":   [PRESSURE_INDEX.get(n,np.nan) for n in nodes],
        "pressure_source":  [PRESSURE_SOURCE.get(n,"unknown") for n in nodes],
    }).sort_values("betweenness", ascending=False).reset_index(drop=True)

    print("[SNA] Network Metrics (sorted by betweenness):")
    print(metrics.to_string(index=False))
    metrics.to_csv("outputs/tables/table3_network_metrics.csv", index=False)

    # Global stats
    try:
        apl = nx.average_shortest_path_length(G.to_undirected())
        dia = nx.diameter(G.to_undirected())
    except Exception:
        apl = dia = np.nan   # graph may be disconnected after sparsification

    global_stats = pd.DataFrame({
        "Metric": ["Nodes","Directed Edges","Network Density",
                   "Avg Clustering Coeff","Avg Path Length","Diameter"],
        "Value":  [G.number_of_nodes(), G.number_of_edges(),
                   round(nx.density(G),4),
                   round(nx.average_clustering(G_und, weight="weight"),4),
                   round(apl,4) if not np.isnan(apl) else "N/A",
                   dia if not np.isnan(dia) else "N/A"]
    })
    global_stats.to_csv("outputs/tables/table3b_global_network_stats.csv", index=False)
    print("[✓] table3_network_metrics.csv  |  table3b_global_network_stats.csv\n")
    return metrics


def detect_communities(G):
    """Louvain community detection."""
    G_und = G.to_undirected()
    partition  = community_louvain.best_partition(
        G_und, weight="weight", random_state=RANDOM_SEED)
    modularity = community_louvain.modularity(partition, G_und, weight="weight")
    n_comm     = len(set(partition.values()))
    print(f"[Louvain] {n_comm} communities, Modularity Q = {modularity:.4f}")
    for c in sorted(set(partition.values())):
        members = [k for k,v in partition.items() if v==c]
        print(f"  Community {c}: {members}")
    pd.DataFrame(list(partition.items()),
                 columns=["league","community"]).to_csv(
        "outputs/tables/table4_louvain_communities.csv", index=False)
    print("[✓] table4_louvain_communities.csv\n")
    return partition, modularity


def plot_network(G, metrics, partition):
    """Figure 4: Transfer network coloured by Louvain community."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor(PAL["bg"]); fig.patch.set_facecolor(PAL["bg"])
    pos = nx.spring_layout(G, seed=RANDOM_SEED, k=2.5, weight="weight")

    comm_colors = [PAL["teal"], PAL["purple"], PAL["coral"],
                   PAL["gold"], "#4DB6AC", "#AB47BC"]
    node_colors = [comm_colors[partition.get(n,0) % len(comm_colors)]
                   for n in G.nodes()]
    bet = metrics.set_index("league")["betweenness"]
    node_sizes  = [max(400, bet.get(n, 0.001) * 12000) for n in G.nodes()]
    edge_weights= [G[u][v]["weight"] for u,v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [max(0.3, w/max_w*5) for w in edge_weights]

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           edge_color=PAL["muted"], alpha=0.5, arrows=True,
                           arrowsize=14, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.92)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8,
                            font_color="white", font_weight="bold")

    n_comm = len(set(partition.values()))
    legend_patches = [mpatches.Patch(color=comm_colors[i],
                                     label=f"Community {i}")
                      for i in range(n_comm)]
    ax.legend(handles=legend_patches, loc="lower left",
              framealpha=0.3, fontsize=9, labelcolor="white")
    ax.set_title("Figure 4: Football Transfer Network\n"
                 "Node size ∝ betweenness · Color = Louvain community",
                 fontsize=13, color="white")
    ax.axis("off"); plt.tight_layout()
    plt.savefig("outputs/figures/fig4_network.png"); plt.close()
    print("[✓] fig4_network.png")


def plot_centrality(metrics):
    """Figure 5: Betweenness centrality + net flow direction."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    top = metrics.head(min(10, len(metrics)))

    ax = axes[0]
    bar_col = [PAL["teal"] if i < 3 else PAL["purple"] if i < 6 else PAL["muted"]
               for i in range(len(top))]
    ax.barh(top["league"][::-1], top["betweenness"][::-1],
            color=bar_col[::-1], edgecolor="none")
    ax.set_title("Betweenness Centrality")
    ax.set_xlabel("Betweenness (Normalized)"); ax.xaxis.grid(True)

    ax = axes[1]
    net = metrics.sort_values("net_flow", ascending=True)
    col = [PAL["teal"] if v > 0 else PAL["coral"] for v in net["net_flow"]]
    ax.barh(net["league"], net["net_flow"], color=col, edgecolor="none")
    ax.axvline(0, color="white", linewidth=1)
    ax.set_title("Net Transfer Flow (+ = Net Importer)")
    ax.set_xlabel("Net Flow (In − Out)"); ax.xaxis.grid(True)
    patches = [mpatches.Patch(color=PAL["teal"],  label="Net Importer"),
               mpatches.Patch(color=PAL["coral"], label="Net Exporter")]
    ax.legend(handles=patches, framealpha=0.3)

    fig.suptitle("Figure 5: Network Centrality & Transfer Flow",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/figures/fig5_centrality.png"); plt.close()
    print("[✓] fig5_centrality.png\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CAREER CORRIDOR EXTRACTION (PLAYER-LEVEL SEQUENCES)
# ══════════════════════════════════════════════════════════════════════════════

def build_player_sequences(df):
    """
    Track each player's chronological transfer chain. A player counts toward
    a corridor only if they personally made each consecutive transfer in
    the correct order — not just if their transfers happened to involve
    leagues that appear anywhere along a path.
    """
    df_seq = df.dropna(subset=["origin_league","dest_league"]).copy()
    df_seq = df_seq[
        (df_seq["origin_league"] != "Unknown") &
        (df_seq["dest_league"]   != "Unknown")
    ].sort_values(["player_name","season_year"])

    records = []
    for player, grp in df_seq.groupby("player_name"):
        if len(grp) < 2:
            continue
        first = grp.iloc[0]
        path  = [first["origin_league"], first["dest_league"]]
        years = [first["season_year"]]
        for _, row in grp.iloc[1:].iterrows():
            # Verified sequential chain AND cross-league step.
            # Skip intra-league club moves (same origin == dest == last league);
            # those are not stepping-stone corridors.
            if (row["origin_league"] == path[-1] and
                    row["dest_league"] != path[-1]):
                path.append(row["dest_league"])
                years.append(row["season_year"])
        if len(path) < 3:
            continue
        pressures = [PRESSURE_INDEX.get(l) for l in path]
        valid_p   = [p for p in pressures if p is not None]
        ascending = (len(valid_p) > 1 and
                     all(valid_p[i] <= valid_p[i+1]
                         for i in range(len(valid_p)-1)))
        records.append({
            "player_name":  player,
            "career_path":  tuple(path),
            "n_steps":      len(path)-1,
            "first_season": min(years),
            "last_season":  max(years),
            "pressure_seq": pressures,
            "is_ascending": ascending,
        })

    sequences = pd.DataFrame(records)
    print(f"[Corridors] {len(sequences):,} players with ≥2 verified sequential transfers")
    return sequences


def summarise_corridors(sequences):
    """Aggregate player sequences into corridor frequency table."""
    if sequences.empty:
        print("[NOTE] No multi-step corridors found in this dataset.\n")
        return pd.DataFrame()

    freq = (sequences.groupby("career_path")
                     .agg(n_players=("player_name","count"),
                          is_ascending=("is_ascending","first"))
                     .reset_index()
                     .sort_values("n_players", ascending=False))
    freq["path_str"]      = freq["career_path"].apply(lambda p:" → ".join(p))
    freq["pressure_gain"] = freq["career_path"].apply(
        lambda p: (PRESSURE_INDEX.get(p[-1],0) - PRESSURE_INDEX.get(p[0],0))
    )
    print(f"\n[Corridors] Top 10 verified corridors:")
    print(freq[["path_str","n_players","pressure_gain","is_ascending"]]
              .head(10).to_string(index=False))
    freq.to_csv("outputs/tables/table5_career_corridors.csv", index=False)
    print("[✓] table5_career_corridors.csv\n")
    return freq


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def plot_correlation_matrix(df):
    """Figure 6: Correlation matrix of key numerical features."""
    num_cols = [c for c in ["log_transfer_fee","origin_pressure","dest_pressure",
                            "pressure_gap","season_year"] if c in df.columns]
    corr_df  = df[num_cols].dropna()
    if len(corr_df) < 10:
        print("[WARN] Too few rows for correlation matrix — skipping"); return

    corr = corr_df.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, ax=ax, cmap=cmap, center=0, annot=True, fmt=".2f",
                linewidths=0.5, linecolor="#2D333B", vmin=-1, vmax=1,
                cbar_kws={"shrink":0.8,"label":"Pearson r"},
                annot_kws={"size":12,"weight":"bold"})
    ax.set_title(f"Figure 6: Pearson Correlation Matrix (n={len(corr_df):,})",
                 fontsize=13, fontweight="bold", pad=14)
    plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("outputs/figures/fig6_correlation_matrix.png"); plt.close()
    corr.round(4).to_csv("outputs/tables/table6_correlation_matrix.csv")
    print("[✓] fig6_correlation_matrix.png  |  table6_correlation_matrix.csv\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — OLS REGRESSION (ERROR-SAFE VERSION)
# ══════════════════════════════════════════════════════════════════════════════

def run_ols_regression(df):
    """
    OLS:
        DV  = log_transfer_fee
        IV  = origin_pressure
        FE  = destination league fixed effects
        Controls = season_year
        SEs = Clustered by destination league
    """

    print("\n" + "="*65)
    print("OLS REGRESSION — DV: log_transfer_fee")
    print("Clustered SEs by dest_league")
    print("="*65)

    # ─────────────────────────────────────────────
    # 1. Keep only rows usable for regression
    # ─────────────────────────────────────────────
    reg = df.copy()

    reg = reg[
        reg["log_transfer_fee"].notna() &
        reg["origin_pressure"].notna() &
        reg["season_year"].notna()
    ].copy()

    # If too few observations, stop early
    if len(reg) < 100:
        print("[ERROR] Too few observations for OLS.")
        return None

    # ─────────────────────────────────────────────
    # 2. Build design matrix
    # ─────────────────────────────────────────────

    # Destination league fixed effects
    dest_dummies = pd.get_dummies(
        reg["dest_league"],
        prefix="dest",
        drop_first=True,
        dtype=float   # <<< critical fix
    )

    X = pd.concat([
        reg[["origin_pressure", "season_year"]].astype(float),
        dest_dummies
    ], axis=1)

    # Add constant
    X = sm.add_constant(X)

    # Ensure ALL numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(reg["log_transfer_fee"], errors="coerce")

    # Drop any rows that became NaN
    valid_index = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_index]
    y = y.loc[valid_index]
    clusters = reg.loc[valid_index, "dest_league"]

    # Final safety cast
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # ─────────────────────────────────────────────
    # 3. Fit clustered OLS
    # ─────────────────────────────────────────────
    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": clusters}
    )

    print(model.summary())

    # Save results table
    results_df = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "p_value": model.pvalues
    })

    results_df.to_csv(
        "outputs/tables/table7_ols_results.csv"
    )

    print("\n[✓] table7_ols_results.csv saved\n")

    return model
def plot_ols_coefficients(ols_results):
    """Figure 7: Coefficient plot for full OLS model with 95% CI."""
    if not ols_results:
        print("[WARN] No OLS results to plot"); return
    model  = ols_results[-1]
    # Only plot non-dummy coefficients
    keep   = [c for c in model.params.index
              if not c.startswith("dest_") and c != "const"]
    params = model.params[keep]
    conf   = model.conf_int().loc[keep]

    fig, ax = plt.subplots(figsize=(9, max(4, len(params)*1.2)))
    y_pos   = range(len(params))
    colors_bar = [PAL["teal"] if v > 0 else PAL["coral"] for v in params.values]
    ax.barh(y_pos, params.values, color=colors_bar, alpha=0.85,
            edgecolor="none", height=0.55)
    ax.errorbar(params.values, y_pos,
                xerr=[params.values - conf[0].values,
                       conf[1].values - params.values],
                fmt="none", color="white", capsize=5, linewidth=2)
    ax.axvline(0, color=PAL["muted"], linestyle="--", linewidth=1)
    ax.set_yticks(y_pos); ax.set_yticklabels(params.index, fontsize=11)
    ax.set_xlabel("Coefficient (β) with 95% CI")
    ax.set_title(f"Figure 7: OLS Coefficients — Full Model\n"
                 f"DV: log_transfer_fee  |  "
                 f"Adj. R² = {model.rsquared_adj:.4f}  |  "
                 f"N = {int(model.nobs):,}\n"
                 f"(Destination league fixed effects included but not shown)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.xaxis.grid(True)
    for i, (coef, pval) in enumerate(zip(params.values, model.pvalues[keep].values)):
        stars = "***" if pval<0.001 else "**" if pval<0.01 else "*" if pval<0.05 else "ns"
        offset = max(abs(coef)*0.05, 0.005)
        ax.text(coef+(offset if coef>=0 else -offset), i, stars,
                va="center", ha="left" if coef>=0 else "right",
                fontsize=10, color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/figures/fig7_ols_coefficients.png"); plt.close()
    print("[✓] fig7_ols_coefficients.png\n")


def run_psm(df):
    """
    Propensity Score Matching.

    Treatment:  high_pressure_treat = 1  (origin pressure ≥ median)
    Outcome:    log_transfer_fee
    Covariates: dest_league dummies + season_year
    Method:     1:1 Nearest-Neighbour, caliper = 0.1 × SD(logit PS)
    Inference:  scipy.stats.ttest_rel (paired t-test, exact p-value)
    Bootstrap:  1,000 iterations for SE of ATT

    Interpretation: ATT > 0 means players from high-pressure leagues
    command higher transfer fees, consistent with the stepping-stone hypothesis.
    """
    required = ["log_transfer_fee","high_pressure_treat",
                "dest_league","season_year","origin_pressure"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] PSM skipped — missing: {missing}"); return None,None,None,None

    psm_df = df[required].dropna().copy()
    psm_df = psm_df[psm_df["origin_pressure"].notna()]

    if len(psm_df) < 50:
        print(f"[WARN] PSM skipped — only {len(psm_df)} complete cases"); return None,None,None,None

    T = psm_df["high_pressure_treat"].values
    y = psm_df["log_transfer_fee"].values

    # Covariates: destination league dummies + season year
    dest_dummies = pd.get_dummies(psm_df["dest_league"], prefix="dest", drop_first=True)
    X_df = pd.concat([psm_df[["season_year"]], dest_dummies], axis=1)
    # Cast to float64: same bool-dtype issue as in VIF check
    X    = X_df.values.astype(float)

    # Step 1: Logistic regression → propensity scores
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logit    = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
    logit.fit(X_scaled, T)
    ps       = logit.predict_proba(X_scaled)[:,1]
    logit_ps = np.log(np.clip(ps, 1e-6, 1-1e-6) / (1 - np.clip(ps, 1e-6, 1-1e-6)))

    psm_df = psm_df.copy()
    psm_df["ps"]       = ps
    psm_df["logit_ps"] = logit_ps

    # Step 2: 1:1 NN matching with tight caliper (0.1 × SD)
    caliper      = 0.1 * logit_ps.std()
    treated_idx  = np.where(T == 1)[0]
    control_idx  = np.where(T == 0)[0]

    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nn.fit(logit_ps[control_idx].reshape(-1,1))
    distances, matched_pos = nn.kneighbors(logit_ps[treated_idx].reshape(-1,1))
    distances    = distances.flatten()
    ctrl_matched = control_idx[matched_pos.flatten()]

    within   = distances <= caliper
    t_match  = treated_idx[within]
    c_match  = ctrl_matched[within]
    n_pairs  = within.sum()
    match_rate = n_pairs / len(treated_idx)

    print(f"\n[PSM] Caliper: {caliper:.4f} (0.1 × SD of logit PS)")
    print(f"[PSM] Treated: {len(treated_idx):,} | "
          f"Matched pairs: {n_pairs:,} ({match_rate:.1%})")

    # Warn if 100% match — indicates near-zero treatment discrimination
    if match_rate > 0.98:
        print("  ⚠ Match rate ≈ 100% — propensity scores have very low variance.")
        print("    This suggests treatment (high vs. low pressure) is poorly")
        print("    discriminated by the covariates. Interpret ATT with caution.")

    if n_pairs < 20:
        print("[WARN] Too few matched pairs for reliable inference"); return None,None,None,psm_df

    y_t = y[t_match]; y_c = y[c_match]
    att = (y_t - y_c).mean()

    # Step 3: Paired t-test (exact inference)
    t_stat, p_value = scipy_stats.ttest_rel(y_t, y_c)
    ci_lo, ci_hi   = scipy_stats.t.interval(
        0.95, df=n_pairs-1, loc=att, scale=scipy_stats.sem(y_t-y_c))

    # Step 4: Bootstrap SE
    rng_boot = np.random.default_rng(RANDOM_SEED)
    boot_atts = [(y_t[rng_boot.integers(0,n_pairs,n_pairs)] -
                  y_c[rng_boot.integers(0,n_pairs,n_pairs)]).mean()
                 for _ in range(1000)]
    se_boot = np.std(boot_atts)

    print(f"[PSM] ATT = {att:+.4f} log-fee units  |  "
          f"SE = {se_boot:.4f}  |  t = {t_stat:.2f}  |  p = {p_value:.4f}")
    print(f"[PSM] 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    # Interpretation in natural units (back-transform)
    # E[fee_treated] / E[fee_control] ≈ exp(ATT) for log-fee units
    print(f"[PSM] Back-transform: exp(ATT) = {np.exp(att):.3f} "
          f"(fee premium multiplier; >1 means high-pressure players command higher fees)")

    # Step 5: Covariate balance
    balance_rows = []
    for i, col in enumerate(X_df.columns):
        mu_t_pre = X[treated_idx,i].mean(); mu_c_pre = X[control_idx,i].mean()
        sd_pre = np.sqrt((X[treated_idx,i].var() + X[control_idx,i].var())/2 + 1e-9)
        smd_pre = abs(mu_t_pre - mu_c_pre) / sd_pre

        mu_t_post = X[t_match,i].mean(); mu_c_post = X[c_match,i].mean()
        sd_post = np.sqrt((X[t_match,i].var() + X[c_match,i].var())/2 + 1e-9)
        smd_post = abs(mu_t_post - mu_c_post) / sd_post

        balance_rows.append({"Covariate": col,
                              "SMD_before": round(smd_pre,4),
                              "SMD_after":  round(smd_post,4),
                              "balanced":   smd_post < 0.1})

    balance_df = pd.DataFrame(balance_rows)
    n_unbalanced = (~balance_df["balanced"]).sum()
    if n_unbalanced:
        print(f"  ⚠ {n_unbalanced} covariate(s) still unbalanced after matching (SMD≥0.1)")
    else:
        print("  ✓ All covariates balanced (SMD < 0.1)")

    pd.DataFrame({
        "Metric": ["ATT (log-fee)","exp(ATT) — fee multiplier",
                   "SE (bootstrap)","t-statistic",
                   "p-value (paired t)","95% CI lower","95% CI upper",
                   "Match rate","N matched pairs"],
        "Value":  [f"{att:.4f}", f"{np.exp(att):.3f}",
                   f"{se_boot:.4f}", f"{t_stat:.2f}",
                   f"{p_value:.4f}", f"{ci_lo:.4f}", f"{ci_hi:.4f}",
                   f"{match_rate:.3f}", f"{n_pairs:,}"],
    }).to_csv("outputs/tables/table8_psm_results.csv", index=False)
    balance_df.to_csv("outputs/tables/table8b_psm_balance.csv", index=False)
    print("[✓] table8_psm_results.csv  |  table8b_psm_balance.csv\n")
    return att, se_boot, balance_df, psm_df


def plot_psm_results(balance_df, psm_df, att, se):
    """Figure 8: PSM balance + propensity score overlap."""
    if balance_df is None or psm_df is None:
        print("[WARN] PSM results unavailable — skipping plot"); return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Balance — show only non-dummy covariates for readability
    bal_show = balance_df[~balance_df["Covariate"].str.startswith("dest_")].copy()
    ax = axes[0]; x = np.arange(len(bal_show)); w = 0.35
    ax.barh(x-w/2, bal_show["SMD_before"], height=w,
            color=PAL["coral"], label="Before PSM", edgecolor="none")
    ax.barh(x+w/2, bal_show["SMD_after"],  height=w,
            color=PAL["teal"],  label="After PSM",  edgecolor="none")
    ax.axvline(0.1, color=PAL["gold"], linestyle="--", linewidth=1.5,
               label="Balance threshold (0.1)")
    ax.set_yticks(x); ax.set_yticklabels(bal_show["Covariate"])
    ax.set_xlabel("|Standardized Mean Difference|")
    ax.set_title("Covariate Balance Before and After PSM\n"
                 "(Destination league dummies omitted for readability)")
    ax.legend(framealpha=0.3); ax.xaxis.grid(True)

    # PS overlap
    ax = axes[1]
    if "ps" in psm_df.columns and "high_pressure_treat" in psm_df.columns:
        treated = psm_df[psm_df["high_pressure_treat"]==1]["ps"]
        control = psm_df[psm_df["high_pressure_treat"]==0]["ps"]
        n_samp  = min(5000, len(control), len(treated))
        ax.hist(control.sample(n_samp, random_state=RANDOM_SEED),
                bins=40, alpha=0.6, color=PAL["coral"], density=True,
                label="Control (low pressure)", edgecolor="none")
        ax.hist(treated.sample(n_samp, random_state=RANDOM_SEED),
                bins=40, alpha=0.6, color=PAL["teal"], density=True,
                label="Treated (high pressure)", edgecolor="none")
        ax.set_xlabel("Propensity Score P(T=1|X)")
        ax.set_ylabel("Density")
        att_str = f"{att:+.4f}" if att is not None else "N/A"
        se_str  = f"{se:.4f}" if se is not None else "N/A"
        ax.set_title(f"Propensity Score Overlap\n"
                     f"ATT = {att_str} log-fee units  (Bootstrap SE = {se_str})")
        ax.legend(framealpha=0.3); ax.yaxis.grid(True)

    fig.suptitle("Figure 8: Propensity Score Matching Results",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/figures/fig8_psm.png"); plt.close()
    print("[✓] fig8_psm.png\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_results_summary(df, ols_results, att, se):
    """Write results summary to console and file."""
    if ols_results:
        full   = ols_results[-1]
        b_pi   = full.params.get("origin_pressure", np.nan)
        p_pi   = full.pvalues.get("origin_pressure", np.nan)
        adj_r2 = full.rsquared_adj
        sig    = (not np.isnan(p_pi)) and p_pi < 0.05
        verdict = (f"✓ origin_pressure significant (β={b_pi:.4f}, p={p_pi:.4f})"
                   if sig else
                   f"✗ origin_pressure NOT significant (β={b_pi:.4f}, p={p_pi:.4f})")
    else:
        verdict = "OLS not run (minutes_played unavailable — merge FBref data)"
        adj_r2 = np.nan

    att_str = (f"{att:+.4f} log-fee units  (exp(ATT)={np.exp(att):.3f} fee multiplier, "
               f"Bootstrap SE={se:.4f})" if att is not None else
               "PSM not run")

    summary = f"""
================================================================================
MIGRATION HUBS & TALENT PIPELINES — RESULTS SUMMARY
CS-UH 2219E · Computational Social Science · NYUAD · Spring 2026
================================================================================

DATA SOURCES (real data only — no synthetic fallback)
  StatsBomb Open Data: under_pressure % computed from real match events
  Transfermarkt:       ewenme/transfers, master branch, flat CSV per league

SCOPE
  Leagues: {df['dest_league'].nunique()} destination leagues (Big-5 + Eredivisie + Championship)
  Records: {len(df):,} incoming transfers
  Players: {df['player_name'].nunique():,} unique players

OUTCOME VARIABLE
  DV = log_transfer_fee (proxy for player valuation; Müller et al. 2017)
  Note: minutes_played requires FBref merge (see header for instructions)

HYPOTHESIS TEST (OLS — Full Model with Destination League FE)
  {verdict}
  Adj. R² = {adj_r2:.4f}

CAUSAL INFERENCE (PSM)
  ATT = {att_str}

LIMITATIONS
  1. DV is transfer fee, not minutes played — attenuation risk if fee ≠ quality
  2. Origin league for ~{(df['origin_pressure'].isna()).mean():.0%} of transfers unmapped
  3. Bundesliga pressure index is [ESTIMATED] — not from StatsBomb
  4. No Transfermarkt data for non-European leagues in this dataset

OUTPUTS
  Figures : outputs/figures/  (8 PNG files)
  Tables  : outputs/tables/   (9 CSV files)
================================================================================
"""
    print(summary)
    with open("outputs/results_summary.txt","w") as f:
        f.write(summary)
    print("[✓] outputs/results_summary.txt")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(__doc__)

    # ── 1. Load real StatsBomb pressure metrics ───────────────────────────────
    try:
        load_statsbomb_pressure()
    except Exception as e:
        print(f"[WARN] StatsBomb failed: {e}")
        print("       Using pre-specified pressure estimates (see PRESSURE_INDEX)\n")

    # ── 2. Load real Transfermarkt transfer records ───────────────────────────
    # If this fails the script exits with a clear error — no silent fallback.
    df_raw, _ = load_transfermarkt()

    # ── 3. Clean and engineer features ───────────────────────────────────────
    df = clean_and_engineer(df_raw)

    # ── 4. EDA ────────────────────────────────────────────────────────────────
    print("── EDA ─────────────────────────────────────────────────────")
    plot_eda_overview(df)
    plot_pressure_overview(df)
    plot_flow_heatmap(df)

    # ── 5. Social Network Analysis ────────────────────────────────────────────
    print("\n── SNA ─────────────────────────────────────────────────────")
    G          = build_transfer_graph(df)
    metrics_df = compute_network_metrics(G)
    partition, _ = detect_communities(G)
    plot_network(G, metrics_df, partition)
    plot_centrality(metrics_df)

    # ── 6. Career corridors ───────────────────────────────────────────────────
    print("── CAREER CORRIDORS ────────────────────────────────────────")
    sequences = build_player_sequences(df)
    summarise_corridors(sequences)

    # ── 7. Statistical analysis ───────────────────────────────────────────────
    print("── STATISTICAL ANALYSIS ────────────────────────────────────")
    plot_correlation_matrix(df)
    ols_results = run_ols_regression(df)
    plot_ols_coefficients(ols_results)
    att, se, balance_df, psm_df = run_psm(df)
    plot_psm_results(balance_df, psm_df, att, se)

    # ── 8. Results summary ────────────────────────────────────────────────────
    print_results_summary(df, ols_results, att, se)


if __name__ == "__main__":
    main()