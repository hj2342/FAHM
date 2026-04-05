"""
eda.py
------
Exploratory Data Analysis — three figures saved immediately on creation.

Figure 1 : Annual transfer count | top destination leagues | fee distribution
Figure 2 : Pressure index by league | pressure-gap histogram
Figure 3 : League-to-league transfer flow heatmap
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

from config import PAL, PLOT_STYLE, PRESSURE_INDEX, PRESSURE_SOURCE, DIR_FIGURES, DIR_TABLES

plt.rcParams.update(PLOT_STYLE)


# ── Figure 1 ──────────────────────────────────────────────────────────────────

def plot_eda_overview(df: pd.DataFrame) -> None:
    """Annual counts, top leagues, fee distribution — saved as fig1."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Annual transfer count
    ax = axes[0]
    annual = (
        df.groupby("season_year").size()
          .reset_index(name="n")
          .query("season_year > 2000")
    )
    ax.bar(
        annual["season_year"].astype(str), annual["n"],
        color=PAL["purple"], edgecolor="none", width=0.7,
    )
    ax.set_title("Annual Transfer Count (Incoming)")
    ax.set_xlabel("Season"); ax.set_ylabel("Transfers")
    ax.yaxis.grid(True)
    ax.set_xticklabels(annual["season_year"].astype(str), rotation=45, ha="right")

    # (b) Top destination leagues
    ax = axes[1]
    top = df["dest_league"].value_counts().head(10)
    ax.barh(top.index[::-1], top.values[::-1],
            color=PAL["teal"], edgecolor="none")
    ax.set_title("Transfers by Destination League")
    ax.set_xlabel("Number of Transfers"); ax.xaxis.grid(True)

    # (c) Fee distribution (non-zero, log-x scale)
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

    fig.suptitle("Figure 1: Transfer Market Overview",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = f"{DIR_FIGURES}/fig1_eda_overview.png"
    plt.savefig(out); plt.close()

    # Summary table
    summary = pd.DataFrame({
        "Metric": [
            "Total transfers", "Unique players", "Dest leagues", "Seasons",
            "Non-zero fee transfers", "Median fee (€M)", "Max fee (€M)",
            "Origin pressure mapped",
        ],
        "Value": [
            f"{len(df):,}",
            f"{df['player_name'].nunique():,}",
            f"{df['dest_league'].nunique()}",
            f"{df.query('season_year>2000')['season_year'].nunique()}",
            f"{(df['transfer_fee']>0).sum():,} ({(df['transfer_fee']>0).mean():.1%})",
            f"€{fees.median():.2f}M",
            f"€{fees.max():.1f}M",
            f"{df['origin_pressure'].notna().mean():.1%}",
        ],
    })
    csv_out = f"{DIR_TABLES}/table1_dataset_summary.csv"
    summary.to_csv(csv_out, index=False)
    print(f"[✓] {out}")
    print(f"[✓] {csv_out}")


# ── Figure 2 ──────────────────────────────────────────────────────────────────

def plot_pressure_overview(df: pd.DataFrame) -> None:
    """Pressure index bar chart + pressure-gap histogram — saved as fig2."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (a) Pressure bar — colour by data source
    ax = axes[0]
    pi = (
        pd.Series(PRESSURE_INDEX)
          .sort_values(ascending=True)
          .reset_index()
    )
    pi.columns = ["league", "pressure"]
    colors = [
        PAL["teal"] if PRESSURE_SOURCE.get(lg) == "statsbomb" else PAL["gold"]
        for lg in pi["league"]
    ]
    ax.barh(pi["league"], pi["pressure"], color=colors, edgecolor="none")
    ax.axvline(pi["pressure"].median(), color=PAL["coral"],
               linestyle="--", linewidth=1.5, label="Median")
    patches = [
        mpatches.Patch(color=PAL["teal"],  label="StatsBomb-computed"),
        mpatches.Patch(color=PAL["gold"],  label="[ESTIMATED]"),
    ]
    ax.legend(handles=patches, framealpha=0.3, fontsize=9)
    ax.set_title(
        "Pressure Index (Under-Pressure % per Event)\n"
        "Teal = StatsBomb-computed · Gold = [ESTIMATED]"
    )
    ax.set_xlabel("Under Pressure %"); ax.xaxis.grid(True)

    # (b) Pressure-gap distribution
    ax = axes[1]
    gap = df["pressure_gap"].dropna()
    ax.hist(gap, bins=40, color=PAL["purple"], edgecolor="none", density=True)
    ax.axvline(0, color=PAL["coral"], linestyle="--", linewidth=2,
               label="No change")
    ax.axvline(gap.mean(), color=PAL["teal"], linestyle="--", linewidth=1.5,
               label=f"Mean = {gap.mean():.2f}")
    ax.set_title("Pressure Gap Distribution\n(dest − origin)")
    ax.set_xlabel("Pressure Gap"); ax.set_ylabel("Density")
    ax.legend(framealpha=0.3); ax.yaxis.grid(True)

    fig.suptitle("Figure 2: Tactical Intensity Metrics",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = f"{DIR_FIGURES}/fig2_pressure.png"
    plt.savefig(out); plt.close()
    print(f"[✓] {out}")


# ── Figure 3 ──────────────────────────────────────────────────────────────────

def plot_flow_heatmap(df: pd.DataFrame) -> None:
    """League-to-league flow heatmap — saved as fig3."""
    known = df[
        (df["origin_league"] != "Unknown") &
        (df["dest_league"]   != "Unknown") &
        df["origin_league"].notna() &
        df["dest_league"].notna()
    ]
    if len(known) < 20:
        print("[WARN] Too few inter-league transfers for heatmap — skipping.")
        return

    top_leagues = (
        pd.concat([known["origin_league"], known["dest_league"]])
          .value_counts().head(9).index.tolist()
    )
    sub = known[
        known["origin_league"].isin(top_leagues) &
        known["dest_league"].isin(top_leagues)
    ]
    flow = (
        sub.groupby(["origin_league", "dest_league"]).size()
           .unstack(fill_value=0)
           .reindex(index=top_leagues, columns=top_leagues, fill_value=0)
    )
    # Zero out diagonal (self-loops are intra-league club moves, not corridors)
    arr = flow.values.copy()
    np.fill_diagonal(arr, 0)
    flow = pd.DataFrame(arr, index=flow.index, columns=flow.columns)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        flow, ax=ax,
        cmap=sns.color_palette("mako", as_cmap=True),
        annot=True, fmt="d",
        linewidths=0.5, linecolor="#2D333B",
        cbar_kws={"label": "Transfer Count"},
        annot_kws={"size": 9},
    )
    ax.set_title(
        "Figure 3: League-to-League Transfer Flow\n"
        "(Rows = Origin  ·  Columns = Destination)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Destination League"); ax.set_ylabel("Origin League")
    plt.xticks(rotation=35, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()

    out     = f"{DIR_FIGURES}/fig3_flow_heatmap.png"
    csv_out = f"{DIR_TABLES}/table2_transfer_flow_matrix.csv"
    plt.savefig(out); plt.close()
    flow.to_csv(csv_out)
    print(f"[✓] {out}")
    print(f"[✓] {csv_out}")


# ── Public runner ─────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    """Run all three EDA figures in sequence."""
    print("\n── EDA ──────────────────────────────────────────────────────")
    plot_eda_overview(df)
    plot_pressure_overview(df)
    plot_flow_heatmap(df)
    print()
