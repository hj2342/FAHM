"""
network.py
----------
Social Network Analysis of the inter-league transfer graph.

Figure 4 : Transfer network coloured by Louvain community
Figure 5 : Betweenness centrality + net flow direction

Tables   : table3_network_metrics.csv
           table3b_global_network_stats.csv
           table4_louvain_communities.csv
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import numpy as np

# python-louvain  (pip install python-louvain)
try:
    import community as community_louvain
    _LOUVAIN_OK = True
except ImportError:
    _LOUVAIN_OK = False
    print("[WARN] python-louvain not installed — community detection skipped.")

from config import PAL, PLOT_STYLE, PRESSURE_INDEX, PRESSURE_SOURCE, RANDOM_SEED, DIR_FIGURES, DIR_TABLES

plt.rcParams.update(PLOT_STYLE)


# ══════════════════════════════════════════════════════════════════════════════
# Graph construction
# ══════════════════════════════════════════════════════════════════════════════

def build_transfer_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a weighted directed graph from inter-league transfer counts.

    Edges below the 25th-percentile of volume are pruned to avoid a
    complete graph (which would collapse betweenness centrality to zero
    for all nodes and obscure stepping-stone structure).
    """
    valid = df[
        df["origin_league"].notna() &
        df["dest_league"].notna() &
        (df["origin_league"] != "Unknown") &
        (df["dest_league"]   != "Unknown") &
        (df["origin_league"] != df["dest_league"])
    ]
    edge_counts = (
        valid.groupby(["origin_league", "dest_league"])
             .size()
             .reset_index(name="weight")
    )

    threshold = max(3, edge_counts["weight"].quantile(0.25))
    edge_counts = edge_counts[edge_counts["weight"] >= threshold]
    print(f"[SNA] Edge threshold: ≥{threshold:.0f} transfers "
          f"({len(edge_counts)} edges retained)")

    G = nx.DiGraph()
    all_nodes = (
        set(df["dest_league"].dropna()) |
        set(df["origin_league"].dropna())
    )
    all_nodes.discard("Unknown")
    G.add_nodes_from(all_nodes)

    for _, row in edge_counts.iterrows():
        G.add_edge(row["origin_league"], row["dest_league"],
                   weight=row["weight"])

    print(f"[SNA] Graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges\n")
    return G


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_network_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute betweenness centrality, PageRank, in/out-degree, and net flow.
    Saves table3 and table3b immediately.
    """
    in_deg   = dict(G.in_degree(weight="weight"))
    out_deg  = dict(G.out_degree(weight="weight"))
    between  = nx.betweenness_centrality(G, weight="weight", normalized=True)
    pagerank = nx.pagerank(G, weight="weight", alpha=0.85)
    G_und    = G.to_undirected()
    cluster  = nx.clustering(G_und, weight="weight")

    nodes = list(G.nodes())
    metrics = pd.DataFrame({
        "league":          nodes,
        "in_degree":       [in_deg.get(n, 0)   for n in nodes],
        "out_degree":      [out_deg.get(n, 0)  for n in nodes],
        "net_flow":        [in_deg.get(n,0) - out_deg.get(n,0) for n in nodes],
        "betweenness":     [round(between.get(n, 0), 4)  for n in nodes],
        "pagerank":        [round(pagerank.get(n, 0), 4) for n in nodes],
        "clustering_coef": [round(cluster.get(n, 0), 4)  for n in nodes],
        "pressure_index":  [PRESSURE_INDEX.get(n, np.nan) for n in nodes],
        "pressure_source": [PRESSURE_SOURCE.get(n, "unknown") for n in nodes],
    }).sort_values("betweenness", ascending=False).reset_index(drop=True)

    print("[SNA] Network Metrics (sorted by betweenness):")
    print(metrics.to_string(index=False))

    csv1 = f"{DIR_TABLES}/table3_network_metrics.csv"
    metrics.to_csv(csv1, index=False)
    print(f"[✓] {csv1}")

    # Global statistics
    try:
        apl = nx.average_shortest_path_length(G_und)
        dia = nx.diameter(G_und)
    except Exception:
        apl = dia = float("nan")

    global_stats = pd.DataFrame({
        "Metric": [
            "Nodes", "Directed Edges", "Network Density",
            "Avg Clustering Coeff", "Avg Path Length", "Diameter",
        ],
        "Value": [
            G.number_of_nodes(),
            G.number_of_edges(),
            round(nx.density(G), 4),
            round(nx.average_clustering(G_und, weight="weight"), 4),
            round(apl, 4) if not (isinstance(apl, float) and np.isnan(apl)) else "N/A",
            dia if not (isinstance(dia, float) and np.isnan(dia)) else "N/A",
        ],
    })
    csv2 = f"{DIR_TABLES}/table3b_global_network_stats.csv"
    global_stats.to_csv(csv2, index=False)
    print(f"[✓] {csv2}\n")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Community detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_communities(G: nx.DiGraph) -> tuple[dict, float]:
    """
    Louvain community detection on the undirected projection.
    Saves table4 immediately.
    Returns (partition dict, modularity Q).
    """
    if not _LOUVAIN_OK:
        # Fallback: all nodes in one community
        partition  = {n: 0 for n in G.nodes()}
        modularity = 0.0
        return partition, modularity

    G_und      = G.to_undirected()
    partition  = community_louvain.best_partition(
        G_und, weight="weight", random_state=RANDOM_SEED
    )
    modularity = community_louvain.modularity(
        partition, G_und, weight="weight"
    )
    n_comm = len(set(partition.values()))
    print(f"[Louvain] {n_comm} communities, Modularity Q = {modularity:.4f}")
    for c in sorted(set(partition.values())):
        members = [k for k, v in partition.items() if v == c]
        print(f"  Community {c}: {members}")

    csv_out = f"{DIR_TABLES}/table4_louvain_communities.csv"
    pd.DataFrame(
        list(partition.items()), columns=["league", "community"]
    ).to_csv(csv_out, index=False)
    print(f"[✓] {csv_out}\n")
    return partition, modularity


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Transfer network
# ══════════════════════════════════════════════════════════════════════════════

def plot_network(
    G: nx.DiGraph,
    metrics: pd.DataFrame,
    partition: dict,
) -> None:
    """Spring-layout directed graph coloured by community — saved as fig4."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor(PAL["bg"]); fig.patch.set_facecolor(PAL["bg"])

    pos = nx.spring_layout(G, seed=RANDOM_SEED, k=2.5, weight="weight")

    comm_colors = [
        PAL["teal"], PAL["purple"], PAL["coral"],
        PAL["gold"], "#4DB6AC", "#AB47BC",
    ]
    node_colors = [
        comm_colors[partition.get(n, 0) % len(comm_colors)]
        for n in G.nodes()
    ]
    bet        = metrics.set_index("league")["betweenness"]
    node_sizes = [max(400, bet.get(n, 0.001) * 12000) for n in G.nodes()]

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w        = max(edge_weights) if edge_weights else 1
    edge_widths  = [max(0.3, w / max_w * 5) for w in edge_weights]

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths, edge_color=PAL["muted"], alpha=0.5,
        arrows=True, arrowsize=14,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors, node_size=node_sizes, alpha=0.92,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=8, font_color="white", font_weight="bold",
    )

    n_comm = len(set(partition.values()))
    patches = [
        mpatches.Patch(color=comm_colors[i], label=f"Community {i}")
        for i in range(n_comm)
    ]
    ax.legend(handles=patches, loc="lower left",
              framealpha=0.3, fontsize=9, labelcolor="white")
    ax.set_title(
        "Figure 4: Football Transfer Network\n"
        "Node size ∝ betweenness  ·  Color = Louvain community",
        fontsize=13, color="white",
    )
    ax.axis("off"); plt.tight_layout()

    out = f"{DIR_FIGURES}/fig4_network.png"
    plt.savefig(out); plt.close()
    print(f"[✓] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Centrality & flow
# ══════════════════════════════════════════════════════════════════════════════

def plot_centrality(metrics: pd.DataFrame) -> None:
    """Betweenness centrality + net flow bars — saved as fig5."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    top = metrics.head(min(10, len(metrics)))

    # (a) Betweenness
    ax = axes[0]
    bar_col = [
        PAL["teal"]   if i < 3 else
        PAL["purple"] if i < 6 else
        PAL["muted"]
        for i in range(len(top))
    ]
    ax.barh(top["league"][::-1], top["betweenness"][::-1],
            color=bar_col[::-1], edgecolor="none")
    ax.set_title("Betweenness Centrality")
    ax.set_xlabel("Betweenness (Normalized)"); ax.xaxis.grid(True)

    # (b) Net flow
    ax = axes[1]
    net = metrics.sort_values("net_flow", ascending=True)
    col = [PAL["teal"] if v > 0 else PAL["coral"] for v in net["net_flow"]]
    ax.barh(net["league"], net["net_flow"], color=col, edgecolor="none")
    ax.axvline(0, color="white", linewidth=1)
    ax.set_title("Net Transfer Flow (+ = Net Importer)")
    ax.set_xlabel("Net Flow (In − Out)"); ax.xaxis.grid(True)
    patches = [
        mpatches.Patch(color=PAL["teal"],  label="Net Importer"),
        mpatches.Patch(color=PAL["coral"], label="Net Exporter"),
    ]
    ax.legend(handles=patches, framealpha=0.3)

    fig.suptitle("Figure 5: Network Centrality & Transfer Flow",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = f"{DIR_FIGURES}/fig5_centrality.png"
    plt.savefig(out); plt.close()
    print(f"[✓] {out}\n")


# ── Public runner ─────────────────────────────────────────────────────────────

def run_network_analysis(df: pd.DataFrame) -> tuple[nx.DiGraph, pd.DataFrame, dict]:
    """Run full SNA pipeline. Returns (G, metrics_df, partition)."""
    print("\n── SNA ──────────────────────────────────────────────────────")
    G           = build_transfer_graph(df)
    metrics_df  = compute_network_metrics(G)
    partition, _ = detect_communities(G)
    plot_network(G, metrics_df, partition)
    plot_centrality(metrics_df)
    return G, metrics_df, partition
