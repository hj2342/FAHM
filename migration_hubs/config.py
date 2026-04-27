"""
config.py
---------
All shared constants, league maps, and pressure-index seeds.
Import this module everywhere instead of duplicating magic strings.
"""

import os

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Output directories ────────────────────────────────────────────────────────
DIR_FIGURES  = "outputs/figures"
DIR_TABLES   = "outputs/tables"
DIR_CACHE    = "cache"
DIR_INPUTS   = "inputs"

for _d in (DIR_FIGURES, DIR_TABLES, DIR_CACHE, DIR_INPUTS):
    os.makedirs(_d, exist_ok=True)

# ── Plot palette ──────────────────────────────────────────────────────────────
PAL = dict(
    teal    = "#00E5A0",
    purple  = "#7C5CBF",
    coral   = "#FF6B6B",
    gold    = "#F0B429",
    muted   = "#8B949E",
    bg      = "#0D1117",
    panel   = "#161B22",
)

PLOT_STYLE = {
    "figure.facecolor":  PAL["bg"],
    "axes.facecolor":    PAL["panel"],
    "axes.edgecolor":    PAL["muted"],
    "axes.labelcolor":   "white",
    "xtick.color":       PAL["muted"],
    "ytick.color":       PAL["muted"],
    "text.color":        "white",
    "grid.color":        "#2D333B",
    "grid.linewidth":    0.5,
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": PAL["bg"],
}

# ── StatsBomb competition name → internal key ─────────────────────────────────
SB_LEAGUE_MAP = {
    "Premier League":      "EPL",
    "La Liga":             "La Liga",
    "Ligue 1":             "Ligue 1",
    "Serie A":             "Serie A",
    "Major League Soccer": "MLS",
    "Eredivisie":          "Eredivisie",
}

# ── Transfermarkt CSV slug → internal key ─────────────────────────────────────
# URL: https://raw.githubusercontent.com/ewenme/transfers/master/data/{slug}.csv
TM_LEAGUE_MAP = {
    "premier-league":   "EPL",
    "primera-division": "La Liga",
    "1-bundesliga":     "Bundesliga",
    "serie-a":          "Serie A",
    "ligue-1":          "Ligue 1",
    "eredivisie":       "Eredivisie",
    "championship":     "Championship",
}

# ── Pressure Index seed values ────────────────────────────────────────────────
# StatsBomb-computed values overwrite these at runtime for leagues with data.
# [ESTIMATED] values come from published CIES/Opta aggregate reports.
PRESSURE_INDEX: dict[str, float] = {
    "EPL":          21.3,    # overwritten by StatsBomb
    "La Liga":      20.3,    # overwritten by StatsBomb
    "Bundesliga":   22.5,    # [ESTIMATED] — not in StatsBomb open data
    "Serie A":      21.2,    # overwritten by StatsBomb
    "Ligue 1":      20.8,    # overwritten by StatsBomb
    "Eredivisie":   19.8,    # [ESTIMATED]
    "MLS":          18.9,    # overwritten by StatsBomb (partial)
    "Championship": 25.1,    # [ESTIMATED] — high-intensity English 2nd tier
    "Liga NOS":     19.2,    # [ESTIMATED]
}

# Tracks which values came from real StatsBomb data vs estimates
PRESSURE_SOURCE: dict[str, str] = {k: "estimated" for k in PRESSURE_INDEX}
