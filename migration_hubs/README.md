# Migration Hubs & Talent Pipelines
### A Network Topology of the Global Football Transfer Market

**CS-UH 2219E · Computational Social Science · NYUAD · Spring 2026**  
Team: Mahmoud Kassem · Aymane Omari · Fady John · Hariharan Janardhanan  
Instructor: Professor Talal Rahwan

---

## Project structure

```
migration_hubs/
├── main.py            ← entry point; run this
├── config.py          ← constants, league maps, pressure-index seeds
├── cache.py           ← disk-caching utilities (pickle-based)
├── data_loader.py     ← StatsBomb + Transfermarkt loaders (cached)
├── features.py        ← cleaning & feature engineering
├── eda.py             ← EDA figures 1–3
├── network.py         ← SNA figures 4–5
├── corridors.py       ← career corridor extraction
├── stats.py           ← OLS, PSM, figures 6–8
├── cache/             ← auto-created; holds .pkl cache files
└── outputs/
    ├── figures/       ← 8 PNG figures
    ├── tables/        ← 9 CSV tables
    └── results_summary.txt
```

---

## Installation

```bash
pip install statsbombpy pandas numpy networkx python-louvain \
            statsmodels scikit-learn matplotlib seaborn \
            requests tqdm scipy
```

---

## Running the script

```bash
# Normal run (uses cache on subsequent runs — no repeated downloads)
python main.py

# Force re-download of all raw data (ignores existing cache)
python main.py --refresh-data

# Wipe the cache directory and exit
python main.py --clear-cache
```

---

## Data sources

| Source | What it provides |
|--------|-----------------|
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Under-pressure % per event by league (Pressure Index) |
| [ewenme/transfers](https://github.com/ewenme/transfers) | Transfermarkt transfer records, one flat CSV per league |

---

## Caching behaviour

On **first run** each loader downloads data from the internet and writes
a `.pkl` file to `./cache/`.  
On **subsequent runs** the loaders read from cache — no network calls are
made, and the script runs in seconds.

Pass `--refresh-data` to force a full re-download (e.g. after the
upstream Transfermarkt data is updated).

---

## Outputs

| File | Description |
|------|-------------|
| `fig1_eda_overview.png` | Annual counts · top leagues · fee distribution |
| `fig2_pressure.png` | Pressure index bar · pressure-gap histogram |
| `fig3_flow_heatmap.png` | League-to-league transfer flow |
| `fig4_network.png` | Directed transfer network (Louvain communities) |
| `fig5_centrality.png` | Betweenness centrality + net flow |
| `fig6_correlation_matrix.png` | Pearson correlation matrix |
| `fig7_ols_coefficients.png` | OLS coefficient plot with 95% CI |
| `fig8_psm.png` | PSM balance + propensity-score overlap |
| `table0_statsbomb_pressure.csv` | StatsBomb-derived pressure metrics |
| `table1_dataset_summary.csv` | Dataset overview statistics |
| `table2_transfer_flow_matrix.csv` | League-to-league flow counts |
| `table3_network_metrics.csv` | Node-level centrality metrics |
| `table3b_global_network_stats.csv` | Graph-level statistics |
| `table4_louvain_communities.csv` | Community membership |
| `table5_career_corridors.csv` | Top career corridor paths |
| `table6_correlation_matrix.csv` | Pearson correlation matrix |
| `table7_ols_results.csv` | OLS coefficients, SEs, p-values |
| `table8_psm_results.csv` | PSM ATT, CI, bootstrap SE |
| `table8b_psm_balance.csv` | Covariate balance (SMD before/after) |
| `results_summary.txt` | Plain-text results summary |

---

## Key bugs fixed

| Bug | Fix |
|-----|-----|
| `TypeError: 'RegressionResultsWrapper' not subscriptable` | `plot_ols_coefficients()` now receives the model object directly, not wrapped in a list |
| OLS coefficients labelled `x1, x2, …` | Design matrix kept as a named `DataFrame` throughout; `sm.add_constant()` called on the DataFrame, not a numpy array |
| No caching — data re-downloaded every run | `cache.py` persists raw data to `./cache/` on first download; subsequent runs load from disk |
| Outputs saved collectively at end | Every figure and table is saved immediately after generation |
| `bool` dtype crash in VIF / PSM dummies | All dummy matrices cast to `dtype=float` at creation time |

---

## Outcome variable note

Transfermarkt records do not include `minutes_played`.  
The dependent variable is therefore **`log_transfer_fee = log(1 + fee_EUR)`**,
a widely-used proxy for perceived player quality in football economics
(Müller et al. 2017; Sæbø & Hvattum 2019).

To use `minutes_played` directly, merge an FBref or WhoScored player-season
dataset on `(player_name, season)` before running `stats.py`.
