# Migration Hubs & Talent Pipelines
### A Network Topology of the Global Football Transfer Market

**CS-UH 2219E - Computational Social Science - NYUAD - Spring 2026**  
Team: Mahmoud Kassem - Aymane Omari - Fady John - Hariharan Janardhanan  
Instructor: Professor Talal Rahwan

---

## Project Structure

```text
FAHM/
|-- migration_hubs/
|   |-- main.py
|   |-- build_player_metadata.py
|   |-- config.py
|   |-- cache.py
|   |-- data_loader.py
|   |-- features.py
|   |-- eda.py
|   |-- network.py
|   |-- corridors.py
|   |-- stats.py
|   `-- milestone3.py
|-- inputs/
|   |-- player_metadata.csv
|   `-- player_metadata_template.csv
|-- cache/
`-- outputs/
    |-- figures/
    |-- tables/
    `-- results_summary.txt
```

---

## Installation

```bash
pip install statsbombpy pandas numpy networkx python-louvain \
            statsmodels scikit-learn matplotlib seaborn tqdm scipy
```

---

## Running The Pipeline

```bash
# Normal run (uses cache on subsequent runs)
python migration_hubs/main.py

# Force re-download of raw data
python migration_hubs/main.py --refresh-data

# Wipe the cache directory and exit
python migration_hubs/main.py --clear-cache
```

---

## Data Sources

| Source | What it provides |
|--------|------------------|
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Under-pressure share from real match events |
| [ewenme/transfers](https://github.com/ewenme/transfers) | Transfermarkt transfer records, one flat CSV per league |
| [salimt/football-datasets](https://github.com/salimt/football-datasets) | Public player profiles with citizenship, birth country, and EU status |

---

## Optional Player Metadata

The base Transfermarkt transfer file does **not** contain a clean player-level
race, ethnicity, or nationality field. The pipeline can optionally merge
player-level metadata from:

`inputs/player_metadata.csv`

A header-only template is included at:

`inputs/player_metadata_template.csv`

You can rebuild the metadata file from the public player-profiles source with:

```bash
python migration_hubs/build_player_metadata.py
python migration_hubs/build_player_metadata.py --limit 250
python migration_hubs/build_player_metadata.py --dest-leagues "Serie A,Bundesliga"
python migration_hubs/build_player_metadata.py --refresh-source
```

Required column:

- `player_name`

Recommended optional columns:

- `player_nationality`
- `player_birth_country`
- `player_birth_city`
- `is_eu`
- `turkey_link_flag`
- `bias_focus_group`
- `race_group`
- `ethnicity_group`
- `skin_tone_group`
- `metadata_source`
- `source_match_quality`
- `source_candidate_count`
- `source_candidate_count_post_birth_year`
- `source_player_names`
- `source_name_in_home_country`
- `source_main_position`
- `source_date_of_birth`
- `source_dest_leagues`
- `notes`

Important notes:

- The raw Transfermarkt `country` field is the club or league country, not the player's nationality. In the engineered dataset this is renamed `club_country`.
- The metadata builder uses normalized `player_name` and, when possible, approximate birth year from transfer age and season to disambiguate same-name profiles.
- Ambiguous same-name cases stay conservative: unresolved fields are left blank instead of guessed.
- The code does **not** infer race or ethnicity labels on its own.
- `milestone3.py` will use fetched nationality, birth-country, EU-status, and optional supplied group columns for descriptive gap checks, but it does not produce causal discrimination claims.

---

## Outputs

| File | Description |
|------|-------------|
| `fig1_eda_overview.png` | Annual counts, top leagues, fee distribution |
| `fig2_pressure.png` | Pressure index bar and pressure-gap histogram |
| `fig3_flow_heatmap.png` | League-to-league transfer flow |
| `fig4_network.png` | Directed transfer network |
| `fig5_centrality.png` | Betweenness centrality and net flow |
| `fig6_correlation_matrix.png` | Pearson correlation matrix |
| `fig7_ols_coefficients.png` | OLS coefficient plot with 95% CI |
| `fig8_psm.png` | PSM balance and propensity-score overlap |
| `fig9_position_heterogeneity.png` | Pressure-effect heterogeneity by player type |
| `fig10_feature_importance.png` | Explainable model feature importance |
| `fig11_common_support.png` | Common-support diagnostic |
| `fig12_group_gap.png` | Exploratory observed-vs-expected gap by supplied metadata group |
| `fig13_target_league_gap.png` | Bundesliga and Serie A destination gap checks from fetched metadata |
| `fig14_serie_a_selection_threshold.png` | Serie A selection-threshold check by EU status |
| `table0_statsbomb_pressure.csv` | StatsBomb-derived pressure metrics |
| `table1_dataset_summary.csv` | Dataset overview statistics |
| `table2_transfer_flow_matrix.csv` | League-to-league flow counts |
| `table3_network_metrics.csv` | Node-level centrality metrics |
| `table3b_global_network_stats.csv` | Graph-level statistics |
| `table4_louvain_communities.csv` | Community membership |
| `table5_career_corridors.csv` | Top career corridor paths |
| `table6_correlation_matrix.csv` | Pearson correlation matrix |
| `table7_ols_results.csv` | OLS coefficients, standard errors, p-values |
| `table8_psm_results.csv` | PSM ATT, confidence interval, bootstrap SE |
| `table8b_psm_balance.csv` | Covariate balance summary |
| `table9_position_heterogeneity.csv` | Position-specific OLS coefficients |
| `table10_feature_importance.csv` | Explainable model feature rankings |
| `table11_common_support.csv` | Common-support summary |
| `table12_group_gap.csv` | Exploratory group-gap summary |
| `table13_target_league_gap.csv` | Target-league gap summary |
| `table14_serie_a_selection_threshold.csv` | Serie A selection-threshold summary |
| `results_summary.txt` | Plain-text results summary |

---

## Outcome Variable Note

Transfermarkt records do not include `minutes_played`, so the dependent variable
used here is:

`log_transfer_fee = log(1 + fee_EUR)`

This is a proxy for player valuation used in football-economics work. If you
want to analyze playing time directly, merge an external player-season source
such as FBref or WhoScored on `(player_name, season)` before rerunning the
pipeline.
