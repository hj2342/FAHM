"""
milestone3.py
-------------
Exploratory analyses for milestone 3 that stay within observed variables only.

Figure 9  : Position-wise heterogeneity in the origin-pressure coefficient
Figure 10 : Explainable model feature importance for transfer fees
Figure 11 : Common-support diagnostic for propensity-score matching
Figure 12 : Group-level step-up gap using optional player metadata
Figure 13 : Target-league gap checks using fetched player metadata

Tables
------
table9_position_heterogeneity.csv
table10_feature_importance.csv
table11_common_support.csv
table12_group_gap.csv
table13_target_league_gap.csv
"""

import warnings
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DIR_FIGURES, DIR_TABLES, PAL, PLOT_STYLE, RANDOM_SEED

warnings.filterwarnings("ignore")
plt.rcParams.update(PLOT_STYLE)


def _position_group(val: object) -> str:
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


def _ensure_position_group(df: pd.DataFrame) -> pd.DataFrame:
    """Create position_group on the fly if the feature layer predates it."""
    if "position_group" in df.columns:
        return df

    out = df.copy()
    if "position" in out.columns:
        out["position_group"] = out["position"].apply(_position_group)
    else:
        out["position_group"] = "Unknown"
    return out


def _normalize_meta_text(val: object) -> str:
    """Normalize metadata text for robust matching on country labels."""
    if pd.isna(val):
        return ""

    s = unicodedata.normalize("NFKD", str(val))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_boolish(val: object):
    """Interpret common string/bool encodings used in the optional metadata file."""
    if pd.isna(val):
        return pd.NA
    if isinstance(val, bool):
        return val

    s = str(val).strip().lower()
    if s in {"true", "1", "yes", "y", "eu"}:
        return True
    if s in {"false", "0", "no", "n", "non-eu", "non eu"}:
        return False
    return pd.NA


def _fit_position_ols(reg: pd.DataFrame):
    """Fit the position-specific OLS used in the heterogeneity figure."""
    if len(reg) < 80 or reg["dest_league"].nunique() < 2:
        return None

    dest_dummies = pd.get_dummies(
        reg["dest_league"], prefix="dest", drop_first=True, dtype=float
    )
    X = pd.concat(
        [reg[["origin_pressure", "season_year"]].astype(float), dest_dummies],
        axis=1,
    )
    X = sm.add_constant(X)
    y = reg["log_transfer_fee"].astype(float)
    clusters = reg["dest_league"]

    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    clusters = clusters.loc[valid_idx]

    if len(y) < 80:
        return None

    try:
        return sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": clusters},
        )
    except Exception:
        return sm.OLS(y, X).fit(cov_type="HC1")


def run_position_heterogeneity(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Compare the origin-pressure coefficient across broad player types.

    This directly addresses the milestone-3 suggestion to check whether the
    pressure story differs for defenders, midfielders, attackers, or keepers.
    """
    df = _ensure_position_group(df)
    required = [
        "position_group",
        "log_transfer_fee",
        "origin_pressure",
        "season_year",
        "dest_league",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[M3] Position heterogeneity skipped - missing columns: {missing}")
        return None

    pos_df = df[required].dropna().copy()
    counts = pos_df["position_group"].value_counts()
    eligible = counts[counts >= 150].index.tolist()

    rows = []
    for group in eligible:
        reg = pos_df[pos_df["position_group"] == group].copy()
        model = _fit_position_ols(reg)
        if model is None or "origin_pressure" not in model.params.index:
            continue

        ci = model.conf_int().loc["origin_pressure"]
        rows.append({
            "position_group": group,
            "n_obs": int(model.nobs),
            "coef": float(model.params["origin_pressure"]),
            "std_err": float(model.bse["origin_pressure"]),
            "p_value": float(model.pvalues["origin_pressure"]),
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
            "adj_r2": float(model.rsquared_adj),
        })

    if not rows:
        print("[M3] Position heterogeneity skipped - no subgroup models converged.")
        return None

    results = pd.DataFrame(rows).sort_values("coef").reset_index(drop=True)
    csv_out = f"{DIR_TABLES}/table9_position_heterogeneity.csv"
    results.to_csv(csv_out, index=False)
    print(f"[✓] {csv_out}")

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(results))
    colors = [
        PAL["teal"] if coef >= 0 else PAL["coral"]
        for coef in results["coef"]
    ]

    ax.barh(
        y_pos,
        results["coef"],
        color=colors,
        alpha=0.9,
        edgecolor="none",
    )
    ax.errorbar(
        results["coef"],
        y_pos,
        xerr=[
            results["coef"] - results["ci_low"],
            results["ci_high"] - results["coef"],
        ],
        fmt="none",
        color="white",
        capsize=4,
        linewidth=1.8,
    )
    ax.axvline(0, color=PAL["muted"], linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results["position_group"])
    ax.set_xlabel("Coefficient on origin_pressure (beta)")
    ax.set_title(
        "Figure 9: Pressure Effect by Player Type\n"
        "Separate OLS within each broad position group",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.xaxis.grid(True)

    x_floor, x_ceiling = ax.get_xlim()
    x_span = x_ceiling - x_floor
    for i, row in results.iterrows():
        star = (
            "***" if row["p_value"] < 0.001 else
            "**" if row["p_value"] < 0.01 else
            "*" if row["p_value"] < 0.05 else
            "ns"
        )
        label_x = min(row["ci_high"] + 0.03 * x_span, x_ceiling - 0.02 * x_span)
        ax.text(
            label_x,
            i,
            f"n={row['n_obs']:,}  {star}",
            va="center",
            ha="left",
            fontsize=9,
            color="white",
        )

    plt.tight_layout()
    fig_out = f"{DIR_FIGURES}/fig9_position_heterogeneity.png"
    plt.savefig(fig_out)
    plt.close()
    print(f"[✓] {fig_out}\n")
    return results


def run_explainable_model(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Fit an explainable fee model and rank the observed features by importance.

    We use permutation importance because it works with the current local stack
    and keeps the interpretation at the original feature level.
    """
    df = _ensure_position_group(df)
    required = [
        "log_transfer_fee",
        "age",
        "position_group",
        "dest_league",
        "origin_league",
        "season_year",
        "origin_pressure",
        "dest_pressure",
        "pressure_gap",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[M3] Explainable model skipped - missing columns: {missing}")
        return None

    ml_df = df[required].copy()
    ml_df["age"] = pd.to_numeric(ml_df["age"], errors="coerce")
    ml_df = ml_df[ml_df["log_transfer_fee"].notna()].copy()
    ml_df = ml_df[ml_df["position_group"] != "Unknown"].copy()

    if len(ml_df) < 500:
        print(f"[M3] Explainable model skipped - only {len(ml_df)} usable rows.")
        return None

    numeric = ["age", "season_year", "origin_pressure", "dest_pressure", "pressure_gap"]
    categorical = ["position_group", "dest_league", "origin_league"]

    X = ml_df[numeric + categorical]
    y = ml_df["log_transfer_fee"].astype(float)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("forest", RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    csv_out = f"{DIR_TABLES}/table10_feature_importance.csv"
    importance_df.to_csv(csv_out, index=False)
    print(f"[✓] {csv_out}")

    top = importance_df.head(8).sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        color=PAL["purple"],
        alpha=0.9,
        edgecolor="none",
    )
    ax.set_xlabel("Permutation importance (mean decrease in out-of-sample R^2)")
    ax.set_title(
        "Figure 10: Explainable Fee Model\n"
        f"Random forest feature importance  |  Test R^2 = {r2:.3f}  |  MAE = {mae:.3f}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.xaxis.grid(True)
    plt.tight_layout()

    fig_out = f"{DIR_FIGURES}/fig10_feature_importance.png"
    plt.savefig(fig_out)
    plt.close()
    print(f"[✓] {fig_out}\n")
    return importance_df


def _available_group_column(df: pd.DataFrame) -> str | None:
    """Return the highest-priority optional group column available."""
    for col in [
        "bias_focus_group",
        "race_group",
        "ethnicity_group",
        "skin_tone_group",
        "player_nationality",
        "player_birth_country",
    ]:
        if col in df.columns:
            return col
    return None


def run_group_gap_analysis(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Exploratory gap analysis for optional protected-group metadata.

    Outcome
    -------
    step_up_move = 1 if pressure_gap > 0

    Interpretation
    --------------
    We fit a model *without* the group label, estimate each player's expected
    chance of a step-up move from observed football features, and then compare
    observed vs expected rates by group. This is descriptive and not a proof of
    discrimination, but it gives the team a defensible way to explore whether a
    supplied group variable is associated with systematic under-attainment.
    """
    df = _ensure_position_group(df)
    group_col = _available_group_column(df)
    if group_col is None:
        print("[M3] Group-gap analysis skipped - no optional metadata group column found.")
        return None

    required = [
        group_col,
        "pressure_gap",
        "origin_league",
        "origin_pressure",
        "position_group",
        "season_year",
        "age",
        "log_transfer_fee",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[M3] Group-gap analysis skipped - missing columns: {missing}")
        return None

    gap_df = df[required].copy()
    gap_df["age"] = pd.to_numeric(gap_df["age"], errors="coerce")
    gap_df = gap_df[gap_df["pressure_gap"].notna()].copy()
    gap_df[group_col] = gap_df[group_col].astype(str).str.strip()
    gap_df = gap_df[
        gap_df[group_col].notna() &
        gap_df[group_col].ne("") &
        gap_df[group_col].ne("nan") &
        gap_df[group_col].ne("Unknown")
    ].copy()
    gap_df["step_up_move"] = (gap_df["pressure_gap"] > 0).astype(int)

    if len(gap_df) < 200:
        print(f"[M3] Group-gap analysis skipped - only {len(gap_df)} usable rows.")
        return None

    counts = gap_df[group_col].value_counts()
    keep_groups = counts[counts >= 40].index.tolist()
    if len(keep_groups) > 10:
        keep_groups = counts.head(10).index.tolist()
    gap_df = gap_df[gap_df[group_col].isin(keep_groups)].copy()

    if gap_df[group_col].nunique() < 2:
        print(f"[M3] Group-gap analysis skipped - only one populated group in {group_col}.")
        return None

    numeric = ["age", "season_year", "origin_pressure"]
    categorical = ["position_group", "origin_league"]
    X = gap_df[numeric + categorical]
    y = gap_df["step_up_move"].astype(int)
    if y.nunique() < 2:
        print("[M3] Group-gap analysis skipped - step_up_move has no variation.")
        return None

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("logit", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
    ])
    model.fit(X, y)
    gap_df["expected_step_up_prob"] = model.predict_proba(X)[:, 1]

    summary = (
        gap_df.groupby(group_col)
              .agg(
                  n_obs=("step_up_move", "size"),
                  observed_step_up_rate=("step_up_move", "mean"),
                  expected_step_up_rate=("expected_step_up_prob", "mean"),
                  mean_log_transfer_fee=("log_transfer_fee", "mean"),
              )
              .reset_index()
    )
    summary["gap_pp"] = (
        (summary["observed_step_up_rate"] - summary["expected_step_up_rate"]) * 100
    )
    summary = summary.sort_values("gap_pp").reset_index(drop=True)

    csv_out = f"{DIR_TABLES}/table12_group_gap.csv"
    summary.to_csv(csv_out, index=False)
    print(f"[✓] {csv_out}")

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [
        PAL["teal"] if v >= 0 else PAL["coral"]
        for v in summary["gap_pp"]
    ]
    ax.barh(summary[group_col], summary["gap_pp"], color=colors, edgecolor="none", alpha=0.9)
    ax.axvline(0, color=PAL["muted"], linestyle="--", linewidth=1)
    ax.set_xlabel("Observed - expected step-up rate (percentage points)")
    ax.set_title(
        f"Figure 12: Exploratory Group Gap by {group_col}\n"
        "Expected step-up probability estimated without the group label",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.xaxis.grid(True)

    x_floor, x_ceiling = ax.get_xlim()
    x_span = x_ceiling - x_floor
    for i, row in summary.iterrows():
        label_x = (
            row["gap_pp"] + 0.02 * x_span
            if row["gap_pp"] >= 0
            else row["gap_pp"] - 0.02 * x_span
        )
        ax.text(
            label_x,
            i,
            f"n={int(row['n_obs'])}",
            va="center",
            ha="left" if row["gap_pp"] >= 0 else "right",
            fontsize=9,
            color="white",
        )

    plt.tight_layout()
    fig_out = f"{DIR_FIGURES}/fig12_group_gap.png"
    plt.savefig(fig_out)
    plt.close()
    print(f"[✓] {fig_out}")
    print(
        f"[M3] Group-gap analysis used '{group_col}' as the supplied metadata field. "
        "Interpret the gaps as descriptive, not causal.\n"
    )
    return summary


def run_target_league_bias_checks(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Run targeted destination-league gap checks from fetched player metadata.

    We estimate each player's expected probability of moving into a target
    destination league without using the group label itself, then compare
    observed vs expected destination rates by group. This stays descriptive:
    systematic gaps are not causal proof of discrimination.
    """
    df = _ensure_position_group(df)
    required = [
        "age",
        "season_year",
        "origin_pressure",
        "position_group",
        "origin_league",
        "dest_league",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[M3] Target-league bias checks skipped - missing columns: {missing}")
        return None

    work = df[required + [c for c in [
        "player_nationality",
        "player_birth_country",
        "is_eu",
        "turkey_link_flag",
    ] if c in df.columns]].copy()
    scenarios = []

    if "turkey_link_flag" in work.columns:
        turkey_flag = work["turkey_link_flag"].map(_parse_boolish)
    else:
        nat_norm = (
            work["player_nationality"].map(_normalize_meta_text)
            if "player_nationality" in work.columns else
            pd.Series("", index=work.index)
        )
        birth_norm = (
            work["player_birth_country"].map(_normalize_meta_text)
            if "player_birth_country" in work.columns else
            pd.Series("", index=work.index)
        )
        turkey_flag = (
            nat_norm.str.contains(r"\bturkiye\b|\bturkey\b", regex=True, na=False) |
            birth_norm.str.contains(r"\bturkiye\b|\bturkey\b", regex=True, na=False)
        )
    turkey_true = int(pd.Series(turkey_flag).fillna(False).astype(bool).sum())
    if turkey_true >= 40:
        scenarios.append({
            "analysis_name": "bundesliga_turkey_link",
            "target_league": "Bundesliga",
            "flag": pd.Series(turkey_flag, index=work.index).fillna(False).astype(bool),
            "positive_label": "Turkey-linked",
            "negative_label": "Other",
            "title": "Bundesliga destination gap by Turkey-linked status",
        })

    if "is_eu" in work.columns:
        eu_flag = work["is_eu"].map(_parse_boolish)
        non_eu_n = int((eu_flag == False).sum())
        eu_n = int((eu_flag == True).sum())
        if non_eu_n >= 100 and eu_n >= 500:
            scenarios.append({
                "analysis_name": "serie_a_eu_status",
                "target_league": "Serie A",
                "flag": (eu_flag == False),
                "positive_label": "Non-EU",
                "negative_label": "EU",
                "title": "Serie A destination gap by EU status",
            })

    if not scenarios:
        print("[M3] Target-league bias checks skipped - no metadata groups with enough support.")
        return None

    numeric = ["age", "season_year", "origin_pressure"]
    categorical = ["position_group", "origin_league"]
    rows = []
    plot_specs = []

    for scenario in scenarios:
        use = work.copy()
        use["group_flag"] = scenario["flag"]
        use = use[use["group_flag"].notna()].copy()
        use["group_label"] = use["group_flag"].map({
            False: scenario["negative_label"],
            True: scenario["positive_label"],
        })
        counts = use["group_label"].value_counts()
        if len(counts) < 2 or counts.min() < 40:
            continue

        use["to_target_league"] = (use["dest_league"] == scenario["target_league"]).astype(int)
        if use["to_target_league"].nunique() < 2:
            continue

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), numeric),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]), categorical),
            ]
        )
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("logit", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
        ])
        model.fit(use[numeric + categorical], use["to_target_league"].astype(int))
        use["expected_target_prob"] = model.predict_proba(use[numeric + categorical])[:, 1]

        summary = (
            use.groupby("group_label")
               .agg(
                   n_obs=("to_target_league", "size"),
                   observed_target_rate=("to_target_league", "mean"),
                   expected_target_rate=("expected_target_prob", "mean"),
               )
               .reset_index()
        )
        summary["gap_pp"] = (
            (summary["observed_target_rate"] - summary["expected_target_rate"]) * 100
        )
        summary["analysis_name"] = scenario["analysis_name"]
        summary["target_league"] = scenario["target_league"]
        summary["group_type"] = scenario["title"]
        summary["overall_target_rate"] = use["to_target_league"].mean()
        summary["group_label"] = pd.Categorical(
            summary["group_label"],
            categories=[scenario["negative_label"], scenario["positive_label"]],
            ordered=True,
        )
        summary = summary.sort_values("group_label").reset_index(drop=True)
        plot_specs.append((scenario, summary.copy()))
        rows.extend(summary.to_dict("records"))

    if not rows:
        print("[M3] Target-league bias checks skipped - no scenario produced usable output.")
        return None

    results = pd.DataFrame(rows)
    csv_out = f"{DIR_TABLES}/table13_target_league_gap.csv"
    results.to_csv(csv_out, index=False)
    print(f"[✓] {csv_out}")

    fig, axes = plt.subplots(len(plot_specs), 1, figsize=(11, 4.5 * len(plot_specs)))
    if len(plot_specs) == 1:
        axes = [axes]

    for ax, (scenario, summary) in zip(axes, plot_specs):
        colors = [
            PAL["teal"] if v >= 0 else PAL["coral"]
            for v in summary["gap_pp"]
        ]
        ax.barh(summary["group_label"].astype(str), summary["gap_pp"], color=colors, edgecolor="none", alpha=0.9)
        ax.axvline(0, color=PAL["muted"], linestyle="--", linewidth=1)
        ax.set_xlabel("Observed - expected destination rate (percentage points)")
        ax.set_title(
            scenario["title"] + "\nExpected destination rate estimated without the group label",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.xaxis.grid(True)

        x_floor, x_ceiling = ax.get_xlim()
        x_span = x_ceiling - x_floor
        for i, row in summary.iterrows():
            label_x = (
                row["gap_pp"] + 0.02 * x_span
                if row["gap_pp"] >= 0
                else row["gap_pp"] - 0.02 * x_span
            )
            ax.text(
                label_x,
                i,
                f"n={int(row['n_obs'])}",
                va="center",
                ha="left" if row["gap_pp"] >= 0 else "right",
                fontsize=9,
                color="white",
            )

    fig.suptitle(
        "Figure 13: Target-League Gap Checks from Fetched Player Metadata",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig_out = f"{DIR_FIGURES}/fig13_target_league_gap.png"
    plt.savefig(fig_out)
    plt.close()
    print(f"[✓] {fig_out}")
    print(
        "[M3] Target-league gap checks are descriptive only. They use fetched "
        "nationality, birth-country, and EU-status metadata; they do not infer race labels.\n"
    )
    return results


def _match_on_scores(psm_df: pd.DataFrame) -> dict[str, float] | None:
    """1:1 nearest-neighbor matching on precomputed logit propensity scores."""
    T = psm_df["high_pressure_treat"].to_numpy()
    y = psm_df["log_transfer_fee"].to_numpy()
    logit_ps = psm_df["logit_ps"].to_numpy()

    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    if len(treated_idx) == 0 or len(control_idx) == 0:
        return None

    caliper = 0.1 * np.std(logit_ps)
    if not np.isfinite(caliper) or caliper <= 0:
        return None

    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nn.fit(logit_ps[control_idx].reshape(-1, 1))
    distances, matched_pos = nn.kneighbors(logit_ps[treated_idx].reshape(-1, 1))
    within = distances.flatten() <= caliper
    n_pairs = int(within.sum())
    if n_pairs < 20:
        return None

    matched_controls = control_idx[matched_pos.flatten()[within]]
    matched_treated = treated_idx[within]
    att = float((y[matched_treated] - y[matched_controls]).mean())

    return {
        "att": att,
        "n_pairs": n_pairs,
        "match_rate": float(n_pairs / len(treated_idx)),
        "caliper": float(caliper),
    }


def run_common_support_diagnostic(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Diagnose whether the PSM design has usable common support.

    The overlap figure is descriptive. Any trimmed ATT is saved as exploratory,
    not as a clean causal estimate.
    """
    required = [
        "log_transfer_fee",
        "high_pressure_treat",
        "dest_league",
        "season_year",
        "origin_pressure",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[M3] Common-support diagnostic skipped - missing columns: {missing}")
        return None

    psm_df = df[required].dropna().copy()
    if len(psm_df) < 100:
        print(f"[M3] Common-support diagnostic skipped - only {len(psm_df)} rows.")
        return None

    dest_dummies = pd.get_dummies(
        psm_df["dest_league"], prefix="dest", drop_first=True, dtype=float
    )
    X_df = pd.concat([psm_df[["season_year"]].astype(float), dest_dummies], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values.astype(float))

    logit = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
    logit.fit(X_scaled, psm_df["high_pressure_treat"].astype(int))
    ps = logit.predict_proba(X_scaled)[:, 1]
    clipped = np.clip(ps, 1e-6, 1 - 1e-6)

    psm_df["ps"] = ps
    psm_df["logit_ps"] = np.log(clipped / (1 - clipped))

    treated_ps = psm_df.loc[psm_df["high_pressure_treat"] == 1, "ps"]
    control_ps = psm_df.loc[psm_df["high_pressure_treat"] == 0, "ps"]

    lower = float(max(treated_ps.min(), control_ps.min()))
    upper = float(min(treated_ps.max(), control_ps.max()))
    has_overlap = lower < upper

    psm_df["in_common_support"] = (
        has_overlap & psm_df["ps"].between(lower, upper, inclusive="both")
    )
    trimmed = psm_df[psm_df["in_common_support"]].copy()
    exploratory = _match_on_scores(trimmed) if has_overlap else None

    summary = pd.DataFrame({
        "metric": [
            "all_rows",
            "treated_rows",
            "control_rows",
            "common_support_lower",
            "common_support_upper",
            "rows_in_support",
            "share_in_support",
            "treated_in_support",
            "control_in_support",
            "exploratory_trimmed_pairs",
            "exploratory_trimmed_att",
        ],
        "value": [
            len(psm_df),
            int((psm_df["high_pressure_treat"] == 1).sum()),
            int((psm_df["high_pressure_treat"] == 0).sum()),
            round(lower, 6) if has_overlap else "N/A",
            round(upper, 6) if has_overlap else "N/A",
            len(trimmed),
            round(len(trimmed) / len(psm_df), 4),
            int((trimmed["high_pressure_treat"] == 1).sum()),
            int((trimmed["high_pressure_treat"] == 0).sum()),
            exploratory["n_pairs"] if exploratory else "N/A",
            round(exploratory["att"], 4) if exploratory else "N/A",
        ],
    })

    csv_out = f"{DIR_TABLES}/table11_common_support.csv"
    summary.to_csv(csv_out, index=False)
    print(f"[✓] {csv_out}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.hist(
        control_ps,
        bins=40,
        alpha=0.6,
        density=True,
        color=PAL["coral"],
        edgecolor="none",
        label="Control (low pressure)",
    )
    ax.hist(
        treated_ps,
        bins=40,
        alpha=0.6,
        density=True,
        color=PAL["teal"],
        edgecolor="none",
        label="Treated (high pressure)",
    )
    if has_overlap:
        ax.axvspan(lower, upper, color=PAL["gold"], alpha=0.15, label="Common support")
        ax.axvline(lower, color=PAL["gold"], linestyle="--", linewidth=1.2)
        ax.axvline(upper, color=PAL["gold"], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Propensity score")
    ax.set_ylabel("Density")
    ax.set_title("Propensity-score overlap")
    ax.legend(framealpha=0.3)
    ax.yaxis.grid(True)

    counts_plot = pd.DataFrame({
        "group": ["Control", "Treated"],
        "full_sample": [
            int((psm_df["high_pressure_treat"] == 0).sum()),
            int((psm_df["high_pressure_treat"] == 1).sum()),
        ],
        "inside_support": [
            int((trimmed["high_pressure_treat"] == 0).sum()),
            int((trimmed["high_pressure_treat"] == 1).sum()),
        ],
    })
    counts_long = counts_plot.melt(
        id_vars="group",
        value_vars=["full_sample", "inside_support"],
        var_name="sample",
        value_name="count",
    )

    ax = axes[1]
    sns.barplot(
        data=counts_long,
        x="group",
        y="count",
        hue="sample",
        palette=[PAL["muted"], PAL["gold"]],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Number of observations")
    subtitle = (
        f"Exploratory trimmed ATT = {exploratory['att']:+.4f}"
        if exploratory else
        "Exploratory trimmed ATT unavailable"
    )
    ax.set_title(
        "Sample retained after common-support trimming\n"
        f"{subtitle}",
    )
    ax.yaxis.grid(True)
    ax.legend(framealpha=0.3, title="")

    fig.suptitle(
        "Figure 11: Common-Support Diagnostic for PSM",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    fig_out = f"{DIR_FIGURES}/fig11_common_support.png"
    plt.savefig(fig_out)
    plt.close()
    print(f"[✓] {fig_out}\n")
    return summary


def run_milestone3_analysis(df: pd.DataFrame) -> None:
    """Run all milestone-3 exploratory analyses that the observed data supports."""
    print("\n-- MILESTONE 3 EXPLORATION ------------------------------------------")
    run_position_heterogeneity(df)
    run_explainable_model(df)
    run_common_support_diagnostic(df)
    run_group_gap_analysis(df)
    run_target_league_bias_checks(df)
    print(
        "[M3] Group-gap analysis runs only on externally supplied metadata "
        "columns; this code does not infer race or ethnicity labels.\n"
    )
