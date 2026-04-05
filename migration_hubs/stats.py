"""
stats.py
--------
Statistical analysis: correlation matrix, OLS regression, PSM, and
coefficient plots.  Every table and figure is saved immediately after
it is produced.

Key fixes vs original script
-----------------------------
1. OLS design matrix kept as a named DataFrame throughout so that
   model.params / model.pvalues carry meaningful column labels
   (origin_pressure, season_year, dest_EPL, …) instead of x1, x2 …

2. plot_ols_coefficients() receives the model object directly (not a
   list); the caller in main.py passes the single returned model.

3. VIF check uses float-cast dummies to avoid the
   "could not convert string to float" crash with bool dtype.

4. Clustered SEs are applied correctly even when the number of clusters
   equals the number of dummy levels.

Figures
-------
Figure 6 : Pearson correlation matrix
Figure 7 : OLS coefficient plot with 95% CI
Figure 8 : PSM balance + propensity-score overlap

Tables
------
table6_correlation_matrix.csv
table7_ols_results.csv
table8_psm_results.csv
table8b_psm_balance.csv
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from config import PAL, PLOT_STYLE, RANDOM_SEED, DIR_FIGURES, DIR_TABLES

warnings.filterwarnings("ignore")
plt.rcParams.update(PLOT_STYLE)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Correlation matrix
# ══════════════════════════════════════════════════════════════════════════════

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Pearson correlation heatmap of key numerical features — saved as fig6."""
    num_cols = [c for c in [
        "log_transfer_fee", "origin_pressure", "dest_pressure",
        "pressure_gap", "season_year",
    ] if c in df.columns]
    corr_df = df[num_cols].dropna()
    if len(corr_df) < 10:
        print("[WARN] Too few rows for correlation matrix — skipping.")
        return

    corr = corr_df.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr, ax=ax, cmap=cmap, center=0,
        annot=True, fmt=".2f",
        linewidths=0.5, linecolor="#2D333B",
        vmin=-1, vmax=1,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax.set_title(
        f"Figure 6: Pearson Correlation Matrix (n={len(corr_df):,})",
        fontsize=13, fontweight="bold", pad=14,
    )
    plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()

    out     = f"{DIR_FIGURES}/fig6_correlation_matrix.png"
    csv_out = f"{DIR_TABLES}/table6_correlation_matrix.csv"
    plt.savefig(out); plt.close()
    corr.round(4).to_csv(csv_out)
    print(f"[✓] {out}")
    print(f"[✓] {csv_out}\n")


# ══════════════════════════════════════════════════════════════════════════════
# OLS Regression  (FIXED)
# ══════════════════════════════════════════════════════════════════════════════

def run_ols_regression(df: pd.DataFrame):
    """
    OLS with destination-league fixed effects and clustered standard errors.

    DV       : log_transfer_fee
    IV       : origin_pressure
    FE       : destination league dummies (drop_first=True)
    Control  : season_year
    Cluster  : dest_league

    The design matrix is kept as a *named* DataFrame so that
    model.params carries human-readable labels, not x1 / x2 / …

    Returns
    -------
    model : statsmodels RegressionResultsWrapper  (or None on failure)
    """
    print("\n" + "=" * 65)
    print("OLS REGRESSION — DV: log_transfer_fee")
    print("Clustered SEs by dest_league")
    print("=" * 65)

    reg = df[
        df["log_transfer_fee"].notna() &
        df["origin_pressure"].notna() &
        df["season_year"].notna()
    ].copy()

    if len(reg) < 100:
        print(f"[ERROR] Only {len(reg)} usable observations — OLS skipped.")
        return None

    # ── Build named design matrix (keep as DataFrame!) ────────────────────────
    dest_dummies = pd.get_dummies(
        reg["dest_league"],
        prefix="dest",
        drop_first=True,
        dtype=float,          # avoid bool-dtype crash
    )

    X = pd.concat(
        [reg[["origin_pressure", "season_year"]].astype(float), dest_dummies],
        axis=1,
    )
    X = sm.add_constant(X)   # adds column named 'const'

    y = reg["log_transfer_fee"].astype(float)
    clusters = reg["dest_league"]

    # Drop any rows that became NaN during concat/cast
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X        = X.loc[valid_idx]
    y        = y.loc[valid_idx]
    clusters = clusters.loc[valid_idx]

    print(f"[OLS] n = {len(y):,}  |  features = {X.shape[1]}")

    # ── Fit clustered OLS ─────────────────────────────────────────────────────
    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": clusters},
    )
    print(model.summary())

    # Save results: named params thanks to DataFrame design matrix
    results_df = pd.DataFrame({
        "coef":    model.params,
        "std_err": model.bse,
        "t_stat":  model.tvalues,
        "p_value": model.pvalues,
        "ci_low":  model.conf_int()[0],
        "ci_high": model.conf_int()[1],
    })
    csv_out = f"{DIR_TABLES}/table7_ols_results.csv"
    results_df.to_csv(csv_out)
    print(f"\n[✓] {csv_out}\n")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — OLS coefficient plot  (FIXED: receives model, not list)
# ══════════════════════════════════════════════════════════════════════════════

def plot_ols_coefficients(model) -> None:
    """
    Coefficient plot for the OLS model with 95% CI.

    Parameters
    ----------
    model : statsmodels RegressionResultsWrapper returned by run_ols_regression.
            Pass None to skip silently.
    """
    if model is None:
        print("[WARN] No OLS model to plot — skipping fig7.")
        return

    # Show only substantive coefficients — exclude destination dummies + const
    keep   = [c for c in model.params.index
              if not c.startswith("dest_") and c != "const"]
    if not keep:
        print("[WARN] No non-dummy coefficients to plot.")
        return

    params = model.params[keep]
    conf   = model.conf_int().loc[keep]

    fig, ax = plt.subplots(figsize=(9, max(4, len(params) * 1.2)))
    y_pos   = range(len(params))
    colors  = [PAL["teal"] if v > 0 else PAL["coral"] for v in params.values]

    ax.barh(y_pos, params.values, color=colors, alpha=0.85,
            edgecolor="none", height=0.55)
    ax.errorbar(
        params.values, y_pos,
        xerr=[params.values - conf[0].values,
               conf[1].values - params.values],
        fmt="none", color="white", capsize=5, linewidth=2,
    )
    ax.axvline(0, color=PAL["muted"], linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params.index, fontsize=11)
    ax.set_xlabel("Coefficient (β) with 95% CI")
    ax.set_title(
        f"Figure 7: OLS Coefficients — Full Model\n"
        f"DV: log_transfer_fee  |  "
        f"Adj. R² = {model.rsquared_adj:.4f}  |  "
        f"N = {int(model.nobs):,}\n"
        f"(Destination league fixed effects included but not shown)",
        fontsize=11, fontweight="bold", pad=12,
    )
    ax.xaxis.grid(True)

    pvals = model.pvalues[keep]
    for i, (coef, pval) in enumerate(zip(params.values, pvals.values)):
        stars  = (
            "***" if pval < 0.001 else
            "**"  if pval < 0.01  else
            "*"   if pval < 0.05  else
            "ns"
        )
        offset = max(abs(coef) * 0.05, 0.005)
        ax.text(
            coef + (offset if coef >= 0 else -offset), i,
            stars, va="center",
            ha="left" if coef >= 0 else "right",
            fontsize=10, color="white", fontweight="bold",
        )

    plt.tight_layout()
    out = f"{DIR_FIGURES}/fig7_ols_coefficients.png"
    plt.savefig(out); plt.close()
    print(f"[✓] {out}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Propensity Score Matching
# ══════════════════════════════════════════════════════════════════════════════

def run_psm(
    df: pd.DataFrame,
) -> tuple:
    """
    1:1 Nearest-Neighbour PSM with caliper = 0.1 × SD(logit PS).

    Treatment : high_pressure_treat = 1  (origin pressure ≥ median)
    Outcome   : log_transfer_fee
    Covariates: destination league dummies + season_year

    Returns
    -------
    att        : float | None   Average Treatment Effect on the Treated
    se_boot    : float | None   Bootstrap SE (1 000 iterations)
    balance_df : pd.DataFrame | None   Covariate balance table
    psm_df     : pd.DataFrame | None   Dataset with propensity scores
    """
    required = ["log_transfer_fee", "high_pressure_treat",
                "dest_league", "season_year", "origin_pressure"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] PSM skipped — missing columns: {missing}")
        return None, None, None, None

    psm_df = df[required].dropna().copy()
    psm_df = psm_df[psm_df["origin_pressure"].notna()]

    if len(psm_df) < 50:
        print(f"[WARN] PSM skipped — only {len(psm_df)} complete cases.")
        return None, None, None, None

    T = psm_df["high_pressure_treat"].values
    y = psm_df["log_transfer_fee"].values

    # Covariates: destination league dummies + season year
    dest_dummies = pd.get_dummies(
        psm_df["dest_league"], prefix="dest", drop_first=True, dtype=float
    )
    X_df = pd.concat(
        [psm_df[["season_year"]].astype(float), dest_dummies], axis=1
    )
    X    = X_df.values.astype(float)

    # ── Step 1: Logistic regression → propensity scores ───────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logit    = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
    logit.fit(X_scaled, T)
    ps       = logit.predict_proba(X_scaled)[:, 1]
    logit_ps = np.log(
        np.clip(ps, 1e-6, 1 - 1e-6) / (1 - np.clip(ps, 1e-6, 1 - 1e-6))
    )

    psm_df = psm_df.copy()
    psm_df["ps"]       = ps
    psm_df["logit_ps"] = logit_ps

    # ── Step 2: 1:1 NN matching with caliper ─────────────────────────────────
    caliper     = 0.1 * logit_ps.std()
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nn.fit(logit_ps[control_idx].reshape(-1, 1))
    distances, matched_pos = nn.kneighbors(
        logit_ps[treated_idx].reshape(-1, 1)
    )
    distances    = distances.flatten()
    ctrl_matched = control_idx[matched_pos.flatten()]

    within   = distances <= caliper
    t_match  = treated_idx[within]
    c_match  = ctrl_matched[within]
    n_pairs  = within.sum()
    match_rate = n_pairs / len(treated_idx)

    print(f"\n[PSM] Caliper: {caliper:.4f}  (0.1 × SD logit PS)")
    print(f"[PSM] Treated: {len(treated_idx):,}  |  "
          f"Matched pairs: {n_pairs:,}  ({match_rate:.1%})")

    if match_rate > 0.98:
        print(
            "  ⚠ Match rate ≈ 100% — propensity scores have very low "
            "variance. ATT should be interpreted with caution."
        )

    if n_pairs < 20:
        print("[WARN] Too few matched pairs for reliable inference.")
        return None, None, None, psm_df

    y_t = y[t_match]; y_c = y[c_match]
    att = (y_t - y_c).mean()

    # ── Step 3: Paired t-test ─────────────────────────────────────────────────
    t_stat, p_value = scipy_stats.ttest_rel(y_t, y_c)
    ci_lo, ci_hi   = scipy_stats.t.interval(
        0.95, df=n_pairs - 1,
        loc=att, scale=scipy_stats.sem(y_t - y_c),
    )

    # ── Step 4: Bootstrap SE (1 000 iterations) ───────────────────────────────
    rng = np.random.default_rng(RANDOM_SEED)
    boot_atts = [
        (y_t[rng.integers(0, n_pairs, n_pairs)] -
         y_c[rng.integers(0, n_pairs, n_pairs)]).mean()
        for _ in range(1000)
    ]
    se_boot = float(np.std(boot_atts))

    print(f"[PSM] ATT = {att:+.4f} log-fee units  |  "
          f"SE = {se_boot:.4f}  |  t = {t_stat:.2f}  |  p = {p_value:.4f}")
    print(f"[PSM] 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"[PSM] exp(ATT) = {np.exp(att):.3f}  "
          f"(fee premium multiplier; >1 = high-pressure advantage)")

    # ── Step 5: Covariate balance ─────────────────────────────────────────────
    balance_rows = []
    for i, col in enumerate(X_df.columns):
        mu_t_pre = X[treated_idx, i].mean()
        mu_c_pre = X[control_idx, i].mean()
        sd_pre   = np.sqrt(
            (X[treated_idx, i].var() + X[control_idx, i].var()) / 2 + 1e-9
        )
        smd_pre  = abs(mu_t_pre - mu_c_pre) / sd_pre

        mu_t_post = X[t_match, i].mean()
        mu_c_post = X[c_match, i].mean()
        sd_post   = np.sqrt(
            (X[t_match, i].var() + X[c_match, i].var()) / 2 + 1e-9
        )
        smd_post = abs(mu_t_post - mu_c_post) / sd_post

        balance_rows.append({
            "Covariate":  col,
            "SMD_before": round(smd_pre,  4),
            "SMD_after":  round(smd_post, 4),
            "balanced":   smd_post < 0.1,
        })

    balance_df    = pd.DataFrame(balance_rows)
    n_unbalanced  = (~balance_df["balanced"]).sum()
    balance_note  = (
        f"  ⚠ {n_unbalanced} covariate(s) still unbalanced (SMD ≥ 0.1)"
        if n_unbalanced else
        "  ✓ All covariates balanced (SMD < 0.1)"
    )
    print(balance_note)

    # Save tables immediately
    psm_summary = pd.DataFrame({
        "Metric": [
            "ATT (log-fee)", "exp(ATT) — fee multiplier",
            "SE (bootstrap)", "t-statistic",
            "p-value (paired t)", "95% CI lower", "95% CI upper",
            "Match rate", "N matched pairs",
        ],
        "Value": [
            f"{att:.4f}",        f"{np.exp(att):.3f}",
            f"{se_boot:.4f}",   f"{t_stat:.2f}",
            f"{p_value:.4f}",   f"{ci_lo:.4f}", f"{ci_hi:.4f}",
            f"{match_rate:.3f}", f"{n_pairs:,}",
        ],
    })
    csv1 = f"{DIR_TABLES}/table8_psm_results.csv"
    csv2 = f"{DIR_TABLES}/table8b_psm_balance.csv"
    psm_summary.to_csv(csv1, index=False)
    balance_df.to_csv(csv2, index=False)
    print(f"[✓] {csv1}")
    print(f"[✓] {csv2}\n")
    return att, se_boot, balance_df, psm_df


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — PSM results
# ══════════════════════════════════════════════════════════════════════════════

def plot_psm_results(
    balance_df,
    psm_df,
    att,
    se,
) -> None:
    """Covariate balance bars + PS overlap histogram — saved as fig8."""
    if balance_df is None or psm_df is None:
        print("[WARN] PSM results unavailable — skipping fig8.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (a) Balance: non-dummy covariates only
    bal_show = balance_df[
        ~balance_df["Covariate"].str.startswith("dest_")
    ].copy()
    ax = axes[0]
    x  = np.arange(len(bal_show)); w = 0.35
    ax.barh(x - w / 2, bal_show["SMD_before"], height=w,
            color=PAL["coral"], label="Before PSM", edgecolor="none")
    ax.barh(x + w / 2, bal_show["SMD_after"], height=w,
            color=PAL["teal"],  label="After PSM",  edgecolor="none")
    ax.axvline(0.1, color=PAL["gold"], linestyle="--", linewidth=1.5,
               label="Balance threshold (0.1)")
    ax.set_yticks(x); ax.set_yticklabels(bal_show["Covariate"])
    ax.set_xlabel("|Standardized Mean Difference|")
    ax.set_title(
        "Covariate Balance Before and After PSM\n"
        "(Destination league dummies omitted for readability)"
    )
    ax.legend(framealpha=0.3); ax.xaxis.grid(True)

    # (b) PS overlap
    ax = axes[1]
    if "ps" in psm_df.columns and "high_pressure_treat" in psm_df.columns:
        treated = psm_df[psm_df["high_pressure_treat"] == 1]["ps"]
        control = psm_df[psm_df["high_pressure_treat"] == 0]["ps"]
        n_samp  = min(5000, len(control), len(treated))
        rng     = np.random.default_rng(RANDOM_SEED)
        ax.hist(
            rng.choice(control.values, n_samp, replace=False),
            bins=40, alpha=0.6, color=PAL["coral"], density=True,
            label="Control (low pressure)", edgecolor="none",
        )
        ax.hist(
            rng.choice(treated.values, n_samp, replace=False),
            bins=40, alpha=0.6, color=PAL["teal"], density=True,
            label="Treated (high pressure)", edgecolor="none",
        )
        ax.set_xlabel("Propensity Score P(T=1|X)")
        ax.set_ylabel("Density")
        att_str = f"{att:+.4f}" if att is not None else "N/A"
        se_str  = f"{se:.4f}"  if se  is not None else "N/A"
        ax.set_title(
            f"Propensity Score Overlap\n"
            f"ATT = {att_str} log-fee units  (Bootstrap SE = {se_str})"
        )
        ax.legend(framealpha=0.3); ax.yaxis.grid(True)

    fig.suptitle("Figure 8: Propensity Score Matching Results",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = f"{DIR_FIGURES}/fig8_psm.png"
    plt.savefig(out); plt.close()
    print(f"[✓] {out}\n")


# ── Public runner ─────────────────────────────────────────────────────────────

def run_statistical_analysis(df: pd.DataFrame) -> tuple:
    """
    Run the full statistical pipeline in order.

    Returns
    -------
    model      : OLS model (or None)
    att        : PSM ATT (or None)
    se_boot    : PSM bootstrap SE (or None)
    balance_df : PSM balance table (or None)
    """
    print("\n── STATISTICAL ANALYSIS ─────────────────────────────────────")
    plot_correlation_matrix(df)
    model  = run_ols_regression(df)
    plot_ols_coefficients(model)          # receives model directly — not a list
    att, se_boot, balance_df, psm_df = run_psm(df)
    plot_psm_results(balance_df, psm_df, att, se_boot)
    return model, att, se_boot, balance_df
