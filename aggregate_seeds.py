"""
Aggregate v6 seeds — compute mean ± SD across 10 seeds.

Inputs:
  history_ISIC2018_v6/seed_{0,1,2,3,7,11,31,42,99,123}/summary_*.csv

Outputs:
  history_ISIC2018_v6/aggregated.csv
     10-seed mean ± SD per (Cohort, Model, Metric)
  history_ISIC2018_v6/paired_delta.csv
     Per-seed Δ(Causal − Image-Only) Accuracy + paired-test stats
  history_ISIC2018_v6/aggregated_summary.png
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# n=10 multi-seed analysis. All 10 seeds are trained under the SAME
# 30-epoch protocol and written to history_ISIC2018_v6/seed_{s}/ by
# train_ISIC_2018_04.py — including seed 42, which previously lived in
# history_ISIC2018_v2 at a different epoch budget.
SEED_DIRS = {
    s: f"history_ISIC2018_v6/seed_{s}"
    for s in [0, 1, 2, 3, 7, 11, 31, 42, 99, 123]
}

COHORT_FILES_V6 = {
    "HAM val (in-dist)":           "summary_val.csv",
    "ISIC2018 Test (full)":        "summary_isic2018_test.csv",
    "ISIC2018 Test (known-only)":  "summary_isic2018_test_known_only.csv",
    "Fitzpatrick17k":              "summary_fitzpatrick17k.csv",
    "PAD-UFES-20":                 "summary_pad_ufes_20.csv",
}


def load_seed_other(seed, cohort, model_name):
    """Load v6 per-seed summary CSV."""
    seed_dir = SEED_DIRS[seed]
    fname = COHORT_FILES_V6[cohort]
    path = os.path.join(seed_dir, fname)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    row = df[df["Model"] == model_name]
    if len(row) == 0:
        return None
    return {
        "Accuracy":         float(row["Accuracy"].iloc[0]),
        "Mean Sensitivity": float(row["Mean Sensitivity"].iloc[0]),
        "Mean Specificity": float(row["Mean Specificity"].iloc[0]),
    }


def collect_all():
    """Return long-form DataFrame: (Seed, Cohort, Model, Metric, Value)."""
    rows = []
    for seed in SEED_DIRS:
        for cohort in COHORT_FILES_V6:
            for model in ["Image-Only", "Meta-Only", "Causal-Factorized"]:
                res = load_seed_other(seed, cohort, model)
                if res is None:
                    continue
                for metric, val in res.items():
                    rows.append({"Seed": seed, "Cohort": cohort,
                                 "Model": model, "Metric": metric,
                                 "Value": val})
    return pd.DataFrame(rows)


def aggregate(long_df):
    """Compute mean and SD across seeds."""
    agg = (long_df.groupby(["Cohort", "Model", "Metric"])["Value"]
                  .agg(["mean", "std", "count"]).reset_index())
    agg["mean±sd"] = agg.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)
    return agg


def paired_delta(long_df):
    """Per-seed paired Δ (Causal-Factorized − Image-Only) for Accuracy.

    Reports mean Δ, SD, t-stat, and one-sided p-value (Δ > 0).
    """
    rows = []
    sub = long_df[(long_df["Metric"] == "Accuracy") &
                  (long_df["Model"].isin(["Causal-Factorized", "Image-Only"]))]
    for cohort in sub["Cohort"].unique():
        cohort_df = sub[sub["Cohort"] == cohort]
        wide = cohort_df.pivot_table(index="Seed", columns="Model",
                                     values="Value")
        if "Causal-Factorized" not in wide.columns \
                or "Image-Only" not in wide.columns:
            continue
        wide = wide.dropna()
        deltas = wide["Causal-Factorized"] - wide["Image-Only"]
        if len(deltas) < 2:
            continue
        t, p_two = stats.ttest_1samp(deltas, 0.0)
        # one-sided p (Δ > 0)
        p_one = p_two / 2 if t > 0 else 1 - p_two / 2
        rows.append({
            "Cohort": cohort,
            "N seeds": int(len(deltas)),
            "Mean Δ": float(deltas.mean()),
            "SD Δ":   float(deltas.std()),
            "Min Δ":  float(deltas.min()),
            "Max Δ":  float(deltas.max()),
            "t-stat": float(t),
            "p (Δ>0, one-sided)": float(p_one),
        })
    return pd.DataFrame(rows)


def plot_summary(agg_df, paired_df, save_path):
    """Bar chart with error bars: mean accuracy per (Cohort, Model)."""
    acc = agg_df[agg_df["Metric"] == "Accuracy"].copy()
    cohorts = list(acc["Cohort"].unique())
    models  = ["Image-Only", "Meta-Only", "Causal-Factorized"]
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(cohorts))
    w = 0.27
    palette = {"Image-Only": "#4C72B0",
               "Meta-Only": "#55A868",
               "Causal-Factorized": "#C44E52"}
    for i, m in enumerate(models):
        means, stds = [], []
        for c in cohorts:
            sub = acc[(acc["Cohort"] == c) & (acc["Model"] == m)]
            if len(sub):
                means.append(float(sub["mean"].iloc[0]))
                stds.append(float(sub["std"].iloc[0]))
            else:
                means.append(0); stds.append(0)
        means = np.array(means); stds = np.array(stds)
        ax.bar(x + (i - 1) * w, means, w, yerr=stds, label=m,
               color=palette[m], capsize=4)
        for j, (m_, s_) in enumerate(zip(means, stds)):
            ax.text(j + (i - 1) * w, m_ + s_ + 0.005,
                    f"{m_:.3f}\n±{s_:.3f}", ha="center", fontsize=7)

    # Annotate paired p-value on x-axis
    p_lookup = {r["Cohort"]: r["p (Δ>0, one-sided)"]
                for _, r in paired_df.iterrows()}
    tick_labels = []
    for c in cohorts:
        p = p_lookup.get(c, None)
        if p is None:
            tick_labels.append(c)
        else:
            star = "***" if p < 0.001 else "**" if p < 0.01 else \
                   "*" if p < 0.05 else "n.s."
            tick_labels.append(f"{c}\n(Δ p={p:.3f} {star})")
    ax.set_xticks(x); ax.set_xticklabels(tick_labels, rotation=10, ha="right")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.05)
    n_seeds = len(SEED_DIRS)
    ax.set_title(f"Cross-cohort accuracy — {n_seeds}-seed mean ± SD")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    out_dir = "history_ISIC2018_v6"
    os.makedirs(out_dir, exist_ok=True)

    long_df = collect_all()
    print(f"Collected {len(long_df)} rows from "
          f"{long_df['Seed'].nunique()} seeds × "
          f"{long_df['Cohort'].nunique()} cohorts × "
          f"{long_df['Model'].nunique()} models × "
          f"{long_df['Metric'].nunique()} metrics.")
    print(f"Seeds: {sorted(long_df['Seed'].unique())}")
    long_df.to_csv(os.path.join(out_dir, "all_seeds_long.csv"), index=False)

    agg_df = aggregate(long_df)
    agg_df.to_csv(os.path.join(out_dir, "aggregated.csv"), index=False)
    print("\nAggregated (Accuracy only):")
    print(agg_df[agg_df["Metric"] == "Accuracy"]
          [["Cohort", "Model", "mean", "std", "count", "mean±sd"]]
          .to_string(index=False))

    paired_df = paired_delta(long_df)
    paired_df.to_csv(os.path.join(out_dir, "paired_delta.csv"), index=False)
    print("\nPaired Δ (Causal − Image-Only) per cohort:")
    print(paired_df.to_string(index=False))

    plot_summary(agg_df, paired_df,
                 os.path.join(out_dir, "aggregated_summary.png"))
    print(f"\nOutputs:")
    print(f"  {out_dir}/aggregated.csv")
    print(f"  {out_dir}/paired_delta.csv")
    print(f"  {out_dir}/aggregated_summary.png")
