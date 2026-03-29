# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Candidate Feature Evaluation
#
# Informed by notebooks 01 and 02, run on leak-free features (rolling_mean_6 uses shift(1)).
#
# Key findings from 01 and 02:
#   - lag_1 dominates: ~90% of gain, SHAP 22.2, +2.0 RMSE when dropped
#   - zone_id and hour_of_day are the next most critical (+1.1 RMSE each)
#   - rolling_mean_6 lost most of its apparent power after the leakage fix (SHAP 0.37, +0.2 RMSE)
#   - lag_24 is near-zero neutral (+0.02 RMSE when dropped) - no longer harmful
#   - lag_12 now shows real contribution (~+0.22 RMSE), comparable to lag_3 and lag_6
#   - is_weekend now shows some value (~+0.15 RMSE), less clearly redundant than before
#
# Part 1 - Pruning: test removal of the weakest features
#   - lag_24: near-zero contribution, clearest pruning candidate
#   - lag_12 and is_weekend: moderate contributors, test removal to see if they help
#   - Combinations: test removing multiple at once
#
# Part 2 - Additions: test new features against the best pruned set
#   - lag_168 (same hour last week): weekly seasonality is absent from current features
#   - rolling_mean_3: faster window; may capture recent changes rolling_mean_6 smooths over
#   - rolling_std_6: volatility signal (different info from the mean)
#   - rolling_mean_12: medium window between 6h and the absent lag_24
#   - hour_sin + hour_cos: cyclical hour encoding
#
# Part 3 - Best combined model: pruned set + top new features vs original baseline
#
# Runtime: ~15-20 min (15 training runs).

# %%
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from utils import (
    AGGREGATED_FILE, FEATURE_COLS, TARGET_COL,
    load_features, load_aggregated, split, train_and_evaluate,
)

# %% [markdown]
# ## Section 1: Pruning experiments
#
# Uses features.parquet directly - no new feature computation needed.
# Baseline should match notebook 02 (with leak-free rolling_mean_6).
# lag_24 is the clearest pruning candidate (near-zero ablation impact).
# lag_12 and is_weekend have real but moderate contributions - tested for completeness.

# %%
df = load_features()
train_df, test_df = split(df)

print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")
print(f"Test cutoff: {test_df['time_bucket'].min()}")

print("\nTraining baseline (all 11 features)...")
baseline = train_and_evaluate(train_df, test_df, FEATURE_COLS)
print(f"Baseline -> RMSE={baseline['rmse']:.3f}  MAE={baseline['mae']:.3f}  MAPE={baseline['mape']:.1f}%")

# %%
PRUNE_EXPERIMENTS = [
    ("drop lag_24",                    [f for f in FEATURE_COLS if f != "lag_24"]),
    ("drop lag_12",                    [f for f in FEATURE_COLS if f != "lag_12"]),
    ("drop is_weekend",                [f for f in FEATURE_COLS if f != "is_weekend"]),
    ("drop lag_24 + lag_12",           [f for f in FEATURE_COLS if f not in ("lag_24", "lag_12")]),
    ("drop lag_24 + lag_12 + is_wknd", [f for f in FEATURE_COLS if f not in ("lag_24", "lag_12", "is_weekend")]),
]

prune_results = []

for label, cols in PRUNE_EXPERIMENTS:
    print(f"  {label} ({len(cols)} features)...", end=" ", flush=True)
    r = train_and_evaluate(train_df, test_df, cols)
    d_rmse = r["rmse"] - baseline["rmse"]
    prune_results.append({
        "experiment": label,
        "n_features": len(cols),
        "rmse":       r["rmse"],
        "mae":        r["mae"],
        "mape":       r["mape"],
        "d_rmse":     d_rmse,
    })
    print(f"RMSE={r['rmse']:.3f}  d={d_rmse:+.3f}")

# %%
prune_df = pd.DataFrame(prune_results).sort_values("d_rmse").reset_index(drop=True)
print("\nPruning results (best first):")
print(prune_df.to_string(index=False, float_format="{:.3f}".format))

# pick the best pruned set to use as the baseline for section 2
best_prune = prune_df.iloc[0]
best_pruned_cols = dict(PRUNE_EXPERIMENTS)[best_prune["experiment"]]
print(f"\nBest pruned set: '{best_prune['experiment']}'")
print(f"  Features ({len(best_pruned_cols)}): {best_pruned_cols}")

# %% [markdown]
# ### Pruning chart

# %%
fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#2ca02c" if v < 0 else "#d62728" for v in prune_df["d_rmse"]]
ax.barh(prune_df["experiment"], prune_df["d_rmse"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("RMSE change vs. 11-feature baseline (negative = improvement)")
ax.set_title("Section 1: Feature pruning experiments")
plt.tight_layout()
plt.savefig("03_pruning.png", dpi=120)
plt.show()

# %% [markdown]
# ## Section 2: New feature candidates
#
# Computed from aggregated.parquet alongside all baseline features in one Polars pass.
# lag_168 requires 168-row history per zone, so more rows are dropped than in Section 1.
# The baseline here will differ slightly in row count - both baselines are stated explicitly.

# %%
df_agg = load_aggregated()
print(f"Aggregated rows: {len(df_agg):,}")

df_enriched = (
    df_agg
    .sort(["zone_id", "time_bucket"])
    .with_columns(
        # baseline time features
        pl.col("time_bucket").dt.hour().alias("hour_of_day"),
        pl.col("time_bucket").dt.weekday().alias("day_of_week"),
        (pl.col("time_bucket").dt.weekday() >= 5).alias("is_weekend"),
        # baseline lags
        pl.col("trip_count").shift(1).over("zone_id").alias("lag_1"),
        pl.col("trip_count").shift(2).over("zone_id").alias("lag_2"),
        pl.col("trip_count").shift(3).over("zone_id").alias("lag_3"),
        pl.col("trip_count").shift(6).over("zone_id").alias("lag_6"),
        pl.col("trip_count").shift(12).over("zone_id").alias("lag_12"),
        pl.col("trip_count").shift(24).over("zone_id").alias("lag_24"),
        # baseline rolling - shift(1) so window covers [t-6, t-1], not [t-5, t]
        pl.col("trip_count").shift(1).rolling_mean(window_size=6).over("zone_id").alias("rolling_mean_6"),
        # candidate: weekly lag
        pl.col("trip_count").shift(168).over("zone_id").alias("lag_168"),
        # candidate: rolling windows - same shift(1) fix applied to all
        pl.col("trip_count").shift(1).rolling_mean(window_size=3).over("zone_id").alias("rolling_mean_3"),
        pl.col("trip_count").shift(1).rolling_mean(window_size=12).over("zone_id").alias("rolling_mean_12"),
        pl.col("trip_count").shift(1).rolling_std(window_size=6).over("zone_id").alias("rolling_std_6"),
    )
    .with_columns(
        # cyclical hour encoding (computed after hour_of_day exists)
        (pl.col("hour_of_day").cast(pl.Float64) * (2 * np.pi / 24)).sin().alias("hour_sin"),
        (pl.col("hour_of_day").cast(pl.Float64) * (2 * np.pi / 24)).cos().alias("hour_cos"),
    )
    .drop_nulls()
)

print(f"Enriched rows (after lag_168 null-drop): {len(df_enriched):,}")

# %%
train_e, test_e = split(df_enriched)
print(f"Train: {len(train_e):,}  |  Test: {len(test_e):,}")

# Establish section 2 baseline using the best pruned set from section 1
print(f"\nSection 2 baseline: '{best_prune['experiment']}' on enriched dataset...")
base2 = train_and_evaluate(train_e, test_e, best_pruned_cols)
print(f"  Section 2 baseline -> RMSE={base2['rmse']:.3f}  MAE={base2['mae']:.3f}  MAPE={base2['mape']:.1f}%")
print(f"  (Section 1 baseline was RMSE={baseline['rmse']:.3f} on {len(train_df):,} train rows)")

# %%
CANDIDATES = [
    ("lag_168",              ["lag_168"]),
    ("rolling_mean_3",       ["rolling_mean_3"]),
    ("rolling_std_6",        ["rolling_std_6"]),
    ("rolling_mean_12",      ["rolling_mean_12"]),
    ("hour_sin + hour_cos",  ["hour_sin", "hour_cos"]),
]

cand_results = []

for label, extra_cols in CANDIDATES:
    feature_set = best_pruned_cols + extra_cols
    print(f"  Adding {label}...", end=" ", flush=True)
    r = train_and_evaluate(train_e, test_e, feature_set)
    d_rmse = r["rmse"] - base2["rmse"]
    cand_results.append({
        "candidate": label,
        "n_features": len(feature_set),
        "rmse":      r["rmse"],
        "mae":       r["mae"],
        "mape":      r["mape"],
        "d_rmse":    d_rmse,
        "d_mae":     r["mae"] - base2["mae"],
    })
    print(f"RMSE={r['rmse']:.3f}  d={d_rmse:+.3f}")

# %%
cand_df = pd.DataFrame(cand_results).sort_values("d_rmse").reset_index(drop=True)
print("\nCandidate results vs section 2 baseline (best first):")
print(cand_df.to_string(index=False, float_format="{:.3f}".format))

# %% [markdown]
# ### Candidates chart

# %%
fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#2ca02c" if v < 0 else "#d62728" for v in cand_df["d_rmse"]]
ax.barh(cand_df["candidate"], cand_df["d_rmse"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("RMSE change vs. section 2 baseline (negative = improvement)")
ax.set_title("Section 2: New feature candidates")
plt.tight_layout()
plt.savefig("03_candidates.png", dpi=120)
plt.show()

# %% [markdown]
# ## Section 3: Best combined model
#
# Pruned set from Section 1 + top new features from Section 2.
# Compared back to the original 11-feature baseline on the enriched dataset.

# %%
# Pick candidates that improved over the section 2 baseline
improving_candidates = cand_df[cand_df["d_rmse"] < 0]["candidate"].tolist()

if not improving_candidates:
    print("No candidates improved - best model is the pruned set alone.")
    best_combined_cols = best_pruned_cols
else:
    extra = []
    for label, cols in CANDIDATES:
        if label in improving_candidates:
            extra.extend(cols)
    best_combined_cols = best_pruned_cols + extra
    print(f"Combining: '{best_prune['experiment']}' + {improving_candidates}")

print(f"Final feature set ({len(best_combined_cols)} features): {best_combined_cols}")

# %%
# Re-establish original 11-feature baseline on enriched dataset for fair comparison
print("Original 11-feature baseline on enriched dataset...")
base_orig = train_and_evaluate(train_e, test_e, FEATURE_COLS)
print(f"  Original baseline -> RMSE={base_orig['rmse']:.3f}")

print("\nBest combined model...")
combined = train_and_evaluate(train_e, test_e, best_combined_cols)
d_vs_orig = combined["rmse"] - base_orig["rmse"]
print(f"  Combined -> RMSE={combined['rmse']:.3f}  MAE={combined['mae']:.3f}  MAPE={combined['mape']:.1f}%")
print(f"  vs original 11-feature baseline: d={d_vs_orig:+.3f}")

# %% [markdown]
# ## Summary and report

# %%
lines = []
lines.append("=" * 60)
lines.append("FEATURE EXPERIMENT REPORT")
lines.append("=" * 60)
lines.append("")

lines.append("SECTION 1: PRUNING")
lines.append("-" * 60)
lines.append(f"{'Experiment':<35} {'N':>2}  {'RMSE':>7}  {'d_rmse':>7}")
lines.append(f"{'baseline (11 features)':<35} {'11':>2}  {baseline['rmse']:>7.3f}  {'0.000':>7}")
for _, row in prune_df.iterrows():
    lines.append(f"{row['experiment']:<35} {int(row['n_features']):>2}  {row['rmse']:>7.3f}  {row['d_rmse']:>+7.3f}")
lines.append("")

lines.append("SECTION 2: NEW CANDIDATES")
lines.append(f"Section 2 baseline: '{best_prune['experiment']}'  RMSE={base2['rmse']:.3f}")
lines.append("-" * 60)
lines.append(f"{'Candidate':<25} {'N':>2}  {'RMSE':>7}  {'d_rmse':>7}  {'d_mae':>7}")
for _, row in cand_df.iterrows():
    lines.append(
        f"{row['candidate']:<25} {int(row['n_features']):>2}  {row['rmse']:>7.3f}"
        f"  {row['d_rmse']:>+7.3f}  {row['d_mae']:>+7.3f}"
    )
lines.append("")

lines.append("SECTION 3: BEST COMBINED MODEL")
lines.append("-" * 60)
lines.append(f"Original 11-feature baseline:  RMSE={base_orig['rmse']:.3f}  MAE={base_orig['mae']:.3f}  MAPE={base_orig['mape']:.1f}%")
lines.append(f"Best combined model:           RMSE={combined['rmse']:.3f}  MAE={combined['mae']:.3f}  MAPE={combined['mape']:.1f}%")
lines.append(f"Improvement vs original:       d_rmse={d_vs_orig:+.3f}")
lines.append("")
lines.append(f"Best pruned set:    {best_prune['experiment']}")
lines.append(f"New features added: {', '.join(improving_candidates) if improving_candidates else 'none'}")
lines.append(f"Final feature set ({len(best_combined_cols)}): {best_combined_cols}")
lines.append("=" * 60)

report = "\n".join(lines)
print(report)

report_path = "03_report.txt"
with open(report_path, "w") as f:
    f.write(report + "\n")
print(f"\nReport saved -> {report_path}")

# %%
