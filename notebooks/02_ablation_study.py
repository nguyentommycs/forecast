# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Ablation Study
#
# Answers: **How much does each feature hurt accuracy when removed?**
#
# Protocol:
# 1. Train a baseline model on all 11 features; record RMSE / MAE / MAPE
# 2. For each feature: drop it, retrain, record the metric delta vs. baseline
# 3. Also test dropping entire feature groups (all lags, all time features)
#
# > **Runtime:** ~11-15 min total (13 training runs with early stopping).

# %%
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd

from utils import (
    FEATURE_COLS, TARGET_COL,
    load_features, split, train_and_evaluate,
)

# %% [markdown]
# ## Load data and establish baseline

# %%
df = load_features()
train_df, test_df = split(df)

print(f"Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")
print(f"Test cutoff: {test_df['time_bucket'].min()}")

# %%
print("Training baseline (all 11 features)...")
baseline = train_and_evaluate(train_df, test_df, FEATURE_COLS)
print(f"  Baseline  RMSE={baseline['rmse']:.3f}  MAE={baseline['mae']:.3f}  MAPE={baseline['mape']:.1f}%")

# %% [markdown]
# ## Leave-one-out ablation

# %%
results = []

for feature in FEATURE_COLS:
    reduced_cols = [f for f in FEATURE_COLS if f != feature]
    print(f"  Dropping '{feature}'...", end=" ")
    r = train_and_evaluate(train_df, test_df, reduced_cols)
    delta_rmse = r["rmse"] - baseline["rmse"]
    delta_mae  = r["mae"]  - baseline["mae"]
    results.append({
        "dropped":    feature,
        "rmse":       r["rmse"],
        "mae":        r["mae"],
        "mape":       r["mape"],
        "d_rmse":     delta_rmse,
        "d_mae":      delta_mae,
    })
    print(f"RMSE={r['rmse']:.3f}  d={delta_rmse:+.3f}")

# %%
ablation_df = (
    pd.DataFrame(results)
    .sort_values("d_rmse", ascending=False)
    .reset_index(drop=True)
)

print("\nLeave-one-out results (sorted by RMSE increase when dropped):")
print(ablation_df.to_string(index=False, float_format="{:.3f}".format))

# %% [markdown]
# ## Group ablations

# %%
LAG_FEATURES  = [f for f in FEATURE_COLS if f.startswith("lag_") or f.startswith("rolling_")]
TIME_FEATURES = ["hour_of_day", "day_of_week", "is_weekend"]

group_experiments = {
    "drop all lags + rolling": [f for f in FEATURE_COLS if f not in LAG_FEATURES],
    "drop all time features":  [f for f in FEATURE_COLS if f not in TIME_FEATURES],
}

group_results = []
for label, cols in group_experiments.items():
    print(f"  '{label}' ({len(cols)} features)...", end=" ")
    r = train_and_evaluate(train_df, test_df, cols)
    delta_rmse = r["rmse"] - baseline["rmse"]
    group_results.append({
        "experiment": label,
        "n_features": len(cols),
        "rmse":       r["rmse"],
        "mae":        r["mae"],
        "mape":       r["mape"],
        "d_rmse":     delta_rmse,
    })
    print(f"RMSE={r['rmse']:.3f}  d={delta_rmse:+.3f}")

# %%
group_df = pd.DataFrame(group_results)
print("\nGroup ablation results:")
print(group_df.to_string(index=False, float_format="{:.3f}".format))

# %% [markdown]
# ## Visualisation

# %%
fig, ax = plt.subplots(figsize=(10, 5))

colors = ["#d62728" if v > 0 else "#2ca02c" for v in ablation_df["d_rmse"]]
ax.barh(ablation_df["dropped"], ablation_df["d_rmse"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("RMSE change vs. baseline (positive = accuracy loss)")
ax.set_title("Leave-one-out ablation: dRMSE when feature is removed")

plt.tight_layout()
plt.savefig("02_ablation_rmse_delta.png", dpi=120)
plt.show()

# %% [markdown]
# ## Summary
#
# - **Large positive dRMSE** -> feature is load-bearing; removing it hurts significantly
# - **Near-zero dRMSE** -> feature adds little value; candidate for removal
# - **Negative dRMSE** -> feature may be adding noise; consider dropping

# %%
print(f"\nBaseline RMSE:  {baseline['rmse']:.3f}")
print(f"Most critical:  {ablation_df.iloc[0]['dropped']}  (d={ablation_df.iloc[0]['d_rmse']:+.3f})")
print(f"Least critical: {ablation_df.iloc[-1]['dropped']}  (d={ablation_df.iloc[-1]['d_rmse']:+.3f})")

# %%
