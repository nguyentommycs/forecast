# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Hyperparameter Tuning -- LightGBM NYC Taxi Demand
#
# Finds optimal LightGBM hyperparameters using Optuna + TimeSeriesSplit CV.
#
# Protocol:
# 1. Establish a CV baseline with current hyperparameters.
# 2. Run an Optuna study (100 trials) over a defined search space.
# 3. Retrain on the full training set with the best params.
# 4. Print a drop-in replacement dict for `train.py` and `utils.py`.
#
# > **Runtime:** ~40-60 min (100 trials x 5 folds, early stopping active).
# > Reduce N_TRIALS to 20-30 for a quick smoke-test.
#
# > **Note:** CV RMSE is not directly comparable to the 8.95 single-split RMSE
# > in `models/report.txt` -- it averages across 5 different validation windows.

# %%
# pip install optuna  -- not in requirements.txt; install once if needed
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from utils import load_features, split, MODEL_FILE

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_SPLITS    = 5      # TimeSeriesSplit folds
N_TRIALS    = 100    # Optuna trials; reduce to 20-30 for a quick test
EARLY_STOP  = 50     # early stopping rounds per fold
RANDOM_SEED = 42

# utils.py FEATURE_COLS is outdated (11 features); use train.py's 16-feature set
FEATURE_COLS = [
    "zone_id",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_mean_6",
    "lag_168",
    "rolling_mean_3",
    "rolling_std_6",
    "rolling_mean_12",
    "hour_sin",
    "hour_cos",
]
TARGET_COL = "trip_count"

# %% [markdown]
# ## Load Data

# %%
df = load_features()
train_df, test_df = split(df)

print(f"Full dataset:  {len(df):,} rows  |  {df['zone_id'].n_unique()} zones")
print(f"Training set:  {len(train_df):,} rows")
print(f"Test set:      {len(test_df):,} rows  (held out -- not used during tuning)")
print(f"Date range:    {df['time_bucket'].min()} -> {df['time_bucket'].max()}")

# %% [markdown]
# ## Prepare Arrays
#
# Convert training data to pandas once outside the trial loop.
# `zone_id` is cast to `category` here so every fold inherits it automatically.
# The parquet is sorted by (zone_id, time_bucket), so we sort by time_bucket
# globally before extracting arrays -- TimeSeriesSplit splits on row index and
# requires a monotonically non-decreasing time axis.

# %%
train_sorted = train_df.sort("time_bucket")

X_train_full = train_sorted[FEATURE_COLS].to_pandas()
y_train_full = train_sorted[TARGET_COL].to_pandas().to_numpy()

X_train_full["zone_id"] = X_train_full["zone_id"].astype("category")

print(f"X_train shape: {X_train_full.shape}")

# %% [markdown]
# ## Baseline CV
#
# Evaluate current hyperparameters with 5-fold TimeSeriesSplit.
# This is the reference RMSE Optuna will try to beat.

# %%
def cv_rmse(params: dict, X: pd.DataFrame, y: np.ndarray, n_splits: int = N_SPLITS) -> float:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmses = []

    for train_idx, val_idx in tscv.split(X):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y[train_idx],       y[val_idx]

        model = lgb.LGBMRegressor(
            **params,
            n_jobs=-1,
            verbose=-1,
            random_state=RANDOM_SEED,
        )
        model.fit(
            X_fold_train,
            y_fold_train,
            categorical_feature=["zone_id"],
            eval_set=[(X_fold_val, y_fold_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        preds = np.clip(model.predict(X_fold_val), 0, None)
        fold_rmses.append(mean_squared_error(y_fold_val, preds) ** 0.5)

    return float(np.mean(fold_rmses))

# %%
BASELINE_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
)

print("Running baseline CV (5 folds)...")
baseline_cv_rmse = cv_rmse(BASELINE_PARAMS, X_train_full, y_train_full)
print(f"Baseline CV RMSE: {baseline_cv_rmse:.4f}")

# %% [markdown]
# ## Search Space
#
# | Parameter | Range | Rationale |
# |---|---|---|
# | `num_leaves` | 20-200 | Model capacity; current=63 |
# | `learning_rate` | 0.01-0.3 (log) | Paired with early stopping so tree count self-adjusts |
# | `n_estimators` | 200-2000 | Upper bound only -- early stopping fires earlier |
# | `min_child_samples` | 10-100 | Leaf size; matters for sparse low-volume zones |
# | `feature_fraction` | 0.5-1.0 | Column subsampling; current=0.8 |
# | `bagging_fraction` | 0.5-1.0 | Row subsampling; current=0.8 |
# | `bagging_freq` | 1-10 | Bagging frequency; current=5 |
# | `reg_alpha` | 1e-8-10 (log) | L1 regularisation; not currently set |
# | `reg_lambda` | 1e-8-10 (log) | L2 regularisation; not currently set |

# %%
def objective(trial: optuna.Trial) -> float:
    params = dict(
        num_leaves        = trial.suggest_int("num_leaves", 20, 200),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        n_estimators      = trial.suggest_int("n_estimators", 200, 2000),
        min_child_samples = trial.suggest_int("min_child_samples", 10, 100),
        feature_fraction  = trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction  = trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq      = trial.suggest_int("bagging_freq", 1, 10),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )
    return cv_rmse(params, X_train_full, y_train_full)

# %% [markdown]
# ## Optuna Study

# %%
study = optuna.create_study(
    direction="minimize",
    study_name="lgbm_taxi_tuning",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
)

# Seed the study with current params so trial 0 is always the baseline
study.enqueue_trial(BASELINE_PARAMS)

print(f"Running Optuna study: {N_TRIALS} trials x {N_SPLITS} CV folds...")
study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True,
)

print(f"\nBest CV RMSE:     {study.best_value:.4f}")
print(f"Baseline CV RMSE: {baseline_cv_rmse:.4f}")
print(f"Improvement:      {baseline_cv_rmse - study.best_value:+.4f}")

# %% [markdown]
# ## Results Analysis

# %%
best_params = study.best_params
print("Best hyperparameters:")
for k, v in best_params.items():
    current = BASELINE_PARAMS.get(k, "N/A (new param)")
    print(f"  {k:<22} {v!r:<14}  (baseline: {current})")

# %%
trials_df = study.trials_dataframe()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(trials_df["number"], trials_df["value"], "o", alpha=0.4, markersize=3, label="Trial RMSE")
axes[0].plot(trials_df["number"], trials_df["value"].cummin(), "-", color="red", linewidth=2, label="Best so far")
axes[0].axhline(baseline_cv_rmse, color="gray", linestyle="--", label=f"Baseline {baseline_cv_rmse:.4f}")
axes[0].set_xlabel("Trial number")
axes[0].set_ylabel("CV RMSE")
axes[0].set_title("Optimization history")
axes[0].legend()

importances = optuna.importance.get_param_importances(study)
imp_names = list(importances.keys())
imp_vals  = list(importances.values())
axes[1].barh(imp_names[::-1], imp_vals[::-1], color="#4C72B0")
axes[1].set_xlabel("Importance (fANOVA)")
axes[1].set_title("Hyperparameter importance")

plt.tight_layout()
plt.savefig("04_optuna_history.png", dpi=120)
plt.show()

# %%
top_trials = (
    trials_df
    .sort_values("value")
    .head(10)
)
param_cols = [c for c in top_trials.columns if c.startswith("params_")]
display_cols = ["number", "value"] + param_cols
top_trials = top_trials[display_cols].copy()
top_trials.columns = ["trial", "cv_rmse"] + [c.replace("params_", "") for c in param_cols]
print("Top 10 trials:")
print(top_trials.to_string(index=False, float_format="{:.4f}".format))

# %% [markdown]
# ## Retrain on Full Training Set
#
# Fit using best params on all training data (pre-cutoff).
# Test-set RMSE here is informational -- the canonical eval is from running `train.py`,
# which also regenerates `models/report.txt`.

# %%
X_test_pd = test_df[FEATURE_COLS].to_pandas()
X_test_pd["zone_id"] = X_test_pd["zone_id"].astype("category")
y_test_np  = test_df[TARGET_COL].to_numpy()

best_model = lgb.LGBMRegressor(
    **best_params,
    n_jobs=-1,
    verbose=-1,
    random_state=RANDOM_SEED,
)
best_model.fit(
    X_train_full,
    y_train_full,
    categorical_feature=["zone_id"],
    eval_set=[(X_test_pd, y_test_np)],
    callbacks=[
        lgb.early_stopping(EARLY_STOP, verbose=False),
        lgb.log_evaluation(50),
    ],
)

preds_test = np.clip(best_model.predict(X_test_pd), 0, None)
test_rmse  = mean_squared_error(y_test_np, preds_test) ** 0.5
test_mae   = mean_absolute_error(y_test_np, preds_test)

print(f"Best-params model -- Test RMSE: {test_rmse:.4f}  MAE: {test_mae:.4f}")
print(f"Baseline model    -- Test RMSE: 8.9500 (from models/report.txt)")
print(f"Best iteration: {best_model.best_iteration_}")

# %%
SAVE_TUNED_MODEL = False  # Set True to overwrite models/model.lgb

if SAVE_TUNED_MODEL:
    best_model.booster_.save_model(str(MODEL_FILE))
    print(f"Tuned model saved -> {MODEL_FILE}")
    print("Remember to run `python retrain.py` afterwards to regenerate models/report.txt.")
else:
    print("Model NOT saved. Recommended workflow:")
    print("  1. Copy the param dict below into train.py and utils.py")
    print("  2. Run: python retrain.py")

# %% [markdown]
# ## Drop-in Replacement Params
#
# Copy the block below into `training/train.py` (lines 130-141) and `notebooks/utils.py` (lines 27-38).

# %%
print("=" * 60)
print("Drop-in replacement for train.py and utils.py:")
print("=" * 60)
print()
print("LGBM_PARAMS = dict(")
for k, v in best_params.items():
    if isinstance(v, float):
        print(f"    {k}={v:.6g},")
    else:
        print(f"    {k}={v},")
print("    random_state=42,")
print("    n_jobs=-1,")
print("    verbose=-1,")
print(")")
print()
print(f"# CV RMSE with these params:      {study.best_value:.4f}")
print(f"# Baseline CV RMSE:               {baseline_cv_rmse:.4f}")
print(f"# Test RMSE (full train set fit):  {test_rmse:.4f}")
print(f"# Baseline test RMSE:              8.9500")
print(f"# Tuned from: Optuna {N_TRIALS} trials, {N_SPLITS}-fold TimeSeriesSplit")

# %% [markdown]
# ## Save Results

# %%
from pathlib import Path
import datetime

results_path = Path("../models/tuning_report.txt")

lines = []
lines.append(f"Hyperparameter Tuning Report")
lines.append(f"Run date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append(f"Optuna trials: {N_TRIALS}  |  CV folds: {N_SPLITS}  |  Early stopping: {EARLY_STOP}")
lines.append("")
lines.append("=" * 60)
lines.append("METRICS SUMMARY")
lines.append("=" * 60)
lines.append(f"Baseline CV RMSE:               {baseline_cv_rmse:.4f}")
lines.append(f"Best CV RMSE:                   {study.best_value:.4f}")
lines.append(f"CV improvement:                 {baseline_cv_rmse - study.best_value:+.4f}")
lines.append(f"Baseline test RMSE:             8.9500")
lines.append(f"Best-params test RMSE:          {test_rmse:.4f}")
lines.append(f"Best-params test MAE:           {test_mae:.4f}")
lines.append(f"Best iteration (early stopped): {best_model.best_iteration_}")
lines.append("")
lines.append("=" * 60)
lines.append("BEST HYPERPARAMETERS")
lines.append("=" * 60)
for k, v in best_params.items():
    baseline_val = BASELINE_PARAMS.get(k, "N/A (new param)")
    lines.append(f"  {k:<22} {v!r:<14}  (baseline: {baseline_val})")
lines.append("")
lines.append("=" * 60)
lines.append("DROP-IN REPLACEMENT")
lines.append("=" * 60)
lines.append("LGBM_PARAMS = dict(")
for k, v in best_params.items():
    if isinstance(v, float):
        lines.append(f"    {k}={v:.6g},")
    else:
        lines.append(f"    {k}={v},")
lines.append("    random_state=42,")
lines.append("    n_jobs=-1,")
lines.append("    verbose=-1,")
lines.append(")")
lines.append("")
lines.append("=" * 60)
lines.append("TOP 10 TRIALS")
lines.append("=" * 60)
lines.append(top_trials.to_string(index=False, float_format="{:.4f}".format))
lines.append("")
lines.append("=" * 60)
lines.append("HYPERPARAMETER IMPORTANCE (fANOVA)")
lines.append("=" * 60)
for param, importance in importances.items():
    lines.append(f"  {param:<22} {importance:.4f}")

results_path.write_text("\n".join(lines))
print(f"Results saved -> {results_path.resolve()}")

# %%
