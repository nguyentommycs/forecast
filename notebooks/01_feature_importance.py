# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Feature Importance Analysis
#
# Answers: **Which features does the trained model rely on?**
#
# Uses two complementary approaches:
# 1. LightGBM built-in importance (gain and split count)
# 2. SHAP values -model-agnostic, accounts for feature interactions

# %%
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from utils import (
    FEATURE_COLS, TARGET_COL,
    load_booster, load_features, split,
)

# %% [markdown]
# ## Load model and test data

# %%
booster = load_booster()

df = load_features()
_, test_df = split(df)

X_test = test_df[FEATURE_COLS].to_pandas()
X_test["zone_id"] = X_test["zone_id"].astype("category")

print(f"Test rows: {len(X_test):,}")
print(f"Features:  {FEATURE_COLS}")

# %% [markdown]
# ## LightGBM Built-in Importance
#
# - **Gain**: total reduction in loss attributed to splits on this feature -higher = more informative
# - **Split**: number of times the feature was used to split a node -can be inflated for high-cardinality features

# %%
gain   = booster.feature_importance(importance_type="gain")
splits = booster.feature_importance(importance_type="split")

importance_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "gain":    gain,
    "split":   splits,
}).sort_values("gain", ascending=False).reset_index(drop=True)

print(importance_df.to_string(index=False))

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, title in [
    (axes[0], "gain",  "Importance by Gain"),
    (axes[1], "split", "Importance by Split Count"),
]:
    df_plot = importance_df.sort_values(col)
    ax.barh(df_plot["feature"], df_plot[col], color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel(col.capitalize())

plt.tight_layout()
plt.savefig("01_lgbm_importance.png", dpi=120)
plt.show()

# %% [markdown]
# ## SHAP Values
#
# SHAP (SHapley Additive exPlanations) assigns each feature a contribution to each individual
# prediction. We sample 5 000 rows from the test set for speed.

# %%
SAMPLE_SIZE = 5_000
rng = np.random.default_rng(42)
idx = rng.choice(len(X_test), size=min(SAMPLE_SIZE, len(X_test)), replace=False)
X_sample = X_test.iloc[idx].reset_index(drop=True)

explainer   = shap.TreeExplainer(booster)
shap_values = explainer(X_sample)

# %% [markdown]
# ### Mean |SHAP| bar chart -overall feature importance

# %%
shap.plots.bar(shap_values, max_display=len(FEATURE_COLS), show=False)
plt.tight_layout()
plt.savefig("01_shap_bar.png", dpi=120)
plt.show()

# %% [markdown]
# ### Beeswarm plot -direction and magnitude of each feature's effect

# %%
shap.plots.beeswarm(shap_values, max_display=len(FEATURE_COLS), show=False)
plt.tight_layout()
plt.savefig("01_shap_beeswarm.png", dpi=120)
plt.show()

# %% [markdown]
# ## Summary
#
# Features with near-zero importance on **both** gain and mean |SHAP|:

# %%
mean_abs_shap = pd.Series(
    np.abs(shap_values.values).mean(axis=0),
    index=FEATURE_COLS,
).sort_values()

low_threshold = mean_abs_shap.quantile(0.25)
low_importance = mean_abs_shap[mean_abs_shap <= low_threshold]

print("Bottom 25% by mean |SHAP|:")
print(low_importance.to_string())

# %%
