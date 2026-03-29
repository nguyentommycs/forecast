# NYC Taxi Demand Forecasting - Feature Analysis Report

**Date:** 2026-03-28
**Notebooks:** `01_feature_importance.py`, `02_ablation_study.py`, `03_candidate_features.py`
**Model:** LightGBM regressor
**Target:** trip count per zone per hour
**Train/test split:** chronological, last 2 weeks = test set

---

## 1. Initial Model State

### 1.1 Features

The production model uses 11 features derived from hourly aggregated trip counts per zone:

| Group | Feature | Description |
|---|---|---|
| Identity | `zone_id` | NYC TLC zone (1-263), treated as categorical |
| Time | `hour_of_day` | Hour extracted from pickup timestamp (0-23) |
| Time | `day_of_week` | Weekday (0=Mon, 6=Sun) |
| Time | `is_weekend` | Boolean flag (weekday >= 5) |
| Lag | `lag_1` | Trip count 1 hour prior (same zone) |
| Lag | `lag_2` | Trip count 2 hours prior |
| Lag | `lag_3` | Trip count 3 hours prior |
| Lag | `lag_6` | Trip count 6 hours prior |
| Lag | `lag_12` | Trip count 12 hours prior |
| Lag | `lag_24` | Trip count 24 hours prior (same hour yesterday) |
| Rolling | `rolling_mean_6` | Trailing mean of last 6 completed hours (same zone) |

### 1.2 Baseline Accuracy (test set = last 2 weeks)

| Metric | Value |
|---|---|
| RMSE | 9.334 trips/hr |
| MAE | 3.585 trips/hr |
| MAPE | 49.4% |

The high MAPE reflects skewed demand distribution: low-volume outer-borough zones dominate the zone count, and percentage errors are large when true values are near zero. RMSE and MAE are the more meaningful accuracy measures.

---

## 2. Experiment Methods and Objectives

### 2.1 Notebook 01 - Feature Importance Analysis

**Objective:** Understand which features the trained model relies on, using two complementary methods.

**Method A - LightGBM built-in importance:**
- *Gain*: total reduction in loss from splits on each feature. Dominated by the most informative single splits.
- *Split count*: number of times each feature was used in a split. Reflects breadth of use across the ensemble.

**Method B - SHAP values:**
- Computed via `TreeExplainer` on a 5,000-row sample of the test set.
- Mean |SHAP| measures each feature's average absolute contribution to individual predictions, accounting for feature interactions.

Gain and SHAP can disagree. Gain concentrates credit on the most impactful single split; SHAP distributes it more fairly across correlated features. Comparing both reveals which features are genuinely informative vs. inflated by correlation effects.

### 2.2 Notebook 02 - Ablation Study

**Objective:** Measure the actual accuracy impact of removing each feature.

**Method:**
- Train a baseline model on all 11 features.
- For each feature: drop it, retrain from scratch, record `d_rmse = RMSE_reduced - RMSE_baseline`.
- Positive `d_rmse` means the feature is load-bearing. Negative means removing it improves accuracy.
- Also test dropping entire feature groups (all lags, all time features).

Ablation is more reliable than importance metrics alone because it measures the causal impact of each feature rather than correlation-inflated scores.

### 2.3 Notebook 03 - Feature Engineering Experiments

**Objective:** Improve model accuracy based on findings from notebooks 01 and 02.

**Part 1 - Pruning:** Test removing the weakest features individually and in combination. Uses `features.parquet` directly — no new computation needed.

**Part 2 - New candidates:** Compute additional features from `aggregated.parquet` in one Polars pass. Test each against the best pruned set as the baseline. Candidates were selected based on what was absent from or underrepresented in the current feature set:

| Candidate | Rationale |
|---|---|
| `lag_168` | Same hour last week. Weekly seasonality is completely absent from the current feature set. |
| `rolling_mean_3` | Faster 3-hour window; captures rapid recent changes that `rolling_mean_6` smooths over. |
| `rolling_std_6` | 6-hour demand volatility — a different signal from the mean. |
| `rolling_mean_12` | Medium window bridging `rolling_mean_6` and the weekly lag. |
| `hour_sin` + `hour_cos` | Cyclical encoding treats 23:00 and 00:00 as adjacent, which linear `hour_of_day` does not. |

**Part 3 - Combined model:** Pruned set + all candidates that improved accuracy, compared to the original 11-feature baseline.

Note: The Section 1 and Section 2 baselines differ slightly in row count. Section 2 computes `lag_168`, which requires 168 prior rows per zone and drops more nulls than the 24-row minimum in Section 1. Both baselines are stated explicitly.

---

## 3. Results

### 3.1 Feature Importance (Notebook 01)

| Feature | Gain rank | Split rank | SHAP mean \|SHAP\| |
|---|---|---|---|
| lag_1 | 1 (dominant) | 2 | 22.17 |
| lag_24 | 2 | 4 | 5.77 |
| zone_id | 5 | 3 | 1.42 |
| hour_of_day | 3 | 1 | 1.42 |
| lag_2 | 6 | 9 | 1.25 |
| lag_3 | 7 | 8 | 0.61 |
| day_of_week | 8 | 7 | 0.48 |
| rolling_mean_6 | 9 | 10 | 0.37 |
| lag_6 | 10 | 6 | 0.30 |
| lag_12 | 11 | 5 | 0.21 |
| is_weekend | 12 | 11 | 0.08 |

Key observations:
- `lag_1` accounts for the vast majority of total gain. The model is overwhelmingly driven by the most recent observation.
- `lag_24` ranks second by both gain and SHAP, appearing important. Ablation tells a different story (see 3.2).
- `rolling_mean_6` has a high split count but ranks low by gain and SHAP, suggesting it provides broad but moderate contribution across many splits rather than highly impactful ones.
- `is_weekend` is the weakest feature by every metric.

### 3.2 Ablation Study (Notebook 02)

| Feature | d_rmse (removed) | Verdict |
|---|---|---|
| lag_1 | +2.00 | Critical - most load-bearing feature |
| zone_id | +1.15 | Critical - spatial dimension essential |
| hour_of_day | +1.10 | Critical - demand has strong hourly pattern |
| day_of_week | +0.25 | Moderate |
| lag_2 | +0.25 | Moderate |
| lag_6 | +0.23 | Moderate |
| lag_12 | +0.22 | Moderate |
| lag_3 | +0.22 | Moderate |
| rolling_mean_6 | +0.20 | Moderate |
| is_weekend | +0.15 | Marginal |
| lag_24 | +0.02 | Negligible - weakest feature |

Key finding: every feature contributes positively. The results confirm a clean and intuitive hierarchy: the immediate lag (`lag_1`) is most critical, spatial context (`zone_id`) and temporal context (`hour_of_day`) are next, and the remaining lags and rolling features form a cluster of moderate contributors. `lag_24` has the smallest contribution of any feature at +0.02 RMSE.

### 3.3 Feature Engineering Experiments (Notebook 03)

**Section 1 - Pruning (baseline RMSE = 9.334):**

| Experiment | Features | RMSE | d_rmse |
|---|---|---|---|
| drop lag_24 | 10 | 9.342 | +0.007 |
| drop lag_24 + lag_12 | 9 | 9.363 | +0.028 |
| drop lag_24 + lag_12 + is_weekend | 8 | 9.380 | +0.045 |
| drop is_weekend | 10 | 9.464 | +0.130 |
| drop lag_12 | 10 | 9.541 | +0.207 |

No pruning combination improved over the baseline. All 11 features contribute positively, confirming the ablation results. Dropping `lag_24` alone has a near-negligible cost (+0.007 RMSE), making it the only viable pruning candidate if a smaller feature set is desirable. Dropping `lag_12` or `is_weekend` alone incurs meaningful penalties (+0.207 and +0.130 respectively) and should not be done.

The best pruned set — drop `lag_24` only (10 features, RMSE=9.342) — was used as the Section 2 baseline.

**Section 2 - New candidates (baseline RMSE = 9.366 on enriched dataset):**

| Candidate | Features | RMSE | d_rmse | d_mae |
|---|---|---|---|---|
| lag_168 | 11 | 9.065 | -0.300 | -0.100 |
| hour_sin + hour_cos | 12 | 9.195 | -0.171 | -0.087 |
| rolling_mean_3 | 11 | 9.220 | -0.145 | -0.051 |
| rolling_mean_12 | 11 | 9.226 | -0.140 | -0.061 |
| rolling_std_6 | 11 | 9.255 | -0.110 | -0.048 |

All five candidates improved accuracy. `lag_168` is the standout winner (-0.300 RMSE). Weekly seasonality in NYC taxi demand is real and was completely absent from the current feature set. The cyclical hour encoding (`hour_sin` + `hour_cos`) is the second-best addition, suggesting that the hour-of-day boundary effect (midnight-1am vs. 11pm-midnight) is meaningful for the model. The rolling additions (`rolling_mean_3`, `rolling_mean_12`, `rolling_std_6`) each provide incremental but consistent gains.

**Section 3 - Best combined model:**

| Model | Features | RMSE | MAE | MAPE |
|---|---|---|---|---|
| Original 11-feature baseline | 11 | 9.384 | 3.585 | 49.4% |
| Best combined model | 16 | 8.954 | 3.470 | 48.7% |
| Improvement | -5 | -0.430 | -0.115 | -0.7pp |

Final feature set (16 features):
`zone_id`, `hour_of_day`, `day_of_week`, `is_weekend`, `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`, `rolling_mean_6`, `lag_168`, `rolling_mean_3`, `rolling_std_6`, `rolling_mean_12`, `hour_sin`, `hour_cos`

The combined model achieves a 4.6% RMSE reduction (9.384 -> 8.954) from entirely new information sources — weekly seasonality, demand volatility, additional rolling windows, and improved hour encoding.

---

## 4. Conclusions and Recommendations

### 4.1 Recommended changes to the production feature set

**Adopt the full 16-feature set from notebook 03 Section 3.** This delivers a 4.6% RMSE improvement over the baseline.

Changes from the current 11-feature set:

| Action | Feature | Reason | RMSE impact |
|---|---|---|---|
| Remove | `lag_24` | Negligible contribution (+0.007 when dropped); simplifies the feature set | +0.007 (negligible) |
| Add | `lag_168` | Weekly seasonality, the largest single improvement | -0.300 |
| Add | `hour_sin` + `hour_cos` | Cyclical hour encoding improves boundary handling | -0.171 |
| Add | `rolling_mean_3` | Captures rapid demand changes | -0.145 |
| Add | `rolling_mean_12` | Medium-window trend between 6h and weekly | -0.140 |
| Add | `rolling_std_6` | Demand volatility signal | -0.110 |

### 4.2 Implementation notes

**lag_168 requires extending the consumer's history buffer.**
`streaming/consumer.py` currently stores `HISTORY_LEN = 24` hourly buckets per zone. To compute `lag_168`, this must be extended to 168. The deque in `consumer.py` should be updated accordingly.

**All new rolling features use shift(1)** so each window covers only prior completed hours, consistent with `rolling_mean_6`. The same pattern must be used when adding these to `data/features.py` and the streaming consumer.

**hour_sin and hour_cos** are deterministic functions of `hour_of_day` and require no additional history:
```python
hour_sin = sin(hour_of_day * 2 * pi / 24)
hour_cos = cos(hour_of_day * 2 * pi / 24)
```

### 4.3 Features not recommended

| Feature | Reason |
|---|---|
| lag_48 | lag_24 is already the weakest lag; longer stale lags are unlikely to help |
| day_of_month | day_of_week already captures weekly patterns and is a moderate contributor |
| is_rush_hour | hour_of_day and the cyclical encoding already capture intraday patterns |

### 4.4 Priority order

1. Retrain the model on the updated `features.parquet`
2. Add `lag_168` and extend the consumer history buffer to 168 (highest single-feature gain)
3. Add the remaining new features: `rolling_mean_3`, `rolling_mean_12`, `rolling_std_6`, `hour_sin`, `hour_cos`
4. Remove `lag_24` from the feature set (optional — the gain is negligible but it simplifies the set)
