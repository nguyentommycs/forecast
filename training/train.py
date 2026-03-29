"""
Trains a LightGBM regressor to predict hourly trip count per zone.
Chronological train/test split (last 2 weeks = test).
Saves the trained model to models/model.lgb and a report to models/report.txt.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES_FILE = Path(__file__).parent.parent / "processed_data" / "features.parquet"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_FILE = MODEL_DIR / "model.lgb"
REPORT_FILE = MODEL_DIR / "report.txt"

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
TEST_WEEKS = 2


def split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    cutoff = df["time_bucket"].max() - pl.duration(weeks=TEST_WEEKS)
    return df.filter(pl.col("time_bucket") <= cutoff), df.filter(pl.col("time_bucket") > cutoff)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    rmse = mean_squared_error(actual, predicted) ** 0.5
    mae = mean_absolute_error(actual, predicted)
    nonzero = actual > 0
    mape = (np.abs(predicted[nonzero] - actual[nonzero]) / actual[nonzero]).mean() * 100 if nonzero.any() else float("nan")
    return {"rmse": rmse, "mae": mae, "mape": mape}


def build_report(
    df_full: pl.DataFrame,
    test_df: pd.DataFrame,
    preds: np.ndarray,
) -> str:
    lines = []

    # --- Overall metrics ---
    actual = test_df[TARGET_COL].to_numpy()
    m = compute_metrics(actual, preds)
    avg_volume_all = df_full[TARGET_COL].mean()

    lines.append("=" * 60)
    lines.append("OVERALL (test set — last 2 weeks)")
    lines.append("=" * 60)
    lines.append(f"  RMSE:              {m['rmse']:>8.2f} trips/hr")
    lines.append(f"  MAE:               {m['mae']:>8.2f} trips/hr")
    lines.append(f"  MAPE:              {m['mape']:>7.1f}%")
    lines.append(f"  Avg trip volume:   {avg_volume_all:>8.2f} trips/hr  (full dataset)")
    lines.append("")

    # --- Per-zone metrics ---
    lines.append("=" * 60)
    lines.append(f"{'Zone':>6}  {'RMSE':>7}  {'MAE':>7}  {'MAPE':>7}  {'Avg Vol':>8}")
    lines.append("-" * 60)

    results = test_df[["zone_id", TARGET_COL]].copy()
    results["pred"] = preds

    avg_vol_by_zone = (
        df_full.group_by("zone_id")
        .agg(pl.col("trip_count").mean().alias("avg_vol"))
        .to_pandas()
        .set_index("zone_id")["avg_vol"]
    )

    zone_rows = []
    for zone_id, grp in results.groupby("zone_id", sort=True):
        zm = compute_metrics(grp[TARGET_COL].to_numpy(), grp["pred"].to_numpy())
        avg_vol = avg_vol_by_zone.get(zone_id, float("nan"))
        zone_rows.append((zone_id, zm["rmse"], zm["mae"], zm["mape"], avg_vol))

    zone_rows.sort(key=lambda r: r[1], reverse=True)  # sort by RMSE descending
    for zone_id, rmse, mae, mape, avg_vol in zone_rows:
        lines.append(f"{zone_id:>6}  {rmse:>7.2f}  {mae:>7.2f}  {mape:>6.1f}%  {avg_vol:>8.2f}")

    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> None:
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Features file not found: {FEATURES_FILE}\n"
            "Run data/preprocessing.py then data/features.py first."
        )

    print("Loading features...")
    df = pl.read_parquet(FEATURES_FILE)
    print(f"  {len(df):,} rows, {df['zone_id'].n_unique()} zones")

    train_df, test_df = split(df)
    print(f"  Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")
    print(f"  Test cutoff: {test_df['time_bucket'].min()}")

    X_train = train_df[FEATURE_COLS].to_pandas()
    y_train = train_df[TARGET_COL].to_pandas()
    X_test = test_df[FEATURE_COLS].to_pandas()
    y_test = test_df[TARGET_COL].to_pandas()

    # zone_id is a categorical identifier, not an ordinal number
    X_train["zone_id"] = X_train["zone_id"].astype("category")
    X_test["zone_id"] = X_test["zone_id"].astype("category")

    print("\nTraining LightGBM model...")
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        categorical_feature=["zone_id"],
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    preds = np.clip(model.predict(X_test), 0, None)

    # Build evaluation dataframe (preserve zone_id alongside actuals)
    test_eval = test_df[FEATURE_COLS + [TARGET_COL]].to_pandas()
    test_eval["pred"] = preds

    report = build_report(df, test_eval, preds)
    print("\n" + report)

    MODEL_DIR.mkdir(exist_ok=True)
    model.booster_.save_model(str(MODEL_FILE))
    REPORT_FILE.write_text(report)
    print(f"\nModel saved  -> {MODEL_FILE}")
    print(f"Report saved -> {REPORT_FILE}")
    print(f"Best iteration: {model.best_iteration_}")


if __name__ == "__main__":
    main()
