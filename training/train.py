"""
Trains a LightGBM regressor to predict hourly trip count per zone.
Chronological train/test split (last 2 weeks = test).
Saves the trained model to models/model.lgb.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES_FILE = Path(__file__).parent.parent / "processed_data" / "features.parquet"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_FILE = MODEL_DIR / "model.lgb"

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
    "lag_24",
    "rolling_mean_6",
]
TARGET_COL = "trip_count"
TEST_WEEKS = 2


def split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    cutoff = df["time_bucket"].max() - pl.duration(weeks=TEST_WEEKS)
    return df.filter(pl.col("time_bucket") <= cutoff), df.filter(pl.col("time_bucket") > cutoff)


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

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)  # trip counts can't be negative

    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    # MAPE only over rows where actual > 0 to avoid division by zero
    nonzero = y_test > 0
    mape = (np.abs(preds[nonzero] - y_test[nonzero]) / y_test[nonzero]).mean() * 100

    print(f"\nTest RMSE: {rmse:.2f}")
    print(f"Test MAE:  {mae:.2f}")
    print(f"Test MAPE: {mape:.1f}%")

    MODEL_DIR.mkdir(exist_ok=True)
    model.booster_.save_model(str(MODEL_FILE))
    print(f"\nModel saved -> {MODEL_FILE}")
    print(f"  Best iteration: {model.best_iteration_}")


if __name__ == "__main__":
    main()
