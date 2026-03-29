"""Shared utilities for feature analysis notebooks."""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import LGBM_PARAMS  # noqa: E402

FEATURES_FILE = ROOT / "processed_data" / "features.parquet"
AGGREGATED_FILE = ROOT / "processed_data" / "aggregated.parquet"
MODEL_FILE = ROOT / "models" / "model.lgb"

FEATURE_COLS = [
    "zone_id", "hour_of_day", "day_of_week", "is_weekend",
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "lag_24",
    "rolling_mean_6",
]
TARGET_COL = "trip_count"
TEST_WEEKS = 2


def load_booster() -> lgb.Booster:
    """Load the trained LightGBM booster, stripping the sklearn pandas_categorical
    footer that causes a fatal parse error in LightGBM 4.6.0's native loader."""
    text = MODEL_FILE.read_text()
    marker = "\npandas_categorical:"
    clean = text[: text.find(marker)] if marker in text else text
    return lgb.Booster(model_str=clean)


def load_features() -> pl.DataFrame:
    return pl.read_parquet(FEATURES_FILE)


def load_aggregated() -> pl.DataFrame:
    return pl.read_parquet(AGGREGATED_FILE)


def split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    cutoff = df["time_bucket"].max() - pl.duration(weeks=TEST_WEEKS)
    return df.filter(pl.col("time_bucket") <= cutoff), df.filter(pl.col("time_bucket") > cutoff)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    rmse = mean_squared_error(actual, predicted) ** 0.5
    mae = mean_absolute_error(actual, predicted)
    nonzero = actual > 0
    mape = (
        (np.abs(predicted[nonzero] - actual[nonzero]) / actual[nonzero]).mean() * 100
        if nonzero.any() else float("nan")
    )
    return {"rmse": rmse, "mae": mae, "mape": mape}


def train_and_evaluate(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Train a LightGBM model on the given feature set and return metrics + fitted model."""
    X_train = train_df[feature_cols].to_pandas()
    y_train = train_df[TARGET_COL].to_pandas()
    X_test = test_df[feature_cols].to_pandas()
    y_test = test_df[TARGET_COL].to_pandas()

    cat_features = ["zone_id"] if "zone_id" in feature_cols else []
    for col in cat_features:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        categorical_feature=cat_features if cat_features else "auto",
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    preds = np.clip(model.predict(X_test), 0, None)
    metrics = compute_metrics(y_test.to_numpy(), preds)
    return {"model": model, "preds": preds, "X_test": X_test, "y_test": y_test, **metrics}
