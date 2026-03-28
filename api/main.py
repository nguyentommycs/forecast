"""
FastAPI prediction service.

Loads the LightGBM model at startup, reads features from Redis,
and serves per-zone and batch trip-count predictions.

Usage:
    uvicorn api.main:app --port 8000
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import redis
from fastapi import FastAPI, HTTPException, Query

MODEL_FILE = Path(__file__).parent.parent / "models" / "model.lgb"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

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

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_FILE.exists():
        raise RuntimeError(f"Model not found: {MODEL_FILE}")
    _state["model"] = lgb.Booster(model_file=str(MODEL_FILE))
    _state["redis"] = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    _state["redis"].ping()
    yield
    _state.clear()


app = FastAPI(title="NYC Taxi Demand API", lifespan=lifespan)


def _get_features(zone_id: int) -> dict:
    raw = _state["redis"].get(f"features:{zone_id}")
    if raw is None:
        raise HTTPException(status_code=404, detail=f"No features for zone {zone_id}")
    return json.loads(raw)


def _predict(features_list: list[dict]) -> list[float]:
    df = pd.DataFrame(features_list, columns=FEATURE_COLS)[FEATURE_COLS]
    df["zone_id"] = df["zone_id"].astype("category")
    preds = np.clip(_state["model"].predict(df), 0, None)
    return preds.tolist()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict/batch")
def predict_batch(zones: str = Query(..., description="Comma-separated zone IDs, e.g. 1,2,3")):
    try:
        zone_ids = [int(z.strip()) for z in zones.split(",")]
    except ValueError:
        raise HTTPException(status_code=422, detail="zones must be comma-separated integers")

    features_list = []
    missing = []
    for zone_id in zone_ids:
        raw = _state["redis"].get(f"features:{zone_id}")
        if raw is None:
            missing.append(zone_id)
        else:
            features_list.append((zone_id, json.loads(raw)))

    preds = _predict([f for _, f in features_list]) if features_list else []

    results = [
        {"zone_id": zone_id, "predicted_trip_count": round(pred, 2)}
        for (zone_id, _), pred in zip(features_list, preds)
    ]
    response: dict = {"predictions": results}
    if missing:
        response["missing_zones"] = missing
    return response


@app.get("/predict/{zone_id}")
def predict_zone(zone_id: int):
    features = _get_features(zone_id)
    pred = _predict([features])[0]
    return {"zone_id": zone_id, "predicted_trip_count": round(pred, 2)}
