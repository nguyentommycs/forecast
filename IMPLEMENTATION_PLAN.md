# NYC Taxi Demand Forecasting — Implementation Plan

## Overview
Real-time taxi demand prediction system using NYC Yellow Taxi data. Predicts trip count per zone for the next hour with <200ms API latency, running locally on a single machine.

**Stack:** LightGBM · FastAPI · Apache Kafka (Docker) · Redis · Streamlit · Python 3.10+

---

## Project Structure
```
forecast/
├── data/
│   ├── preprocessing.py        # Load, clean, aggregate raw parquet data
│   └── features.py             # Feature engineering for training dataset
├── training/
│   └── train.py                # Train LightGBM model, save artifact
├── streaming/
│   ├── producer.py             # Kafka producer: replays historical data as event stream
│   └── consumer.py             # Kafka consumer + feature engine → writes to Redis
├── api/
│   └── main.py                 # FastAPI prediction service
├── dashboard/
│   └── app.py                  # Streamlit visualization
├── models/                     # Saved model artifacts (gitignored)
├── processed_data/             # Intermediate processed data (gitignored)
├── docker-compose.yml          # Kafka + Zookeeper + Redis
└── requirements.txt            # Full dependency list
```

---

## Tasks

### Phase 1 — Environment & Dependencies

- [ ] **1.1** Update `requirements.txt` with all dependencies:
  - `lightgbm`, `scikit-learn` (model)
  - `fastapi`, `uvicorn[standard]` (API)
  - `confluent-kafka` (streaming)
  - `redis` (feature store)
  - `streamlit`, `plotly` (dashboard)
  - `numpy` (numerics)

- [ ] **1.2** Create `docker-compose.yml` with:
  - Zookeeper (port 2181)
  - Kafka broker (port 9092, auto topic creation enabled)
  - Redis (port 6379)

---

### Phase 2 — Data Preprocessing (`data/preprocessing.py`)

- [ ] **2.1** Load all 8 parquet files from `raw_data/` using Polars
- [ ] **2.2** Keep only `tpep_pickup_datetime` and `PULocationID` columns
- [ ] **2.3** Drop nulls; filter zone IDs to valid range (1–263)
- [ ] **2.4** Floor timestamps to 10-minute buckets
- [ ] **2.5** Aggregate: count trips per `(zone_id, time_bucket)` → `trip_count`
- [ ] **2.6** Save to `processed_data/aggregated.parquet`

---

### Phase 3 — Feature Engineering (`data/features.py`)

Transform the aggregated time series into a supervised learning dataset.

- [ ] **3.1** For each `(zone_id, time_bucket)` row, compute:
  - `hour_of_day` (0–23)
  - `day_of_week` (0–6)
  - `is_weekend` (bool)
  - `lag_1` — trip count 10 min ago
  - `lag_2` — trip count 20 min ago
  - `lag_3` — trip count 30 min ago
  - `lag_6` — trip count 1 hour ago
  - `lag_12` — trip count 2 hours ago
  - `rolling_mean_6` — mean of last 6 windows (1 hour)
  - `zone_id` (integer, used as categorical feature)
- [ ] **3.2** Drop rows with NaN lags (start of each zone's history)
- [ ] **3.3** Save to `processed_data/features.parquet`

**Target:** `trip_count` for the current window

---

### Phase 4 — Model Training (`training/train.py`)

- [ ] **4.1** Load `processed_data/features.parquet`
- [ ] **4.2** Chronological train/test split (last 2 weeks = test)
- [ ] **4.3** Train `lgb.LGBMRegressor` (tune: `n_estimators`, `learning_rate`, `num_leaves`)
- [ ] **4.4** Evaluate: print RMSE and MAE on test set
- [ ] **4.5** Save model to `models/model.lgb`

---

### Phase 5 — Kafka Streaming (`streaming/`)

#### Producer (`streaming/producer.py`)
- [ ] **5.1** Load `processed_data/aggregated.parquet`, sort by `time_bucket`
- [ ] **5.2** Publish each row as JSON `{zone_id, timestamp, trip_count}` to topic `taxi-trips`
- [ ] **5.3** Support configurable replay speed via `--speed` CLI arg (default: accelerated)

#### Consumer + Feature Engine (`streaming/consumer.py`)
- [ ] **5.4** Subscribe to `taxi-trips` topic
- [ ] **5.5** Maintain `Dict[zone_id, deque(maxlen=12)]` — rolling 12-window history per zone
- [ ] **5.6** On each event: update deque, compute the 9 features
- [ ] **5.7** Write to Redis: key `features:{zone_id}`, value JSON-encoded feature dict

---

### Phase 6 — Prediction API (`api/main.py`)

- [ ] **6.1** Load `models/model.lgb` at startup (singleton)
- [ ] **6.2** Connect to Redis at startup
- [ ] **6.3** Implement `GET /health` — returns `{"status": "ok"}`
- [ ] **6.4** Implement `GET /predict/{zone_id}` — reads features from Redis, returns prediction
- [ ] **6.5** Implement `GET /predict/batch?zones=1,2,3` — batch predictions for multiple zones

---

### Phase 7 — Streamlit Dashboard (`dashboard/app.py`)

- [ ] **7.1** Fetch all zone predictions from `/predict/batch` every 10 seconds
- [ ] **7.2** Render NYC zone choropleth map (Plotly + zone GeoJSON), color-coded by predicted demand
- [ ] **7.3** Side panel: time series chart for selected zone (last 1 hour of actuals + prediction)

---

## Data Flow Summary
```
raw_data/*.parquet
    → preprocessing.py → processed_data/aggregated.parquet
    → features.py      → processed_data/features.parquet
    → train.py         → models/model.lgb

[Runtime]
processed_data/aggregated.parquet
    → producer.py → Kafka (taxi-trips topic)
    → consumer.py → Redis (features:{zone_id})
    → api/main.py → GET /predict/{zone_id}
    → dashboard/app.py (live map)
```

---

## Verification Checklist
1. `python data/preprocessing.py` — produces `processed_data/aggregated.parquet`
2. `python data/features.py` — produces `processed_data/features.parquet`
3. `python training/train.py` — produces `models/model.lgb`, prints RMSE/MAE
4. `docker compose up -d` — Kafka/Zookeeper/Redis on ports 2181/9092/6379
5. `python streaming/consumer.py` — consumer starts, features populate Redis
6. `python streaming/producer.py` — events stream through Kafka
7. `uvicorn api.main:app --port 8000` — `curl localhost:8000/predict/132` returns prediction
8. `streamlit run dashboard/app.py` — live demand map opens in browser
