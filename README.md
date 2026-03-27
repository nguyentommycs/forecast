# NYC Taxi Demand Forecasting

Real-time taxi demand prediction system using NYC Yellow Taxi data. Predicts trip count per zone for the next 10 minutes with <200ms API latency, running locally on a single machine.

**Stack:** LightGBM · FastAPI · Apache Kafka · Redis · Streamlit · Python 3.10+

---

## Prerequisites

- Python 3.10+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Kafka, Zookeeper, Redis)

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data Preparation

Run these steps once before starting the streaming pipeline.

```bash
# Aggregate raw parquet data into 10-minute trip count buckets
python data/preprocessing.py

# Build the supervised learning feature dataset
python data/features.py

# Train the LightGBM model — saves to models/model.lgb, prints RMSE/MAE
python training/train.py
```

---

## Running the System

### Start infrastructure (Kafka + Zookeeper + Redis)

```bash
docker compose up -d
```

Services exposed:
- Zookeeper: `localhost:2181`
- Kafka broker: `localhost:9092`
- Redis: `localhost:6379`

### Start the streaming pipeline

```bash
# Terminal 1 — consumer (populates Redis with live features)
python streaming/consumer.py

# Terminal 2 — producer (replays historical data into Kafka)
python streaming/producer.py
```
