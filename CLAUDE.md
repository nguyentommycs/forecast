Project Details:
This is a time series forecasting project for NYC taxi cabs. The goal is to predict the trip count for the next 10 minutes per zone.

Objective:
Build a real-time taxi demand prediction system using NYC Yellow Taxi data.

Input: streaming taxi trip events
Output: predicted demand (trip count) per zone for the next 10 minutes
Latency target: < 200ms per prediction
Scope: local development, single machine

System Overview:

Components:
- Data Preprocessing + Model Training (offline)
- Streaming Ingestion (Kafka producer)
- Stream Consumer + Feature Engine
- Prediction API (FastAPI)
- Visualization (Streamlit)

Data Flow:
Historical CSV → preprocessing → training dataset
Train model → save artifact (model.json)
CSV replayed as event stream → Kafka
Consumer reads events → updates rolling features
API reads latest features → returns predictions

Tech Stack:
Streaming: Apache Kafka (Docker)
API: FastAPI
Model: XGBoost (or LightGBM)
Language: Python 3.10+
Optional UI: Streamlit

Coding Practices:
- Follow SOLID principles.
- Do not generate unit tests unless specifically asked to.
- Keep comments minimal, using them only for tricky or important logic