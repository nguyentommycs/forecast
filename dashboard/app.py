"""
Streamlit dashboard for real-time NYC taxi demand visualization.

Polls /predict/batch every 10 seconds and renders:
  - Bar chart of top 30 zones by predicted trip count
  - Side panel with a time series for the selected zone

Usage:
    streamlit run dashboard/app.py
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import redis
import requests
import streamlit as st

API_BASE = "http://localhost:8000"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
ALL_ZONES = list(range(1, 264))
REFRESH_INTERVAL = 10
ZONE_LOOKUP = Path(__file__).parent.parent / "reference_data" / "taxi_zone_lookup.csv"


@st.cache_data
def load_zone_lookup() -> pd.DataFrame:
    df = pd.read_csv(ZONE_LOOKUP)
    df.columns = [c.strip() for c in df.columns]
    return df.set_index("LocationID")


@st.cache_resource
def get_redis() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def fetch_predictions() -> dict[int, float]:
    zones_param = ",".join(str(z) for z in ALL_ZONES)
    resp = requests.get(
        f"{API_BASE}/predict/batch",
        params={"zones": zones_param},
        timeout=5,
    )
    resp.raise_for_status()
    return {p["zone_id"]: p["predicted_trip_count"] for p in resp.json()["predictions"]}


def fetch_zone_features(zone_id: int) -> dict | None:
    raw = get_redis().get(f"features:{zone_id}")
    return json.loads(raw) if raw else None


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="NYC Taxi Demand", layout="wide")
st.title("NYC Taxi Demand — Live Forecast")

if "pred_history" not in st.session_state:
    st.session_state.pred_history = {}

zone_lookup = load_zone_lookup()

with st.sidebar:
    st.header("Zone Inspector")
    zone_options = {
        f"{row['Zone']} ({row['Borough']})": zone_id
        for zone_id, row in zone_lookup.iterrows()
    }
    sorted_labels = sorted(zone_options)
    default_label = next((l for l, zid in zone_options.items() if zid == 132), sorted_labels[0])
    selected_label = st.selectbox("Zone", options=sorted_labels, index=sorted_labels.index(default_label))
    selected_zone = zone_options[selected_label]
    st.caption(f"Auto-refreshes every {REFRESH_INTERVAL}s")

try:
    preds = fetch_predictions()
except Exception as e:
    st.error(f"API unavailable: {e}")
    st.stop()

# ── Top zones bar chart ───────────────────────────────────────────────────────

df_preds = (
    pd.DataFrame(preds.items(), columns=["zone_id", "predicted_trips"])
    .sort_values("predicted_trips", ascending=False)
    .head(30)
)
df_preds["label"] = df_preds["zone_id"].apply(
    lambda z: zone_lookup.loc[z, "Zone"] if z in zone_lookup.index else str(z)
)

fig_bar = go.Figure(
    go.Bar(
        x=df_preds["predicted_trips"],
        y=df_preds["label"],
        orientation="h",
        marker_color=df_preds["predicted_trips"],
        marker_colorscale="YlOrRd",
        marker_showscale=True,
        marker_colorbar=dict(title="Trips/hr"),
    )
)
fig_bar.update_layout(
    title="Top 30 Zones by Predicted Demand",
    xaxis_title="Predicted Trips / hr",
    yaxis=dict(autorange="reversed"),
    height=600,
    margin={"t": 40, "b": 40},
)
st.plotly_chart(fig_bar, use_container_width=True)

# ── Zone time series ──────────────────────────────────────────────────────────

st.subheader(f"{selected_label} — Demand History & Prediction")

features = fetch_zone_features(selected_zone)
zone_pred = preds.get(selected_zone)

if features is not None and zone_pred is not None:
    zone_history = st.session_state.pred_history.setdefault(selected_zone, {})
    zone_history[features["bucket"]] = zone_pred
    if len(zone_history) > 8:
        del zone_history[sorted(zone_history)[0]]

if features is None:
    st.warning("No features in Redis for this zone yet.")
elif zone_pred is None:
    st.warning("No prediction available for this zone.")
else:
    current_bucket = datetime.fromisoformat(features["bucket"])

    hist_times = [current_bucket - timedelta(hours=3), current_bucket - timedelta(hours=2), current_bucket - timedelta(hours=1)]
    hist_values = [features["lag_3"], features["lag_2"], features["lag_1"]]
    pred_times = [current_bucket - timedelta(hours=1), current_bucket + timedelta(hours=1)]
    pred_values = [features["lag_1"], zone_pred]

    fig_ts = go.Figure()
    fig_ts.add_trace(
        go.Scatter(
            x=hist_times,
            y=hist_values,
            mode="lines+markers",
            name="Actuals",
            line=dict(color="#1f77b4"),
        )
    )
    fig_ts.add_trace(
        go.Scatter(
            x=pred_times,
            y=pred_values,
            mode="lines+markers",
            name="Prediction",
            line=dict(color="#ff7f0e", dash="dash"),
        )
    )
    fig_ts.update_layout(
        xaxis_title="Time",
        yaxis_title="Trip Count",
        height=300,
        margin={"t": 20},
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Predicted next hour", f"{zone_pred:.0f} trips")
    col2.metric("Last hour (actual)", f"{int(features['lag_1'])} trips")

    # ── Prediction vs Actuals ─────────────────────────────────────────────────

    st.subheader(f"{selected_label} — Prediction vs Actual (last 6 completed hours)")

    zone_history = st.session_state.pred_history.get(selected_zone, {})
    comparison_rows = []
    for n, lag_key in [(1, "lag_1"), (2, "lag_2"), (3, "lag_3"), (4, "lag_4"), (5, "lag_5"), (6, "lag_6")]:
        past_bucket_iso = (current_bucket - timedelta(hours=n)).isoformat()
        stored_pred = zone_history.get(past_bucket_iso)
        if stored_pred is not None:
            comparison_rows.append({
                "hour": (current_bucket - timedelta(hours=n)).strftime("%H:%M"),
                "predicted": stored_pred,
                "actual": features[lag_key],
            })

    if not comparison_rows:
        st.info("Accumulating prediction history — chart fills in as simulated hours pass.")
    else:
        df_cmp = pd.DataFrame(comparison_rows)
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(name="Predicted", x=df_cmp["hour"], y=df_cmp["predicted"], mode="lines+markers", line=dict(color="#ff7f0e")))
        fig_cmp.add_trace(go.Scatter(name="Actual", x=df_cmp["hour"], y=df_cmp["actual"], mode="lines+markers", line=dict(color="#1f77b4")))
        fig_cmp.update_layout(
            xaxis_title="Hour (simulated)",
            yaxis_title="Trip Count",
            height=300,
            margin={"t": 20},
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────

time.sleep(REFRESH_INTERVAL)
st.rerun()
