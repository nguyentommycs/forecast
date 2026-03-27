"""
Streamlit dashboard for real-time NYC taxi demand visualization.

Polls /predict/batch every 10 seconds and renders:
  - Choropleth map of all zones colored by predicted trip count
  - Side panel with a time series for the selected zone

Usage:
    streamlit run dashboard/app.py
"""

import json
import time
from datetime import datetime, timedelta

import plotly.graph_objects as go
import redis
import requests
import streamlit as st

API_BASE = "http://localhost:8000"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
ALL_ZONES = list(range(1, 264))
REFRESH_INTERVAL = 10
GEOJSON_URL = (
    "https://data.cityofnewyork.us/api/geospatial/d3c5-ddgc"
    "?method=export&type=GeoJSON"
)


@st.cache_data(ttl=86400)
def load_geojson() -> dict:
    resp = requests.get(GEOJSON_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


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

with st.sidebar:
    st.header("Zone Inspector")
    selected_zone = st.number_input("Zone ID", min_value=1, max_value=263, value=132)
    st.caption(f"Auto-refreshes every {REFRESH_INTERVAL}s")

geojson = load_geojson()

try:
    preds = fetch_predictions()
except Exception as e:
    st.error(f"API unavailable: {e}")
    st.stop()

# ── Choropleth map ────────────────────────────────────────────────────────────

zone_ids = list(preds.keys())
trip_counts = list(preds.values())

fig_map = go.Figure(
    go.Choroplethmapbox(
        geojson=geojson,
        locations=[str(z) for z in zone_ids],
        z=trip_counts,
        featureidkey="properties.location_id",
        colorscale="YlOrRd",
        zmin=0,
        zmax=max(trip_counts) if trip_counts else 100,
        marker_opacity=0.7,
        marker_line_width=0.5,
        colorbar_title="Predicted<br>Trips/hr",
    )
)
fig_map.update_layout(
    mapbox_style="carto-positron",
    mapbox_zoom=9.5,
    mapbox_center={"lat": 40.7128, "lon": -74.0060},
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    height=550,
)
st.plotly_chart(fig_map, use_container_width=True)

# ── Zone time series ──────────────────────────────────────────────────────────

st.subheader(f"Zone {selected_zone} — Demand History & Prediction")

features = fetch_zone_features(selected_zone)
zone_pred = preds.get(selected_zone)

if features is None:
    st.warning("No features in Redis for this zone yet.")
elif zone_pred is None:
    st.warning("No prediction available for this zone.")
else:
    now = datetime.now().replace(minute=0, second=0, microsecond=0)

    # 3 most recent consecutive hourly actuals + prediction for the next hour
    hist_times = [now - timedelta(hours=3), now - timedelta(hours=2), now - timedelta(hours=1)]
    hist_values = [features["lag_3"], features["lag_2"], features["lag_1"]]
    pred_times = [now - timedelta(hours=1), now + timedelta(hours=1)]
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

# ── Auto-refresh ──────────────────────────────────────────────────────────────

time.sleep(REFRESH_INTERVAL)
st.rerun()
