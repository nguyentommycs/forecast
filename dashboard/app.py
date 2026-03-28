"""
Streamlit dashboard for real-time NYC taxi demand visualization.

Polls /predict/batch every 10 seconds and renders:
  - Bar chart of top 30 zones by predicted trip count
  - Side panel with a time series for the selected zone

Usage:
    streamlit run dashboard/app.py
"""

import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Record predictions for all zones in one Redis pipeline so history is
# available immediately when the user switches zones.
pipe = get_redis().pipeline()
for zone_id in preds:
    pipe.get(f"features:{zone_id}")
zone_avg_traffic: dict[int, float] = {}
for zone_id, raw in zip(preds, pipe.execute()):
    if raw is not None:
        f = json.loads(raw)
        zone_history = st.session_state.pred_history.setdefault(zone_id, {})
        zone_history[f["bucket"]] = preds[zone_id]
        if len(zone_history) > 8:
            del zone_history[sorted(zone_history)[0]]
        zone_avg_traffic[zone_id] = sum(f.get(f"lag_{n}", 0) for n in range(1, 7)) / 6

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

if features is None:
    st.warning("No features in Redis for this zone yet.")
elif zone_pred is None:
    st.warning("No prediction available for this zone.")
else:
    current_bucket = datetime.fromisoformat(features["bucket"])

    with st.sidebar:
        st.divider()
        st.caption("Simulated time")
        st.write((current_bucket - timedelta(hours=1)).strftime("%a %b %d, %Y  %H:%M"))

    hist_times = [current_bucket - timedelta(hours=3), current_bucket - timedelta(hours=2), current_bucket - timedelta(hours=1)]
    hist_values = [features["lag_3"], features["lag_2"], features["lag_1"]]
    pred_times = [current_bucket - timedelta(hours=1), current_bucket]
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
        xaxis=dict(title="Time", dtick=3600000, tickformat="%H:%M"),
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
    for n, lag_key in [(6, "lag_6"), (5, "lag_5"), (4, "lag_4"), (3, "lag_3"), (2, "lag_2"), (1, "lag_1")]:
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

# ── Zone comparison: predictions vs actuals (small multiples) ────────────────

st.subheader("Zone Comparison — Predictions vs Actuals")

top9_default_ids = sorted(zone_avg_traffic, key=zone_avg_traffic.get, reverse=True)[:9]
top9_default_labels = [
    next((lbl for lbl, zid in zone_options.items() if zid == z), None)
    for z in top9_default_ids
]
top9_default_labels = [l for l in top9_default_labels if l is not None]

selected_comparison_labels = st.multiselect(
    "Zones",
    options=sorted_labels,
    default=top9_default_labels,
    key="comparison_zone_selector",
    max_selections=9,
)
selected_comparison_ids = [zone_options[l] for l in selected_comparison_labels]

if not selected_comparison_ids:
    st.info("Select at least one zone above.")
else:
    n_zones = len(selected_comparison_ids)
    n_cols = 3
    n_rows = math.ceil(n_zones / n_cols)

    subplot_titles_comp = [
        zone_lookup.loc[z, "Zone"] if z in zone_lookup.index else str(z)
        for z in selected_comparison_ids
    ] + [""] * (n_rows * n_cols - n_zones)

    pipe2 = get_redis().pipeline()
    for z in selected_comparison_ids:
        pipe2.get(f"features:{z}")
    comp_features = {
        z: (json.loads(raw) if raw else None)
        for z, raw in zip(selected_comparison_ids, pipe2.execute())
    }

    fig_comp = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles_comp)

    PRED_COLOR = "#ff7f0e"
    ACTUAL_COLOR = "#1f77b4"

    for idx, zone_id in enumerate(selected_comparison_ids):
        row, col = divmod(idx, n_cols)
        row, col = row + 1, col + 1
        show_legend = idx == 0

        f = comp_features.get(zone_id)
        if f is None:
            continue

        bucket = datetime.fromisoformat(f["bucket"])
        zone_hist = st.session_state.pred_history.get(zone_id, {})

        # Historical predictions T-5h … T-1h (solid)
        hist_pred_times, hist_pred_vals = [], []
        for n in range(5, 0, -1):
            val = zone_hist.get((bucket - timedelta(hours=n)).isoformat())
            if val is not None:
                hist_pred_times.append(bucket - timedelta(hours=n))
                hist_pred_vals.append(val)

        if hist_pred_times:
            fig_comp.add_trace(go.Scatter(
                x=hist_pred_times, y=hist_pred_vals,
                mode="lines+markers", name="Prediction",
                legendgroup="prediction", line=dict(color=PRED_COLOR),
                showlegend=show_legend,
            ), row=row, col=col)

            curr_pred = preds.get(zone_id)
            if curr_pred is not None:
                fig_comp.add_trace(go.Scatter(
                    x=[hist_pred_times[-1], bucket],
                    y=[hist_pred_vals[-1], curr_pred],
                    mode="lines+markers", name="Prediction",
                    legendgroup="prediction", line=dict(color=PRED_COLOR, dash="dash"),
                    showlegend=False,
                ), row=row, col=col)

        # Actuals T-5h … T-1h
        fig_comp.add_trace(go.Scatter(
            x=[bucket - timedelta(hours=n) for n in range(5, 0, -1)],
            y=[f.get(f"lag_{n}", 0) for n in range(5, 0, -1)],
            mode="lines+markers", name="Actual",
            legendgroup="actual", line=dict(color=ACTUAL_COLOR),
            showlegend=show_legend,
        ), row=row, col=col)

    fig_comp.update_xaxes(dtick=3600000, tickformat="%H:%M")
    fig_comp.update_layout(
        height=300 * n_rows + 60,
        margin={"t": 60, "b": 20},
        legend=dict(orientation="h", y=1.04),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────

time.sleep(REFRESH_INTERVAL)
st.rerun()
