# =============================================================================
# TwinMesh App · api.py
# FastAPI backend — all endpoints for the Streamlit frontend
#
# Run with:
#   cd TwinMesh
#   uvicorn app.backend.api:app --host 0.0.0.0 --port 8000 --reload
# =============================================================================

import os
import sys
from typing import Optional
from contextlib import asynccontextmanager

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.backend.prediction_service import (
    load_models, predict, get_all_port_status
)
from app.backend.replay_service import (
    get_port_replay, get_replay_summary, get_alert_feed, get_hawkes_series_all
)
from app.backend.policy_service import (
    get_policy_for_state, get_full_policy_table,
    get_safety_report, get_cost_comparison_all_states,
    get_action_explanation,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    print("TwinMesh API starting — loading models...")
    ok = load_models()
    print(f"Models loaded: {ok}")
    yield


app = FastAPI(
    title       = "TwinMesh API",
    description = "Supply chain disruption prediction + adaptive consistency",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "service": "TwinMesh API v1.0"}


# ── Prediction endpoints ──────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    port:               str   = Field(..., example="Rotterdam")
    current_state:      str   = Field(..., example="S2")
    congestion_score:   float = Field(2.5, ge=0, le=20)
    net_dwt_flow:       float = Field(0.0)
    container_ratio:    float = Field(0.5, ge=0, le=1)
    queue_buildup_7d:   float = Field(0.0, ge=0)
    yoy_pct_change:     float = Field(0.0)
    anomaly_flag:       int   = Field(0, ge=0, le=1)
    weather_risk_score: float = Field(0.0, ge=0, le=1)
    storm_alert:        int   = Field(0, ge=0, le=1)
    heavy_rain_alert:   int   = Field(0, ge=0, le=1)
    date:               Optional[str] = None


@app.post("/predict", tags=["Prediction"])
def predict_endpoint(req: PredictRequest):
    """
    Run TwinMesh prediction for a single port observation.
    Returns predicted next state, P(S3), Hawkes λ(t), alert, and MDP actions.
    """
    try:
        result = predict(
            port               = req.port,
            current_state      = req.current_state,
            congestion_score   = req.congestion_score,
            net_dwt_flow       = req.net_dwt_flow,
            container_ratio    = req.container_ratio,
            queue_buildup_7d   = req.queue_buildup_7d,
            yoy_pct_change     = req.yoy_pct_change,
            anomaly_flag       = req.anomaly_flag,
            weather_risk_score = req.weather_risk_score,
            storm_alert        = req.storm_alert,
            heavy_rain_alert   = req.heavy_rain_alert,
            date               = req.date,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ports/status", tags=["Prediction"])
def ports_status(date: Optional[str] = Query(None)):
    """
    Returns current prediction for all 7 ports.
    Used to populate the overview dashboard.
    """
    try:
        return get_all_port_status(date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Replay endpoints ──────────────────────────────────────────────────────────

@app.get("/replay/{port}", tags=["Replay"])
def replay_port(
    port:       str,
    start_date: Optional[str] = Query(None),
    end_date:   Optional[str] = Query(None),
):
    """
    Returns day-by-day replay data for one port:
    actual state, predicted state, P(S3), Hawkes λ(t), alerts.
    """
    data = get_port_replay(port, start_date, end_date)
    if not data:
        raise HTTPException(status_code=404, detail=f"No replay data for {port}")
    return data


@app.get("/replay/summary/all", tags=["Replay"])
def replay_summary():
    """Overall replay statistics: total decisions, alerts, accuracy."""
    return get_replay_summary()


@app.get("/alerts", tags=["Replay"])
def alert_feed(limit: int = Query(20, ge=1, le=100)):
    """Returns most recent Hawkes alerts across all ports."""
    return get_alert_feed(limit)


@app.get("/hawkes/series", tags=["Replay"])
def hawkes_series(
    start_date: Optional[str] = Query(None),
    end_date:   Optional[str] = Query(None),
):
    """
    Returns Hawkes λ(t) time series for all 7 ports.
    Used for the intensity chart in the Replay screen.
    """
    return get_hawkes_series_all(start_date, end_date)


# ── Policy endpoints ──────────────────────────────────────────────────────────

@app.get("/policy/{state}", tags=["Policy"])
def policy_for_state(state: str):
    """
    Returns MDP recommended action for each tier given current disruption state.
    """
    if state not in ("S0","S1","S2","S3","S4"):
        raise HTTPException(status_code=400, detail=f"Invalid state: {state}")
    return get_policy_for_state(state)


@app.get("/policy/table/full", tags=["Policy"])
def policy_table():
    """Full 5-state × 4-tier policy matrix."""
    return get_full_policy_table()


@app.get("/policy/safety/report", tags=["Policy"])
def safety_report():
    """
    MDP safety report from Day 5:
    tier-0 violations, cost reduction vs AlwaysStrong.
    """
    return get_safety_report()


@app.get("/policy/cost/comparison", tags=["Policy"])
def cost_comparison():
    """Per-state cost comparison: policy vs AlwaysStrong."""
    return get_cost_comparison_all_states()


@app.get("/policy/action/{action}", tags=["Policy"])
def action_explanation(action: str):
    """Explains what a given consistency action means."""
    return get_action_explanation(action)