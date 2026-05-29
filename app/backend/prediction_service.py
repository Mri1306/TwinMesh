# =============================================================================
# TwinMesh App · prediction_service.py
# Core prediction logic — loads fitted models + runs inference
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
from typing import Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)

from src.utils.constants import STATES, STATE_MAP, PORTS, PROCESSED_DIR, UNIFIED_DIR
from src.utils.logger import get_logger

log = get_logger("prediction_service")

PORT_NAMES = list(PORTS.keys())

DECAY = 0.5
ALERT_THRESH = 0.30

HAWKES_DIR = os.path.join(ROOT, "results", "hawkes_standalone")
COMBINED_F = os.path.join(PROCESSED_DIR, "all_ports_combined.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Singleton model holder
# ─────────────────────────────────────────────────────────────────────────────

class ModelStore:
    _instance = None

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = []
        self.ccf_df = None
        self.mu_s = None
        self.onset_cache = {}
        self.policy_map = {}
        self.cost_df = None
        self.t0 = None
        self.loaded = False
        self.combined_df = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = ModelStore()
        return cls._instance


# ─────────────────────────────────────────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────────────────────────────────────────

def load_models() -> bool:

    store = ModelStore.get()

    if store.loaded:
        return True

    try:

        # ---------------------------------------------------------------------
        # Load combined dataset
        # ---------------------------------------------------------------------

        store.combined_df = pd.read_csv(
            COMBINED_F,
            parse_dates=["date"]
        )

        store.t0 = store.combined_df["date"].min()

        # ---------------------------------------------------------------------
        # Load Hawkes matrices
        # ---------------------------------------------------------------------

        ccf_path = os.path.join(
            HAWKES_DIR,
            "hawkes_excitation_matrices",
            "ccf_7x7.csv"
        )

        mu_path = os.path.join(
            HAWKES_DIR,
            "hawkes_excitation_matrices",
            "port_level_mu.csv"
        )

        if os.path.exists(ccf_path):

            store.ccf_df = pd.read_csv(
                ccf_path,
                index_col=0
            )

        else:

            store.ccf_df = pd.DataFrame(
                np.eye(len(PORT_NAMES)) * 0.05,
                index=PORT_NAMES,
                columns=PORT_NAMES
            )

        if os.path.exists(mu_path):

            store.mu_s = pd.read_csv(
                mu_path,
                index_col=0
            ).squeeze()

        else:

            store.mu_s = pd.Series(
                0.08,
                index=PORT_NAMES
            )

        # ---------------------------------------------------------------------
        # Build onset cache
        # ---------------------------------------------------------------------

        df = store.combined_df.copy()

        df["t_numeric"] = (
            (df["date"] - store.t0)
            .dt.days
            .astype(float)
        )

        for port in PORT_NAMES:

            pdf = (
                df[df["port"] == port]
                .sort_values("date")
                .reset_index(drop=True)
            )

            is_s3 = pdf["state"] == "S3"
            prev = pdf["state"].shift(1) != "S3"

            mask = is_s3 & prev

            if len(mask):
                mask.iloc[0] = is_s3.iloc[0]

            store.onset_cache[port] = (
                pdf[mask]["t_numeric"]
                .values
                .tolist()
            )

        # ---------------------------------------------------------------------
        # Load training data
        # ---------------------------------------------------------------------

        train = pd.read_csv(
            os.path.join(UNIFIED_DIR, "twinmesh_training.csv"),
            parse_dates=["date"]
        )

        store.combined_df["t_numeric"] = (
            store.combined_df["date"] - store.t0
        ).dt.days.astype(float)

        # ---------------------------------------------------------------------
        # Try V2 model
        # ---------------------------------------------------------------------

        try:

            from src.models.combined.twinmesh_core_v2 import (
                prepare_sequences_v2,
                fit_v2_model,
                compute_hawkes_intensity,
                FEATURES_COMBINED_V2
            )

            lam_df = compute_hawkes_intensity(store.combined_df)

            lam_cols = lam_df[
                [
                    "date",
                    "port",
                    "hawkes_lambda",
                    "hawkes_delta_lambda"
                ]
            ]

            tr = train.merge(
                lam_cols,
                on=["date", "port"],
                how="left"
            )

            tr[
                ["hawkes_lambda", "hawkes_delta_lambda"]
            ] = tr[
                ["hawkes_lambda", "hawkes_delta_lambda"]
            ].fillna(
                {
                    "hawkes_lambda": 0.08,
                    "hawkes_delta_lambda": 0.0
                }
            )

            X_tr, y_tr = prepare_sequences_v2(tr)

            store.model, store.scaler = fit_v2_model(
                X_tr,
                y_tr
            )

            store.feature_cols = FEATURES_COMBINED_V2

            log.info("Prediction model loaded (V2)")

        except Exception as e:

            log.warning(f"V2 failed ({e}) — using Logistic Regression")

            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            FEATS = [
                "congestion_score",
                "net_dwt_flow",
                "weather_risk_score",
                "anomaly_flag",
                "queue_buildup_7d",
                "yoy_pct_change",
            ]

            for col in FEATS:

                if col not in train.columns:
                    train[col] = 0.0

                train[col] = pd.to_numeric(
                    train[col],
                    errors="coerce"
                ).fillna(0)

            for s in STATES:
                train[f"state_is_{s.lower()}"] = (
                    train["state"] == s
                ).astype(float)

            feat_cols = FEATS + [
                f"state_is_{s.lower()}"
                for s in STATES
            ]

            X_list = []
            y_list = []

            for port in PORT_NAMES:

                pf = (
                    train[train["port"] == port]
                    .sort_values("date")
                    .reset_index(drop=True)
                )

                if len(pf) < 2:
                    continue

                X_list.append(
                    pf[feat_cols].values[:-1]
                )

                y_list.append(
                    np.array([
                        STATE_MAP.get(s, 0)
                        for s in pf["state"].values[1:]
                    ])
                )

            X_tr = np.vstack(X_list)
            y_tr = np.concatenate(y_list)

            sc = StandardScaler()

            mdl = LogisticRegression(
                solver="lbfgs",
                max_iter=500,
                class_weight="balanced",
                random_state=42
            )

            mdl.fit(
                sc.fit_transform(X_tr),
                y_tr
            )

            store.model = mdl
            store.scaler = sc
            store.feature_cols = feat_cols

        # ---------------------------------------------------------------------
        # Load MDP policy
        # ---------------------------------------------------------------------

        audit_dir = os.path.join(ROOT, "results", "audit")

        policy_path = os.path.join(
            audit_dir,
            "mdp_policy_table.csv"
        )

        if os.path.exists(policy_path):

            pdf = pd.read_csv(policy_path)

            store.policy_map = {
                (row["state"], int(row["tier"])): row["optimal_action"]
                for _, row in pdf.iterrows()
            }

        else:

            for s in STATES:

                store.policy_map[(s, 0)] = "Strong"
                store.policy_map[(s, 1)] = "Escrow"
                store.policy_map[(s, 2)] = (
                    "Stale" if s != "S3" else "CRDT"
                )
                store.policy_map[(s, 3)] = (
                    "Stale" if s != "S3" else "CRDT"
                )

        # ---------------------------------------------------------------------
        # Cost matrix
        # ---------------------------------------------------------------------

        cost_path = os.path.join(
            audit_dir,
            "mdp_cost_matrix.csv"
        )

        if os.path.exists(cost_path):

            store.cost_df = pd.read_csv(
                cost_path,
                index_col=0
            )

        else:

            store.cost_df = pd.DataFrame(
                {
                    "Strong": [1.0, 1.3, 1.7, 3.8, 1.1],
                    "Escrow": [0.6, 0.8, 1.0, 2.3, 0.7],
                    "CRDT":   [0.2, 0.3, 0.3, 0.8, 0.2],
                    "Stale":  [0.1, 0.1, 0.2, 0.4, 0.1],
                    "Block":  [2.0, 2.6, 3.4, 7.6, 2.2],
                },
                index=STATES
            )

        store.loaded = True

        log.info("All models loaded successfully")

        return True

    except Exception as e:

        log.error(f"Model load failed: {e}")

        return False


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Hawkes Intensity
# ─────────────────────────────────────────────────────────────────────────────

def compute_lambda(
    port: str,
    date: pd.Timestamp,
    congestion_score: float = 0.0,
    weather_risk_score: float = 0.0,
    queue_buildup_7d: float = 0.0,
    anomaly_flag: int = 0,
    storm_alert: int = 0,
    heavy_rain_alert: int = 0,
    yoy_pct_change: float = 0.0,
) -> float:

    store = ModelStore.get()

    if not store.loaded:
        return 0.08

    # -------------------------------------------------------------------------
    # Historical Hawkes Base
    # -------------------------------------------------------------------------

    t = float((date - store.t0).days)

    mu_i = float(
        store.mu_s.get(port, 0.08)
    )

    lam_hist = mu_i

    for src in PORT_NAMES:

        alpha = (
            float(store.ccf_df.loc[port, src])
            if port in store.ccf_df.index
            and src in store.ccf_df.columns
            else 0.0
        )

        if alpha <= 0:
            continue

        for tk in store.onset_cache.get(src, []):

            if tk < t:

                lam_hist += (
                    alpha *
                    np.exp(-DECAY * (t - tk))
                )

    # -------------------------------------------------------------------------
    # Dynamic Feature Amplification
    # -------------------------------------------------------------------------

    congestion_term = (
        0.25 * np.clip(congestion_score, 0, 10)
    )

    weather_term = (
        0.20 * np.clip(weather_risk_score, 0, 10)
    )

    queue_term = (
        0.15 * np.log1p(
            max(queue_buildup_7d, 0)
        )
    )

    anomaly_term = (
        0.30 if anomaly_flag else 0.0
    )

    storm_term = (
        0.35 if storm_alert else 0.0
    )

    rain_term = (
        0.15 if heavy_rain_alert else 0.0
    )

    economic_term = min(
        abs(yoy_pct_change) / 100.0,
        0.25
    )

    # -------------------------------------------------------------------------
    # Combined Dynamic Boost
    # -------------------------------------------------------------------------

    dynamic_boost = (
        congestion_term
        + weather_term
        + queue_term
        + anomaly_term
        + storm_term
        + rain_term
        + economic_term
    )

    cascade_multiplier = (
        1.0 + dynamic_boost
    )

    lam_dynamic = (
        lam_hist * cascade_multiplier
    )

    # Safety Clamp

    lam_dynamic = float(
        np.clip(lam_dynamic, 0.01, 5.0)
    )

    return round(lam_dynamic, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Main Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    port: str,
    current_state: str,
    congestion_score: float,
    net_dwt_flow: float = 0.0,
    container_ratio: float = 0.5,
    queue_buildup_7d: float = 0.0,
    yoy_pct_change: float = 0.0,
    anomaly_flag: int = 0,
    weather_risk_score: float = 0.0,
    storm_alert: int = 0,
    heavy_rain_alert: int = 0,
    date: Optional[str] = None,
) -> dict:

    store = ModelStore.get()

    if not store.loaded:
        load_models()

    ts = (
        pd.Timestamp(date)
        if date
        else pd.Timestamp.now()
    )

    # -------------------------------------------------------------------------
    # Dynamic Hawkes Lambda
    # -------------------------------------------------------------------------

    lam = compute_lambda(
        port=port,
        date=ts,
        congestion_score=congestion_score,
        weather_risk_score=weather_risk_score,
        queue_buildup_7d=queue_buildup_7d,
        anomaly_flag=anomaly_flag,
        storm_alert=storm_alert,
        heavy_rain_alert=heavy_rain_alert,
        yoy_pct_change=yoy_pct_change,
    )

    lam_prev = compute_lambda(
        port=port,
        date=ts - pd.Timedelta(days=1),
        congestion_score=congestion_score * 0.95,
        weather_risk_score=weather_risk_score * 0.95,
        queue_buildup_7d=queue_buildup_7d * 0.95,
        anomaly_flag=anomaly_flag,
        storm_alert=storm_alert,
        heavy_rain_alert=heavy_rain_alert,
        yoy_pct_change=yoy_pct_change,
    )

    delta_lam = lam - lam_prev

    # -------------------------------------------------------------------------
    # One-hot states
    # -------------------------------------------------------------------------

    oh = {
        f"state_is_{s.lower()}": int(current_state == s)
        for s in STATES
    }

    # -------------------------------------------------------------------------
    # Feature Vector
    # -------------------------------------------------------------------------

    feature_vec = {

        "congestion_score": congestion_score,
        "net_dwt_flow": net_dwt_flow,
        "container_ratio": container_ratio,
        "queue_buildup_7d": queue_buildup_7d,
        "yoy_pct_change": yoy_pct_change,
        "anomaly_flag": anomaly_flag,
        "weather_risk_score": weather_risk_score,
        "storm_alert": storm_alert,
        "heavy_rain_alert": heavy_rain_alert,

        "hawkes_lambda": lam,
        "hawkes_delta_lambda": delta_lam,

        **oh,

        "congestion_score_7d": congestion_score,
        "congestion_score_14d": congestion_score * 0.95,
        "congestion_accel": delta_lam,

        "queue_buildup_14d": queue_buildup_7d * 0.9,
        "queue_buildup_30d": queue_buildup_7d * 0.8,
        "queue_accel": delta_lam,

        "days_in_state": 3.0,
        "state_changed_last_day": 0,

        "state_is_s0_prev": int(current_state == "S0"),
        "state_is_s3_prev": int(current_state == "S3"),

        "hawkes_cascade_velocity": delta_lam,
        "hawkes_cascade_acceleration": delta_lam * 0.5,

        "hawkes_intensity_pct": min(lam / 5.0, 1.0),

        "hawkes_port_baseline_dev": lam - 0.08,

        "hawkes_cross_port_strength": lam,

        "cascade_x_congestion": (
            lam * congestion_score
        ),

        "cascade_x_weather": (
            lam * weather_risk_score
        ),

        "anomaly_x_rising_lambda": (
            anomaly_flag * int(delta_lam > 0)
        ),

        "storm_x_cascade": (
            storm_alert * lam
        ),

        "combined_risk_score": (
            lam + weather_risk_score
        ) / 2,

        "economic_shock_flag": (
            int(abs(yoy_pct_change) > 20)
        ),
    }

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    X = np.array([
        [
            feature_vec.get(c, 0.0)
            for c in store.feature_cols
        ]
    ])

    X_sc = store.scaler.transform(X)

    probs = store.model.predict_proba(X_sc)[0]

    pred_idx = int(np.argmax(probs))

    pred_st = (
        STATES[pred_idx]
        if pred_idx < len(STATES)
        else "S0"
    )

    # -------------------------------------------------------------------------
    # S3 Risk
    # -------------------------------------------------------------------------

    s3_idx = STATE_MAP["S3"]

    p_s3 = (
        float(probs[s3_idx])
        if len(probs) > s3_idx
        else 0.0
    )

    lam_norm = float(
        np.clip(
            (lam - 0.08) / 0.35,
            0,
            1
        )
    )

    risk_blend = min(
        lam / 2.0,
        0.5
    )

    p_s3_blended = round(
        ((1 - risk_blend) * p_s3)
        + (risk_blend * lam_norm),
        4
    )

    alert = (
        p_s3_blended > ALERT_THRESH
    )

    # -------------------------------------------------------------------------
    # MDP Actions
    # -------------------------------------------------------------------------

    actions = {}

    for tier in range(4):

        action = store.policy_map.get(
            (current_state, tier),
            "Strong"
        )

        cost = (
            float(
                store.cost_df.loc[current_state, action]
            )
            if current_state in store.cost_df.index
            and action in store.cost_df.columns
            else 1.0
        )

        actions[tier] = {
            "action": action,
            "cost": round(cost, 4)
        }

    # -------------------------------------------------------------------------
    # State probabilities
    # -------------------------------------------------------------------------

    state_probs = {
        STATES[i]: round(float(p), 4)
        for i, p in enumerate(probs)
        if i < len(STATES)
    }

    # -------------------------------------------------------------------------
    # Return
    # -------------------------------------------------------------------------

    return {

        "port": port,

        "current_state": current_state,

        "predicted_state": pred_st,

        "p_s3": p_s3_blended,

        "hawkes_lambda": round(lam, 4),

        "hawkes_delta": round(delta_lam, 4),

        "alert": alert,

        "state_probs": state_probs,

        "actions": actions,

        "confidence": round(
            float(probs.max()),
            4
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Port Status Summary
# ─────────────────────────────────────────────────────────────────────────────

def get_all_port_status(
    date: Optional[str] = None
) -> list:

    store = ModelStore.get()

    if not store.loaded:
        load_models()

    ts = (
        pd.Timestamp(date)
        if date
        else pd.Timestamp("2024-06-01")
    )

    combined = store.combined_df

    results = []

    for port in PORT_NAMES:

        port_df = (
            combined[combined["port"] == port]
            .sort_values("date")
        )

        if len(port_df[port_df["date"] <= ts]):

            row = (
                port_df[port_df["date"] <= ts]
                .iloc[-1]
            )

        else:

            row = port_df.iloc[0]

        pred = predict(

            port=port,

            current_state=row.get("state", "S0"),

            congestion_score=float(
                row.get("congestion_score", 1.0)
            ),

            net_dwt_flow=float(
                row.get("net_dwt_flow", 0)
            ),

            container_ratio=float(
                row.get("container_ratio", 0.5)
            ),

            queue_buildup_7d=float(
                row.get("queue_buildup_7d", 0)
            ),

            yoy_pct_change=float(
                row.get("yoy_pct_change", 0)
            ),

            anomaly_flag=int(
                row.get("anomaly_flag", 0)
            ),

            weather_risk_score=float(
                row.get("weather_risk_score", 0)
            ),

            storm_alert=int(
                row.get("storm_alert", 0)
            ),

            heavy_rain_alert=int(
                row.get("heavy_rain_alert", 0)
            ),

            date=str(
                row["date"].date()
            ),
        )

        results.append(pred)

    return results