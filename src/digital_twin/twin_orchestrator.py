import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field, asdict
from typing import Optional

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.constants import (
    PROCESSED_DIR, UNIFIED_DIR, STATES, STATE_MAP, PORTS,
)
from src.utils.logger import get_logger

log = get_logger("twin_orchestrator")

RESULTS_DIR   = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")),
    "results", "digital_twin"
)
COMBINED_DIR  = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")),
    "results", "combined"
)
HAWKES_DIR    = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")),
    "results", "hawkes_standalone"
)
MARKOV_DIR    = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")),
    "results", "markov_standalone"
)
AUDIT_DIR     = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")),
    "results", "audit"
)

PORT_NAMES    = list(PORTS.keys())
DECAY         = 0.5
ALERT_THRESH  = 0.30   # P(S3) threshold for early warning
TIERS         = [0, 1, 2, 3]

@dataclass
class TwinObservation:
    """One day's observation for a single port."""
    date:               str
    port:               str
    current_state:      str
    congestion_score:   float = 0.0
    net_dwt_flow:       float = 0.0
    container_ratio:    float = 0.0
    queue_buildup_7d:   float = 0.0
    yoy_pct_change:     float = 0.0
    anomaly_flag:       int   = 0
    weather_risk_score: float = 0.0
    storm_alert:        int   = 0
    heavy_rain_alert:   int   = 0


@dataclass
class TwinPrediction:
    """One day's prediction output for a single port."""
    date:              str
    port:              str
    current_state:     str
    predicted_state:   str
    p_s3:              float
    hawkes_lambda:     float
    hawkes_alert:      bool
    actions:           dict = field(default_factory=dict)
    lead_days:         int  = 0
    confidence:        float = 0.0

class TwinMeshModel:
    """
    Loads fitted Day 5 model (V2 if available, V1 fallback).
    Provides predict() and compute_hawkes() methods.
    """

    def __init__(self):
        self.model         = None
        self.scaler        = None
        self.policy_map    = {}
        self.cost_df       = None
        self.ccf_df        = None
        self.mu_s          = None
        self.onset_cache   = {}   # port -> list of onset t_numeric
        self.t0            = None
        self._loaded       = False

    def load(self, combined_df: pd.DataFrame) -> bool:
        """
        Loads all fitted components.
        Returns True if successful.
        """
        log.info("  Loading TwinMesh components...")

        # ── Hawkes CCF matrix ─────────────────────────────────────────────────
        ccf_path = os.path.join(HAWKES_DIR, "hawkes_excitation_matrices", "ccf_7x7.csv")
        mu_path  = os.path.join(HAWKES_DIR, "hawkes_excitation_matrices", "port_level_mu.csv")

        self.ccf_df = pd.read_csv(ccf_path, index_col=0) \
                      if os.path.exists(ccf_path) \
                      else pd.DataFrame(np.eye(len(PORT_NAMES)) * 0.05,
                                        index=PORT_NAMES, columns=PORT_NAMES)
        self.mu_s   = pd.read_csv(mu_path, index_col=0).squeeze() \
                      if os.path.exists(mu_path) \
                      else pd.Series(0.08, index=PORT_NAMES)
        log.info("    CCF matrix loaded")

        self.t0 = combined_df["date"].min()
        combined_df = combined_df.copy()
        combined_df["t_numeric"] = (combined_df["date"] - self.t0).dt.days.astype(float)

        for port in PORT_NAMES:
            pdf   = combined_df[combined_df["port"] == port].sort_values("date").reset_index(drop=True)
            is_s3 = (pdf["state"] == "S3")
            prev  = (pdf["state"].shift(1) != "S3")
            mask  = is_s3 & prev
            if len(mask):
                mask.iloc[0] = is_s3.iloc[0]
            self.onset_cache[port] = pdf[mask]["t_numeric"].values.tolist()
        log.info(f"    S3 onset cache built ({sum(len(v) for v in self.onset_cache.values())} events)")

        try:
            from src.models.combined.twinmesh_core_v2 import (
                prepare_sequences_v2, fit_v2_model,
                compute_hawkes_intensity, FEATURES_COMBINED_V2
            )
            train = pd.read_csv(
                os.path.join(UNIFIED_DIR, "twinmesh_training.csv"),
                parse_dates=["date"]
            )
            combined_with_lam = compute_hawkes_intensity(combined_df)
            lam_df = combined_with_lam[["date", "port", "hawkes_lambda", "hawkes_delta_lambda"]]
            train  = train.merge(lam_df, on=["date", "port"], how="left")
            train[["hawkes_lambda", "hawkes_delta_lambda"]] = \
                train[["hawkes_lambda", "hawkes_delta_lambda"]].fillna(
                    {"hawkes_lambda": 0.08, "hawkes_delta_lambda": 0.0}
                )

            X_train, y_train = prepare_sequences_v2(train)
            self.model, self.scaler = fit_v2_model(X_train, y_train)
            self.feature_cols = FEATURES_COMBINED_V2
            log.info(f"    Predictive model loaded (V2, {len(FEATURES_COMBINED_V2)} features)")

        except Exception as e:
            log.warning(f"    V2 failed ({e}) — loading V1 LogReg")
            try:
                from src.models.combined.twinmesh_core import (
                    prepare_combined_sequences, fit_combined_model,
                    compute_hawkes_intensity as ci_v1, FEATURES_COMBINED
                )
                train = pd.read_csv(
                    os.path.join(UNIFIED_DIR, "twinmesh_training.csv"),
                    parse_dates=["date"]
                )
                combined_lam = ci_v1(combined_df)
                lam_df2 = combined_lam[["date", "port", "hawkes_lambda", "hawkes_delta_lambda"]]
                train   = train.merge(lam_df2, on=["date", "port"], how="left")
                train[["hawkes_lambda", "hawkes_delta_lambda"]] = \
                    train[["hawkes_lambda", "hawkes_delta_lambda"]].fillna(
                        {"hawkes_lambda": 0.08, "hawkes_delta_lambda": 0.0}
                    )

                for s in STATES:
                    col = f"state_is_{s.lower()}"
                    train[col] = (train["state"] == s).astype(float)

                X_train, y_train = prepare_combined_sequences(train, FEATURES_COMBINED)
                self.model, self.scaler = fit_combined_model(X_train, y_train)
                self.feature_cols = FEATURES_COMBINED
                log.info("    Predictive model loaded (V1 fallback)")

            except Exception as e2:
                log.error(f"    Both V1 and V2 failed: {e2}")
                return False

        policy_path = os.path.join(AUDIT_DIR, "mdp_policy_table.csv")
        if os.path.exists(policy_path):
            policy_df = pd.read_csv(policy_path)
            self.policy_map = {
                (row["state"], int(row["tier"])): row["optimal_action"]
                for _, row in policy_df.iterrows()
            }
            log.info(f"    MDP policy loaded ({len(self.policy_map)} entries)")
        else:

            self.policy_map = {
                (s, 0): "Strong" for s in STATES
            }
            log.warning("    MDP policy not found — using default (always Strong for Tier-0)")

        cost_path = os.path.join(AUDIT_DIR, "mdp_cost_matrix.csv")
        if os.path.exists(cost_path):
            self.cost_df = pd.read_csv(cost_path, index_col=0)
        else:
            self.cost_df = pd.DataFrame({
                "Strong": [1.0, 1.3, 1.7, 3.8, 1.1],
                "Escrow": [0.6, 0.8, 1.0, 2.3, 0.7],
                "CRDT":   [0.2, 0.3, 0.3, 0.8, 0.2],
                "Stale":  [0.1, 0.1, 0.2, 0.4, 0.1],
                "Block":  [2.0, 2.6, 3.4, 7.6, 2.2],
            }, index=STATES)

        self._loaded = True
        log.info("  All components loaded successfully")
        return True

    def hawkes_intensity(self, port: str, t: float) -> float:
        """Computes lambda(t) for a port at time t."""
        mu_i = float(self.mu_s.get(port, 0.08))
        lam  = mu_i
        for src in PORT_NAMES:
            alpha = float(self.ccf_df.loc[port, src]) \
                    if port in self.ccf_df.index and src in self.ccf_df.columns \
                    else 0.0
            if alpha <= 0:
                continue
            for tk in self.onset_cache.get(src, []):
                if tk < t:
                    lam += alpha * np.exp(-DECAY * (t - tk))
        return lam

    def predict_one(
        self,
        obs: TwinObservation,
        history_df: pd.DataFrame,
    ) -> TwinPrediction:
        """
        Predicts next state + fires alert for one observation.
        history_df: all rows up to and including obs.date for feature engineering.
        """
        if not self._loaded:
            raise RuntimeError("Call load() first")

        
        t  = float((pd.Timestamp(obs.date) - self.t0).days)
        lam = self.hawkes_intensity(obs.port, t)
        delta_lam = lam - self.hawkes_intensity(obs.port, t - 1)

        
        port_hist = history_df[history_df["port"] == obs.port].sort_values("date")
        cong_7d   = port_hist["congestion_score"].tail(7).mean()
        cong_14d  = port_hist["congestion_score"].tail(14).mean()
        q_7d      = port_hist["queue_buildup_7d"].tail(7).mean() \
                    if "queue_buildup_7d" in port_hist.columns else 0.0
        q_14d     = port_hist["queue_buildup_7d"].tail(14).mean() \
                    if "queue_buildup_7d" in port_hist.columns else 0.0

        
        state_seq  = port_hist["state"].tolist()
        days_in_s  = 1
        for prev_s in reversed(state_seq[:-1]):
            if prev_s == obs.current_state:
                days_in_s += 1
            else:
                break

       
        oh = {f"state_is_{s.lower()}": int(obs.current_state == s) for s in STATES}

        feature_vec = {
            "congestion_score":           obs.congestion_score,
            "net_dwt_flow":               obs.net_dwt_flow,
            "container_ratio":            obs.container_ratio,
            "queue_buildup_7d":           obs.queue_buildup_7d,
            "yoy_pct_change":             obs.yoy_pct_change,
            "anomaly_flag":               obs.anomaly_flag,
            "weather_risk_score":         obs.weather_risk_score,
            "storm_alert":                obs.storm_alert,
            "heavy_rain_alert":           obs.heavy_rain_alert,
            "hawkes_lambda":              lam,
            "hawkes_delta_lambda":        delta_lam,
            **oh,
            "congestion_score_7d":        cong_7d,
            "congestion_score_14d":       cong_14d,
            "congestion_accel":           (cong_7d - cong_14d) / (cong_14d + 0.001),
            "queue_buildup_14d":          q_14d,
            "queue_buildup_30d":          q_14d,
            "queue_accel":                (q_7d - q_14d) / (q_14d + 0.001),
            "days_in_state":              days_in_s,
            "state_changed_last_day":     int(len(state_seq) > 1 and
                                              state_seq[-1] != state_seq[-2]),
            "state_is_s0_prev":           int(len(state_seq) > 1 and
                                              state_seq[-2] == "S0"),
            "state_is_s3_prev":           int(len(state_seq) > 1 and
                                              state_seq[-2] == "S3"),
            "hawkes_cascade_velocity":    delta_lam,
            "hawkes_cascade_acceleration":0.0,
            "hawkes_intensity_pct":       0.5,
            "hawkes_port_baseline_dev":   0.0,
            "hawkes_cross_port_strength": lam,
            "cascade_x_congestion":       lam * obs.congestion_score,
            "cascade_x_weather":          lam * obs.weather_risk_score,
            "anomaly_x_rising_lambda":    obs.anomaly_flag * int(delta_lam > 0),
            "storm_x_cascade":            obs.storm_alert * lam,
            "combined_risk_score":        (lam + obs.weather_risk_score) / 2,
            "economic_shock_flag":        int(abs(obs.yoy_pct_change) > 20),
        }

        X = np.array([[feature_vec.get(c, 0.0) for c in self.feature_cols]])
        X_scaled = self.scaler.transform(X)
        probs    = self.model.predict_proba(X_scaled)[0]

        s3_idx   = STATE_MAP["S3"]
        p_s3     = float(probs[s3_idx]) if len(probs) > s3_idx else 0.0
        pred_idx = int(np.argmax(probs))
        pred_s   = STATES[pred_idx] if pred_idx < len(STATES) else "S0"

        p_s3_hawkes = float(np.clip((lam - 0.08) / 0.35, 0, 1))
        p_s3_combined = 0.7 * p_s3 + 0.3 * p_s3_hawkes

        alert = p_s3_combined > ALERT_THRESH

        actions = {}
        for tier in TIERS:
            action = self.policy_map.get((obs.current_state, tier), "Strong")
            cost   = float(self.cost_df.loc[obs.current_state, action]) \
                     if obs.current_state in self.cost_df.index \
                     and action in self.cost_df.columns else 1.0
            actions[tier] = {"action": action, "cost": round(cost, 4)}

        return TwinPrediction(
            date            = obs.date,
            port            = obs.port,
            current_state   = obs.current_state,
            predicted_state = pred_s,
            p_s3            = round(p_s3_combined, 4),
            hawkes_lambda   = round(lam, 4),
            hawkes_alert    = alert,
            actions         = actions,
            confidence      = round(float(probs.max()), 4),
        )

class TwinReplay:
    """
    Runs the digital twin in replay mode over the test set.
    Simulates day-by-day: feed observation → get prediction → log decision.
    """

    def __init__(self, model: TwinMeshModel):
        self.model   = model
        self.log_rows = []
        self.alerts  = []

    def run(self, test_df: pd.DataFrame, combined_df: pd.DataFrame) -> pd.DataFrame:
        """
        Replays all test days chronologically per port.
        Returns log DataFrame.
        """
        log.info(f"  Replaying {len(test_df):,} test rows...")
        t_start = time.time()

        all_dates = sorted(test_df["date"].unique())
        n_alerts  = 0
        n_correct = 0
        n_total   = 0

        for date in all_dates:
            day_df = test_df[test_df["date"] == date]

            for _, row in day_df.iterrows():
                port = row["port"]

                history = combined_df[
                    (combined_df["port"] == port) &
                    (combined_df["date"] <= date)
                ].sort_values("date")

                obs = TwinObservation(
                    date               = str(date.date()),
                    port               = port,
                    current_state      = row.get("state", "S0"),
                    congestion_score   = float(row.get("congestion_score", 0)),
                    net_dwt_flow       = float(row.get("net_dwt_flow", 0)),
                    container_ratio    = float(row.get("container_ratio", 0)),
                    queue_buildup_7d   = float(row.get("queue_buildup_7d", 0)),
                    yoy_pct_change     = float(row.get("yoy_pct_change", 0)),
                    anomaly_flag       = int(row.get("anomaly_flag", 0)),
                    weather_risk_score = float(row.get("weather_risk_score", 0)),
                    storm_alert        = int(row.get("storm_alert", 0)),
                    heavy_rain_alert   = int(row.get("heavy_rain_alert", 0)),
                )

                pred = self.model.predict_one(obs, history)

                entry = {
                    "date":            pred.date,
                    "port":            pred.port,
                    "current_state":   pred.current_state,
                    "predicted_state": pred.predicted_state,
                    "p_s3":            pred.p_s3,
                    "hawkes_lambda":   pred.hawkes_lambda,
                    "hawkes_alert":    int(pred.hawkes_alert),
                    "confidence":      pred.confidence,
                    "tier0_action":    pred.actions.get(0, {}).get("action", "Strong"),
                    "tier1_action":    pred.actions.get(1, {}).get("action", "Escrow"),
                    "tier2_action":    pred.actions.get(2, {}).get("action", "Stale"),
                    "tier3_action":    pred.actions.get(3, {}).get("action", "Stale"),
                    "tier0_cost":      pred.actions.get(0, {}).get("cost", 1.0),
                }
                self.log_rows.append(entry)

                if pred.hawkes_alert:
                    n_alerts += 1
                    self.alerts.append(entry)

                n_total += 1
                if pred.predicted_state == pred.current_state:
                    n_correct += 1

        elapsed = time.time() - t_start
        log.info(
            f"  Replay complete: {n_total:,} steps in {elapsed:.1f}s\n"
            f"    Alerts fired:  {n_alerts:,} ({n_alerts/n_total:.1%})\n"
            f"    State match:   {n_correct/n_total:.1%}"
        )
        return pd.DataFrame(self.log_rows)

def plot_twin_dashboard(
    log_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    4-panel dashboard showing the Digital Twin in action.
    Panel 1: P(S3) for Rotterdam over time
    Panel 2: Hawkes lambda for Singapore over time
    Panel 3: Action distribution across ports
    Panel 4: Alert timeline for all ports
    Draft of Figure 9.
    """
    log_df["date"] = pd.to_datetime(log_df["date"])

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    COLORS = {
        "S0": "#2ecc71", "S1": "#f1c40f",
        "S2": "#e67e22", "S3": "#e74c3c", "S4": "#3498db"
    }

    ax = axes[0]
    port_log = log_df[log_df["port"] == "Rotterdam"].sort_values("date")
    actual   = test_df[test_df["port"] == "Rotterdam"].sort_values("date")

    if len(port_log):
        ax.plot(port_log["date"], port_log["p_s3"],
                color="#3498db", lw=1.8, label="TwinMesh P(S3)", alpha=0.9)
        ax.axhline(ALERT_THRESH, color="#e74c3c", lw=1.2, ls="--",
                   label=f"Alert threshold ({ALERT_THRESH})")
        ax.fill_between(port_log["date"], port_log["p_s3"],
                        alpha=0.12, color="#3498db")

        if len(actual):
            is_s3 = actual["state"] == "S3"
            for i, (d, s) in enumerate(zip(actual["date"], is_s3)):
                if s:
                    ax.axvline(d, color="#e74c3c", alpha=0.08, lw=0.5)

    ax.set_title("Rotterdam — P(S3) Over Time", fontweight="bold", fontsize=10)
    ax.set_ylabel("P(S3)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sg_log = log_df[log_df["port"] == "Singapore"].sort_values("date")
    if len(sg_log):
        ax.plot(sg_log["date"], sg_log["hawkes_lambda"],
                color="#9b59b6", lw=1.8, label="λ(t) — Hawkes intensity")
        ax.fill_between(sg_log["date"], sg_log["hawkes_lambda"],
                        alpha=0.15, color="#9b59b6")
        alert_days = sg_log[sg_log["hawkes_alert"] == 1]
        if len(alert_days):
            ax.scatter(alert_days["date"], alert_days["hawkes_lambda"],
                       color="#e74c3c", s=25, zorder=5, label="Alert fired")

    ax.set_title("Singapore — Hawkes Intensity λ(t)", fontweight="bold", fontsize=10)
    ax.set_ylabel("λ(t)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    action_counts = log_df["tier0_action"].value_counts()
    action_colors = {
        "Strong": "#e74c3c", "Escrow": "#e67e22",
        "CRDT":   "#2ecc71", "Stale":  "#95a5a6", "Block": "#2c3e50"
    }
    bars = ax.bar(
        action_counts.index,
        action_counts.values,
        color=[action_colors.get(a, "#3498db") for a in action_counts.index],
        alpha=0.85, edgecolor="white"
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"{h:,.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Tier-0 Action Distribution (Test Set)", fontweight="bold", fontsize=10)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis="y")

    ax    = axes[3]
    ports = PORT_NAMES
    y_pos = {p: i for i, p in enumerate(ports)}

    for _, row in log_df[log_df["hawkes_alert"] == 1].iterrows():
        if row["port"] in y_pos:
            ax.scatter(row["date"], y_pos[row["port"]],
                       color="#e74c3c", s=12, alpha=0.6, marker="|")

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()), fontsize=9)
    ax.set_title("Alert Timeline — All Ports", fontweight="bold", fontsize=10)
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        "TwinMesh Digital Twin — Live Simulation Dashboard\n"
        "(Draft Figure 9)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "twin_dashboard.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Dashboard saved -> {out_png}")


def plot_per_port_replay(log_df: pd.DataFrame, port: str) -> None:
    """Per-port replay plot: state timeline + P(S3) + alerts."""
    port_df = log_df[log_df["port"] == port].sort_values("date").copy()
    port_df["date"] = pd.to_datetime(port_df["date"])
    if len(port_df) == 0:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1]})

    STATE_COLORS = {
        "S0": "#2ecc71", "S1": "#f1c40f",
        "S2": "#e67e22", "S3": "#e74c3c", "S4": "#3498db"
    }

    for _, row in port_df.iterrows():
        c = STATE_COLORS.get(row["current_state"], "#95a5a6")
        axes[0].bar(row["date"], 1, color=c, alpha=0.8, width=0.9)
    axes[0].set_ylabel("Current State")
    axes[0].set_yticks([])
    axes[0].set_title(f"{port} — Digital Twin Replay", fontweight="bold")


    axes[1].plot(port_df["date"], port_df["p_s3"],
                 color="#3498db", lw=1.5, label="P(S3)")
    axes[1].axhline(ALERT_THRESH, color="#e74c3c", lw=1.0, ls="--",
                    label="Alert threshold")
    axes[1].fill_between(port_df["date"], port_df["p_s3"], alpha=0.1, color="#3498db")
    axes[1].set_ylabel("P(S3)")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)

  
    alerts = port_df[port_df["hawkes_alert"] == 1]
    axes[2].bar(alerts["date"], 1, color="#e74c3c", alpha=0.8, width=0.9)
    axes[2].set_ylabel("Alert")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["", "ALERT"])
    axes[2].set_xlabel("Date")

    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, f"twin_replay_{port.lower().replace(' ', '_')}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Per-port replay saved -> {out_png}")

def demo_live_prediction(model: TwinMeshModel, combined_df: pd.DataFrame) -> None:
    """
    Demonstrates single-shot live prediction for a hypothetical observation.
    Prints what the Digital Twin would output in production.
    """
    log.info("\n" + "=" * 60)
    log.info("LIVE PREDICTION DEMO")
    log.info("=" * 60)

    obs = TwinObservation(
        date               = "2024-12-15",
        port               = "Rotterdam",
        current_state      = "S2",   
        congestion_score   = 2.8,    
        net_dwt_flow       = -50000, 
        container_ratio    = 0.45,
        queue_buildup_7d   = 3.2,    
        yoy_pct_change     = -8.5,   
        anomaly_flag       = 1,      
        weather_risk_score = 0.6,    
        storm_alert        = 0,
        heavy_rain_alert   = 1,
    )

    hist = combined_df[
        (combined_df["port"] == "Rotterdam") &
        (combined_df["date"] < pd.Timestamp("2024-12-15"))
    ].sort_values("date")

    pred = model.predict_one(obs, hist)

    log.info(f"\n  INPUT:")
    log.info(f"    Port:              {obs.port}")
    log.info(f"    Date:              {obs.date}")
    log.info(f"    Current state:     {obs.current_state} (Congested)")
    log.info(f"    Congestion score:  {obs.congestion_score}")
    log.info(f"    Anomaly flag:      {obs.anomaly_flag}")
    log.info(f"    Weather risk:      {obs.weather_risk_score}")

    log.info(f"\n  OUTPUT:")
    log.info(f"    Predicted state:   {pred.predicted_state}")
    log.info(f"    P(S3):             {pred.p_s3:.4f}  {'⚠ ALERT FIRED' if pred.hawkes_alert else '(below threshold)'}")
    log.info(f"    Hawkes λ(t):       {pred.hawkes_lambda:.4f}")
    log.info(f"    Confidence:        {pred.confidence:.4f}")
    log.info(f"\n  MDP ACTIONS:")
    for tier, rec in pred.actions.items():
        log.info(f"    Tier-{tier}: {rec['action']:<10} (cost={rec['cost']:.2f})")

    if pred.hawkes_alert:
        log.info(
            f"\n  PAPER SENTENCE (Section 6, live prediction):\n"
            f"  \"Given Rotterdam's current S2 (Congested) state and\n"
            f"   rising Hawkes intensity (λ={pred.hawkes_lambda:.3f}),\n"
            f"   TwinMesh predicts transition to {pred.predicted_state} with\n"
            f"   P(S3)={pred.p_s3:.3f}, triggering an early warning alert.\n"
            f"   Tier-0 operations are assigned {pred.actions[0]['action']} consistency;\n"
            f"   Tier-2 operations are downgraded to {pred.actions[2]['action']}.\""
        )

def run_digital_twin() -> pd.DataFrame:
    """
    Full Day 7 pipeline:
    1. Load all fitted components
    2. Replay on test set
    3. Build dashboard + per-port plots
    4. Demo live prediction
    Returns replay log DataFrame.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("DAY 7 — TwinMesh Digital Twin Orchestrator")
    log.info("=" * 60)

    combined = pd.read_csv(
        os.path.join(PROCESSED_DIR, "all_ports_combined.csv"),
        parse_dates=["date"]
    )
    test = pd.read_csv(
        os.path.join(UNIFIED_DIR, "twinmesh_test.csv"),
        parse_dates=["date"]
    )

    log.info(f"  Combined: {len(combined):,} rows | Test: {len(test):,} rows")

    twin_model = TwinMeshModel()
    if not twin_model.load(combined):
        log.error("  Failed to load TwinMesh components")
        return pd.DataFrame()

    log.info("\n  Running replay simulation...")
    replayer = TwinReplay(twin_model)
    log_df   = replayer.run(test, combined)

    log_df.to_csv(os.path.join(RESULTS_DIR, "twin_replay_log.csv"), index=False)

    alert_df = log_df[log_df["hawkes_alert"] == 1]
    alert_df.to_csv(os.path.join(RESULTS_DIR, "twin_alert_summary.csv"), index=False)
    log.info(
        f"\n  Replay log saved: {len(log_df):,} rows\n"
        f"  Alerts:           {len(alert_df):,} ({len(alert_df)/len(log_df):.1%})"
    )

    log.info("\n  Building dashboard...")
    plot_twin_dashboard(log_df, test)

    for port in ["Rotterdam", "Singapore", "Shanghai"]:
        plot_per_port_replay(log_df, port)

    demo_live_prediction(twin_model, combined)

    log.info("\n" + "=" * 60)
    log.info("DIGITAL TWIN SUMMARY")
    log.info("=" * 60)
    log.info(f"  Total decisions:   {len(log_df):,}")
    log.info(f"  Alerts fired:      {len(alert_df):,} ({len(alert_df)/len(log_df):.1%})")
    log.info(f"  Tier-0 always Strong: {(log_df['tier0_action'] == 'Strong').mean():.1%}")
    log.info(
        f"\n  PAPER SENTENCE (Section 6):\n"
        f"  \"The TwinMesh Digital Twin orchestrates a complete\n"
        f"   supply chain disruption prediction pipeline: IMF PortWatch\n"
        f"   and maritime weather data are ingested daily, Hawkes cascade\n"
        f"   intensity λ(t) is computed per port, and the fitted Markov\n"
        f"   observation model produces state predictions with associated\n"
        f"   MDP action recommendations, firing early warning alerts\n"
        f"   in {len(alert_df)/len(log_df):.1%} of test days.\""
    )

    return log_df


if __name__ == "__main__":
    run_digital_twin()