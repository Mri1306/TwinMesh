# =============================================================================
# TwinMesh App · replay_service.py
# Serves replay simulation data: actual vs predicted state over time
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

log = get_logger("replay_service")

PORT_NAMES = list(PORTS.keys())

TWIN_LOG   = os.path.join(ROOT, "results", "digital_twin", "twin_replay_log.csv")
COMBINED_F = os.path.join(PROCESSED_DIR, "all_ports_combined.csv")
TEST_FILE  = os.path.join(UNIFIED_DIR,   "twinmesh_test.csv")
HAWKES_DIR = os.path.join(ROOT, "results", "hawkes_standalone")


# ── Replay data loader ────────────────────────────────────────────────────────

def load_replay_log() -> pd.DataFrame:
    """
    Loads twin_replay_log.csv produced by Day 7.
    Falls back to generating a synthetic replay from test set if not found.
    """
    if os.path.exists(TWIN_LOG):
        df = pd.read_csv(TWIN_LOG, parse_dates=["date"])
        log.info(f"Replay log loaded: {len(df):,} rows")
        return df

    log.warning("twin_replay_log.csv not found — generating from test set")
    return _generate_replay_from_test()


def _generate_replay_from_test() -> pd.DataFrame:
    """
    Generates replay data from test set + CCF Hawkes intensity.
    Used when Day 7 has not been run yet.
    """
    if not os.path.exists(TEST_FILE):
        log.error("Test file not found")
        return pd.DataFrame()

    test     = pd.read_csv(TEST_FILE, parse_dates=["date"])
    combined = pd.read_csv(COMBINED_F, parse_dates=["date"])
    t0       = combined["date"].min()

    # Load CCF + mu
    ccf_path = os.path.join(HAWKES_DIR, "hawkes_excitation_matrices", "ccf_7x7.csv")
    mu_path  = os.path.join(HAWKES_DIR, "hawkes_excitation_matrices", "port_level_mu.csv")
    ccf_df = pd.read_csv(ccf_path, index_col=0) if os.path.exists(ccf_path) \
             else pd.DataFrame(np.eye(len(PORT_NAMES))*0.05,
                               index=PORT_NAMES, columns=PORT_NAMES)
    mu_s   = pd.read_csv(mu_path, index_col=0).squeeze() if os.path.exists(mu_path) \
             else pd.Series(0.08, index=PORT_NAMES)

    # Build onset cache
    combined["t_numeric"] = (combined["date"] - t0).dt.days.astype(float)
    onset_cache = {}
    for port in PORT_NAMES:
        pdf   = combined[combined["port"] == port].sort_values("date").reset_index(drop=True)
        is_s3 = (pdf["state"] == "S3")
        prev  = (pdf["state"].shift(1) != "S3")
        mask  = is_s3 & prev
        if len(mask): mask.iloc[0] = is_s3.iloc[0]
        onset_cache[port] = pdf[mask]["t_numeric"].values.tolist()

    rows = []
    for _, row in test.iterrows():
        port  = row["port"]
        t     = float((row["date"] - t0).days)
        mu_i  = float(mu_s.get(port, 0.08))
        lam   = mu_i
        for src in PORT_NAMES:
            alpha = float(ccf_df.loc[port, src]) \
                    if port in ccf_df.index and src in ccf_df.columns else 0.0
            if alpha <= 0: continue
            for tk in onset_cache.get(src, []):
                if tk < t:
                    lam += alpha * np.exp(-0.5 * (t - tk))

        lam_norm  = float(np.clip((lam - 0.08) / 0.35, 0, 1))
        s3_rate   = (test[test["port"]==port]["state"] == "S3").mean()
        p_s3      = round(0.6 * s3_rate + 0.4 * lam_norm, 4)
        alert     = int(p_s3 > 0.30)

        # Simple next-state prediction
        state    = row.get("state", "S0")
        pred_map = {"S0":"S0","S1":"S0","S2":"S3" if p_s3>0.4 else "S0",
                    "S3":"S3" if p_s3>0.45 else "S4","S4":"S0"}
        pred_st  = pred_map.get(state, "S0")

        rows.append({
            "date":            str(row["date"].date()),
            "port":            port,
            "current_state":   state,
            "predicted_state": pred_st,
            "p_s3":            p_s3,
            "hawkes_lambda":   round(lam, 4),
            "hawkes_alert":    alert,
            "confidence":      round(float(np.clip(0.4 + lam_norm*0.5, 0.4, 0.95)), 4),
            "tier0_action":    "Strong",
            "tier1_action":    "Escrow",
            "tier2_action":    "CRDT" if state=="S3" else "Stale",
            "tier3_action":    "CRDT" if state=="S3" else "Stale",
        })

    return pd.DataFrame(rows)


# ── Replay queries ────────────────────────────────────────────────────────────

def get_port_replay(
    port:       str,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> dict:
    """
    Returns replay data for one port as dict of lists (JSON-serialisable).
    """
    df = load_replay_log()
    if df.empty:
        return {}

    port_df = df[df["port"] == port].sort_values("date").copy()
    port_df["date"] = pd.to_datetime(port_df["date"])

    if start_date:
        port_df = port_df[port_df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        port_df = port_df[port_df["date"] <= pd.Timestamp(end_date)]

    if port_df.empty:
        return {}

    return {
        "dates":            port_df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "actual_state":     port_df["current_state"].tolist(),
        "predicted_state":  port_df["predicted_state"].tolist(),
        "p_s3":             port_df["p_s3"].round(4).tolist(),
        "hawkes_lambda":    port_df["hawkes_lambda"].round(4).tolist(),
        "alerts":           port_df["hawkes_alert"].astype(int).tolist(),
        "tier0_action":     port_df["tier0_action"].tolist() \
                            if "tier0_action" in port_df.columns else [],
        "n_rows":           len(port_df),
    }


def get_replay_summary() -> dict:
    """Overall stats from the replay log."""
    df = load_replay_log()
    if df.empty:
        return {}

    total       = len(df)
    n_alerts    = int(df["hawkes_alert"].sum()) if "hawkes_alert" in df.columns else 0
    n_s3        = int((df["current_state"] == "S3").sum())
    correct     = int((df["current_state"] == df["predicted_state"]).sum()) \
                  if "predicted_state" in df.columns else 0

    per_port    = {}
    for port in PORT_NAMES:
        pf = df[df["port"] == port]
        per_port[port] = {
            "n_rows":     len(pf),
            "n_alerts":   int(pf["hawkes_alert"].sum()) if "hawkes_alert" in pf.columns else 0,
            "s3_rate":    round(float((pf["current_state"]=="S3").mean()), 3),
            "mean_p_s3":  round(float(pf["p_s3"].mean()), 3),
            "mean_lambda":round(float(pf["hawkes_lambda"].mean()), 4),
        }

    return {
        "total_decisions":  total,
        "n_alerts":         n_alerts,
        "alert_rate":       round(n_alerts / total, 3) if total else 0,
        "n_s3_days":        n_s3,
        "accuracy":         round(correct / total, 3) if total else 0,
        "per_port":         per_port,
    }


def get_alert_feed(limit: int = 20) -> list:
    """Returns the most recent alerts across all ports."""
    df = load_replay_log()
    if df.empty or "hawkes_alert" not in df.columns:
        return []

    alerts = df[df["hawkes_alert"] == 1].sort_values("date", ascending=False).head(limit)
    return alerts[["date","port","current_state","p_s3","hawkes_lambda"]]\
           .rename(columns={"current_state":"state"})\
           .to_dict(orient="records")


def get_hawkes_series_all(
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> dict:
    """
    Returns Hawkes lambda(t) series for all ports.
    Used to render the multi-port intensity chart.
    """
    df = load_replay_log()
    if df.empty:
        return {}

    df["date"] = pd.to_datetime(df["date"])
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    result = {}
    dates_set = sorted(df["date"].dt.strftime("%Y-%m-%d").unique())
    result["dates"] = dates_set

    for port in PORT_NAMES:
        pf = df[df["port"] == port].sort_values("date")
        # Align to common date axis
        pf_indexed = pf.set_index(pf["date"].dt.strftime("%Y-%m-%d"))
        result[port] = [
            round(float(pf_indexed.loc[d,"hawkes_lambda"]), 4)
            if d in pf_indexed.index else None
            for d in dates_set
        ]

    return result