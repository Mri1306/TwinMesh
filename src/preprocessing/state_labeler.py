import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.constants import (
    PROCESSED_DIR, STATE_THRESHOLDS, STATES, STATE_MAP,
)
from src.utils.logger import get_logger

log = get_logger("state_labeler")

OUTPUT_FILE = os.path.join(PROCESSED_DIR, "state_labeled.csv")

EXPECTED_DIST = {"S0": 0.45, "S1": 0.25, "S2": 0.15, "S3": 0.10, "S4": 0.05}
TOLERANCE     = 0.15   

def assign_state_row(row: pd.Series) -> str:
    """
    Assigns S0–S3 based on congestion_score, storm_alert,
    disruption_flag, anomaly_flag, and weather_risk_score.
    S4 (Recovery) is assigned in a second pass that requires context.
    """
    c      = float(row.get("congestion_score",   0))
    storm  = int(row.get("storm_alert",          0))
    disr   = int(row.get("disruption_flag",      0))
    anomaly= int(row.get("anomaly_flag",         0))
    wx_risk= float(row.get("weather_risk_score", 0))

    t = STATE_THRESHOLDS

    if (c > t["S3_min_congestion"] or
            (storm == 1 and disr == 1) or
            anomaly == 1):
        return "S3"
    if c > t["S1_max_congestion"] or storm == 1:
        return "S2"
    if (c > t["S0_max_congestion"] or
            disr == 1 or
            wx_risk > t["weather_risk_s1"]):
        return "S1"
    return "S0"


def apply_recovery_pass(df: pd.DataFrame) -> pd.DataFrame:
    """
    Second-pass: mark rows as S4 (Recovery) when:
      - Previous row was S3
      - Current row is NOT S3
      - Current congestion_score < previous congestion_score (declining)
    """
    df = df.reset_index(drop=True)
    states = df["state"].tolist()
    congestion = df["congestion_score"].tolist()

    for i in range(1, len(df)):
        if (states[i - 1] == "S3" and
                states[i] != "S3" and
                congestion[i] < congestion[i - 1]):
            states[i] = "S4"

    df["state"] = states
    return df

def label_states(all_merged: dict) -> dict:
    """
    Applies state labeling to each port DataFrame.
    Returns updated dict with 'state' and 'state_numeric' columns.
    """
    for port_name, df in all_merged.items():
        df = df.sort_values("date").copy()

        for col in ["congestion_score", "storm_alert", "disruption_flag",
                    "anomaly_flag", "weather_risk_score"]:
            if col not in df.columns:
                log.warning(f"  {port_name}: column '{col}' missing — filling 0")
                df[col] = 0

        df["state"] = df.apply(assign_state_row, axis=1)

        df = apply_recovery_pass(df)

        df["state_numeric"] = df["state"].map(STATE_MAP).fillna(0).astype(int)

        dist = df["state"].value_counts(normalize=True).to_dict()
        log.info(f"  {port_name} state distribution:")
        for s in STATES:
            pct = dist.get(s, 0)
            exp = EXPECTED_DIST.get(s, 0)
            flag = "⚠️ " if abs(pct - exp) > TOLERANCE else "  "
            log.info(f"    {flag}{s}: {pct:.1%} (expected ~{exp:.0%})")

        if dist.get("S3", 0) < 0.02:
            log.warning(
                f"  {port_name}: S3 rate very low ({dist.get('S3', 0):.1%}) "
                f"— consider lowering S3_min_congestion threshold in constants.py"
            )
        if dist.get("S3", 0) > 0.25:
            log.warning(
                f"  {port_name}: S3 rate very high ({dist.get('S3', 0):.1%}) "
                f"— consider raising S3_min_congestion threshold in constants.py"
            )

        all_merged[port_name] = df

    return all_merged

def build_markov_transitions(all_merged: dict, start_date: str = "2023-01-01") -> pd.DataFrame:
    """
    Extracts all state-CHANGE transitions (prev_state ≠ curr_state)
    across all ports. Self-transitions (same state) are excluded.

    Output columns:
        port | from_state | to_state | timestamp |
        congestion_at_transition | weather_risk_at_transition |
        disruption_at_transition
    """
    transitions = []

    for port_name, df in all_merged.items():
        df_sorted = df[df["date"] >= start_date].sort_values("date").reset_index(drop=True)

        for i in range(1, len(df_sorted)):
            prev = df_sorted.at[i - 1, "state"]
            curr = df_sorted.at[i,     "state"]

            if prev != curr:   
                transitions.append({
                    "port":                       port_name,
                    "from_state":                 prev,
                    "to_state":                   curr,
                    "timestamp":                  str(df_sorted.at[i, "date"].date()),
                    "congestion_at_transition":   df_sorted.at[i, "congestion_score"],
                    "weather_risk_at_transition": df_sorted.at[i, "weather_risk_score"],
                    "disruption_at_transition":   int(df_sorted.at[i, "disruption_flag"]),
                    "net_dwt_at_transition":      df_sorted.at[i, "net_dwt_flow"],
                })

    transitions_df = pd.DataFrame(transitions)

    if len(transitions_df):
        log.info(
            f"  Markov transitions: {len(transitions_df):,} state changes | "
            f"S3 entries={len(transitions_df[transitions_df['to_state']=='S3'])} | "
            f"S4 entries={len(transitions_df[transitions_df['to_state']=='S4'])}"
        )
        matrix = pd.crosstab(
            transitions_df["from_state"],
            transitions_df["to_state"],
            normalize="index",
        ).round(3)
        log.info(f"\n  Transition matrix (normalized):\n{matrix.to_string()}")
    else:
        log.warning("  No state transitions found — check state labeling")

    return transitions_df

def run_state_labeler(all_merged: dict) -> tuple:
    """
    Returns (all_merged with state columns, markov_transitions DataFrame).
    Saves state_labeled.csv.
    """
    log.info("=" * 60)
    log.info("STEP 4 — State Labeler")
    log.info("=" * 60)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    all_merged = label_states(all_merged)
    parts = []
    for port_name, df in all_merged.items():
        parts.append(df[df["date"] >= "2023-01-01"])
    combined_2023 = pd.concat(parts, ignore_index=True)
    combined_2023.to_csv(OUTPUT_FILE, index=False)
    log.info(f"  Saved state_labeled.csv → {OUTPUT_FILE} ({len(combined_2023):,} rows)")

    markov_df = build_markov_transitions(all_merged)

    log.info("\n✅ STEP 4 COMPLETE — states labeled")
    return all_merged, markov_df


if __name__ == "__main__":
    from src.utils.constants import PER_PORT_DIR, PORTS
    all_merged = {}
    for port_name, prefix in PORTS.items():
        path = os.path.join(PER_PORT_DIR, f"{prefix}_merged.csv")
        if os.path.exists(path):
            all_merged[port_name] = pd.read_csv(path, parse_dates=["date"])
    run_state_labeler(all_merged)