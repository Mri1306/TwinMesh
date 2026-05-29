import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.constants import (
    PROCESSED_DIR, UNIFIED_DIR, HISTORICAL_DIR,
    PORTS, STATES, STATE_MAP,
    COMBINED_START_DATE, HISTORICAL_START, HISTORICAL_END,
    SPLIT_TRAIN, SPLIT_VAL,
)
from src.utils.logger import get_logger

log = get_logger("dataset_builder")

COMBINED_FILE    = os.path.join(PROCESSED_DIR, "all_ports_combined.csv")
TRANSITIONS_FILE = os.path.join(PROCESSED_DIR, "markov_transitions.csv")
HAWKES_FILE      = os.path.join(PROCESSED_DIR, "hawkes_events.csv")
JSON_FILE        = os.path.join(UNIFIED_DIR,   "twinmesh_dataset_full.json")
TRAIN_FILE       = os.path.join(UNIFIED_DIR,   "twinmesh_training.csv")
VAL_FILE         = os.path.join(UNIFIED_DIR,   "twinmesh_validation.csv")
TEST_FILE        = os.path.join(UNIFIED_DIR,   "twinmesh_test.csv")
HISTORICAL_FILE  = os.path.join(HISTORICAL_DIR,"portwatch_2019_2022.csv")

def build_combined(all_merged: dict, start_date: str = COMBINED_START_DATE) -> pd.DataFrame:
    """
    Stack all 7 port DataFrames (filtered to start_date+).
    Sorts by date then port. Adds t_numeric for Hawkes fitting.
    """
    parts = []
    for port_name, df in all_merged.items():
        filtered = df[df["date"] >= start_date].copy()
        filtered["port"] = port_name
        parts.append(filtered)

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values(["date", "port"]).reset_index(drop=True)

    t0 = combined["date"].min()
    combined["t_numeric"] = (combined["date"] - t0).dt.days.astype(float)

    log.info(
        f"  Combined dataset: {len(combined):,} rows | "
        f"{combined['date'].min().date()} → {combined['date'].max().date()} | "
        f"ports: {combined['port'].nunique()}"
    )

    dist = combined["state"].value_counts(normalize=True).round(3)
    log.info(f"  Global state distribution:\n{dist.to_string()}")

    return combined

HAWKES_KEEP_COLS = [
    "date", "port", "t_numeric",
    "total_calls", "container_ratio",
    "total_incoming_dwt", "total_outgoing_dwt", "net_dwt_flow",
    "congestion_score", "queue_buildup_7d",
    "calls_container", "calls_dry_bulk", "calls_roro",
    "calls_tanker", "calls_general_cargo",
    "incoming_dwt_container", "incoming_dwt_dry_bulk",
    "incoming_dwt_roro", "incoming_dwt_tanker",
    "weather_risk_score", "storm_alert",
    "disruption_flag", "disruption_count",
    "state", "state_numeric",
]


def build_hawkes_events(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Hawkes events = rows where anomaly_flag = 1.
    These are the event timestamps used to fit HawkesExpKern.
    Also adds per-vessel-type anomaly columns.
    """
    events = combined[combined["anomaly_flag"] == 1].copy()

    keep = [c for c in HAWKES_KEEP_COLS if c in events.columns]
    events = events[keep]

    vessel_types = ["container", "dry_bulk", "roro", "tanker", "general_cargo"]
    for vtype in vessel_types:
        calls_col = f"calls_{vtype}"
        if calls_col in combined.columns:
            mu    = combined[calls_col].mean()
            sigma = combined[calls_col].std()
            thresh = mu + 1.5 * sigma
            events[f"anomaly_{vtype}"] = (events[calls_col] > thresh).astype(int) \
                if calls_col in events.columns else 0

    log.info(
        f"  Hawkes events: {len(events):,} anomaly rows | "
        f"{events['port'].value_counts().to_dict()}"
    )
    return events

def build_historical(all_merged: dict) -> pd.DataFrame:
    """
    Pre-pandemic + pandemic era: 2019-01-01 to 2022-12-31.
    Used for era comparison in paper (Table, Figure 1).
    No weather or GDELT — PortWatch only.
    """
    parts = []
    for port_name, df in all_merged.items():
        hist = df[
            (df["date"] >= HISTORICAL_START) &
            (df["date"] <= HISTORICAL_END)
        ].copy()
        hist["era"] = hist["date"].apply(
            lambda d: "pre_covid" if d.year < 2020 else "covid"
        )
        parts.append(hist)

    if not parts:
        log.warning("  No historical data found (2019–2022)")
        return pd.DataFrame()

    historical = pd.concat(parts, ignore_index=True)
    historical = historical.sort_values(["date", "port"])
    log.info(
        f"  Historical dataset: {len(historical):,} rows | "
        f"pre_covid={len(historical[historical['era']=='pre_covid']):,} | "
        f"covid={len(historical[historical['era']=='covid']):,}"
    )
    return historical

def chronological_split(combined: pd.DataFrame) -> tuple:
    """
    Chronological split — NOT random.
    Rationale: supply chain time series must not leak future into past.
    70% train · 15% val · 15% test.
    """
    combined_sorted = combined.sort_values("date").reset_index(drop=True)
    n = len(combined_sorted)

    n_train = int(n * SPLIT_TRAIN)
    n_val   = int(n * SPLIT_VAL)

    train = combined_sorted.iloc[:n_train]
    val   = combined_sorted.iloc[n_train : n_train + n_val]
    test  = combined_sorted.iloc[n_train + n_val :]

    log.info(
        f"\n  Chronological split:"
        f"\n    Train: {len(train):,} rows | "
        f"{train['date'].min().date()} → {train['date'].max().date()}"
        f"\n    Val:   {len(val):,} rows | "
        f"{val['date'].min().date()} → {val['date'].max().date()}"
        f"\n    Test:  {len(test):,} rows | "
        f"{test['date'].min().date()} → {test['date'].max().date()}"
    )

    s3_test = (test["state"] == "S3").sum()
    if s3_test < 5:
        log.warning(
            f"  ⚠️  Only {s3_test} S3 events in test set — "
            f"consider adjusting split date or state thresholds"
        )

    return train, val, test

def export_json(combined: pd.DataFrame, path: str) -> None:
    """
    Export full combined dataset as JSON for the TwinMesh dashboard.
    NaN → 0, dates → string.
    """
    records = combined.fillna(0).to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[k] = float(v)
            elif isinstance(v, pd.Timestamp):
                rec[k] = str(v.date())

    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)
    log.info(f"  Exported JSON: {len(records):,} records → {path}")

def print_dataset_summary(combined: pd.DataFrame, train, val, test) -> None:
    log.info("\n" + "=" * 60)
    log.info("DATASET SUMMARY")
    log.info("=" * 60)
    log.info(f"  Total records:   {len(combined):,}")
    log.info(f"  Ports:           {combined['port'].nunique()}")
    log.info(f"  Date range:      {combined['date'].min().date()} → {combined['date'].max().date()}")
    log.info(f"  Features:        {len(combined.columns)}")
    log.info(f"  Training rows:   {len(train):,}")
    log.info(f"  Validation rows: {len(val):,}")
    log.info(f"  Test rows:       {len(test):,}")
    log.info(f"\n  State counts:")
    for s in STATES:
        cnt = (combined["state"] == s).sum()
        pct = cnt / len(combined) * 100
        log.info(f"    {s}: {cnt:,} ({pct:.1f}%)")
    log.info(f"\n  Anomaly events:  {combined['anomaly_flag'].sum():,}")
    log.info(f"  Disruption days: {combined['disruption_flag'].sum():,}")
    log.info(f"  Storm events:    {combined['storm_alert'].sum():,}")

def run_dataset_builder(all_merged: dict, markov_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full dataset export. Returns the combined DataFrame.
    """
    log.info("=" * 60)
    log.info("STEP 5 — Dataset Builder")
    log.info("=" * 60)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(UNIFIED_DIR,   exist_ok=True)
    os.makedirs(HISTORICAL_DIR,exist_ok=True)

    combined = build_combined(all_merged)
    combined.to_csv(COMBINED_FILE, index=False)
    log.info(f"  Saved all_ports_combined.csv ({len(combined):,} rows)")

    if markov_df is not None and len(markov_df):
        markov_df.to_csv(TRANSITIONS_FILE, index=False)
        log.info(f"  Saved markov_transitions.csv ({len(markov_df):,} transitions)")

    hawkes_events = build_hawkes_events(combined)
    hawkes_events.to_csv(HAWKES_FILE, index=False)
    log.info(f"  Saved hawkes_events.csv ({len(hawkes_events):,} events)")

    historical = build_historical(all_merged)
    if len(historical):
        historical.to_csv(HISTORICAL_FILE, index=False)
        log.info(f"  Saved portwatch_2019_2022.csv ({len(historical):,} rows)")

    
    train, val, test = chronological_split(combined)
    train.to_csv(TRAIN_FILE, index=False)
    val.to_csv(VAL_FILE,     index=False)
    test.to_csv(TEST_FILE,   index=False)

    
    export_json(combined, JSON_FILE)

    
    print_dataset_summary(combined, train, val, test)

    log.info("\n✅ STEP 5 COMPLETE — all datasets exported")
    return combined


if __name__ == "__main__":
    from src.utils.constants import PER_PORT_DIR, PORTS, PROCESSED_DIR as PD
    all_merged = {}
    for port_name, prefix in PORTS.items():
        path = os.path.join(PER_PORT_DIR, f"{prefix}_merged.csv")
        if os.path.exists(path):
            all_merged[port_name] = pd.read_csv(path, parse_dates=["date"])
    markov_path = os.path.join(PD, "markov_transitions.csv")
    markov_df = pd.read_csv(markov_path) if os.path.exists(markov_path) else None
    run_dataset_builder(all_merged, markov_df)