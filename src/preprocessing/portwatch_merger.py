# =============================================================================
# TwinMesh · portwatch_merger.py
# STEP 1 — Load & merge the 3 IMF PortWatch files for every port
#
# FIX APPLIED (Day 2 diagnosis):
#   congestion_score is capped at CONGESTION_CAP = 20.0
#   Reason: outgoing_DWT = 0 on idle/weekend days causes
#   congestion_score = incoming_DWT / 1 = raw DWT value (e.g. 85,124)
#   This is a data artifact, not real congestion.
#   Cap prevents Shanghai (234), Nhava Sheva (1034), Hamburg (54)
#   outliers from breaking state labeling and era comparison.
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.constants import (
    PORTWATCH_DIR, PER_PORT_DIR, PORTS,
    VESSEL_TYPES_RAW, VESSEL_RENAME, IMF_EXTRA_COLS,
    STATE_THRESHOLDS,
)
from src.utils.logger import get_logger

log = get_logger("portwatch_merger")

# ── FIX 1: Congestion score cap ───────────────────────────────────────────────
# Why 20.0? Rotterdam's legitimate max is ~16. Any value above 20 is
# a data artifact (outgoing_DWT near zero on idle days).
# This cap preserves all real congestion variation while removing outliers.
CONGESTION_CAP = 20.0

# Minimum outgoing DWT to trust the congestion ratio.
# Below this, port was effectively idle → use port median instead.
MIN_OUTGOING_DWT = 100.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip BOM characters, surrounding quotes, and whitespace from all
    column names. IMF PortWatch CSVs often have a UTF-8 BOM on the
    first column and quoted headers.
    """
    df.columns = [
        c.replace("\ufeff", "").strip().strip('"').strip()
        for c in df.columns
    ]
    return df


def _parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename whatever the date column is called to 'date' and parse it.
    IMF files use 'DateTime' as the date column name.
    """
    date_candidates = ["DateTime", "datetime", "Date", "date"]
    for cand in date_candidates:
        if cand in df.columns:
            df.rename(columns={cand: "date"}, inplace=True)
            break

    if "date" not in df.columns:
        raise ValueError(f"No date column found. Columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad = df["date"].isna().sum()
    if bad:
        log.warning(f"  {bad} rows with unparseable dates — dropped")
        df = df.dropna(subset=["date"])

    return df


def _rename_vessel_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Rename vessel-type columns from their raw IMF names to safe snake_case,
    adding a file-type prefix (calls_ / incoming_dwt_ / outgoing_dwt_).
    prefix = "calls" | "incoming_dwt" | "outgoing_dwt"
    """
    rename_map = {}
    for raw, safe in VESSEL_RENAME.items():
        if raw in df.columns:
            rename_map[raw] = f"{prefix}_{safe}"

    if prefix == "calls":
        for col in IMF_EXTRA_COLS:
            if col in df.columns:
                safe_col = col.lower().replace(" ", "_").replace(":", "").replace("/", "_")
                rename_map[col] = safe_col

    df.rename(columns=rename_map, inplace=True)
    return df


def _numeric_coerce(df: pd.DataFrame, exclude: list) -> pd.DataFrame:
    """Force all non-excluded columns to numeric, filling NaN with 0."""
    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


# ── Per-port loader ───────────────────────────────────────────────────────────

def load_and_merge_port(port_name: str, file_prefix: str) -> pd.DataFrame:
    """
    Loads the 3 PortWatch CSVs for one port and merges them on date.
    Returns a wide DataFrame with all vessel-type columns.
    """
    port_dir = os.path.join(PORTWATCH_DIR, port_name)

    paths = {
        "calls":        os.path.join(port_dir, f"{file_prefix}_port_calls.csv"),
        "incoming_dwt": os.path.join(port_dir, f"{file_prefix}_incoming.csv"),
        "outgoing_dwt": os.path.join(port_dir, f"{file_prefix}_outgoing.csv"),
    }

    frames = {}
    for ftype, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing file for {port_name}: {path}\n"
                f"Expected: data/raw/PortWatch/{port_name}/{file_prefix}_{ftype.replace('_dwt','')}.csv"
            )
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = _clean_columns(df)
        df = _parse_date_column(df)
        df = _rename_vessel_columns(df, ftype)
        df = _numeric_coerce(df, exclude=["date"])
        frames[ftype] = df
        log.debug(f"  Loaded {ftype:15s}: {len(df):,} rows, {len(df.columns)} cols")

    calls_df = frames["calls"]
    inc_df   = frames["incoming_dwt"]
    out_df   = frames["outgoing_dwt"]

    inc_cols = ["date"] + [c for c in inc_df.columns if c.startswith("incoming_dwt_")]
    out_cols = ["date"] + [c for c in out_df.columns if c.startswith("outgoing_dwt_")]

    merged = calls_df.merge(inc_df[inc_cols], on="date", how="left")
    merged = merged.merge(out_df[out_cols], on="date", how="left")
    merged["port"] = port_name

    log.info(f"  {port_name}: merged {len(merged):,} rows")
    return merged


# ── Derived features ──────────────────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features on top of the merged PortWatch data.

    KEY FIX HERE:
      1. Replace congestion_score on idle days (outgoing < MIN_OUTGOING_DWT)
         with the port's own median congestion — removes division artifacts.
      2. Cap congestion_score at CONGESTION_CAP = 20.0 — removes remaining
         extreme outliers (Shanghai 234, Nhava Sheva 1034, Hamburg 54 in 2019).
    """
    df = df.sort_values("date").copy()

    call_cols = [c for c in df.columns if c.startswith("calls_") and
                 any(v in c for v in ["container", "dry_bulk", "general_cargo", "roro", "tanker"])]
    inc_cols  = [c for c in df.columns if c.startswith("incoming_dwt_")]
    out_cols  = [c for c in df.columns if c.startswith("outgoing_dwt_")]

    # ── Volume features ───────────────────────────────────────────────────────
    df["total_calls"]        = df[call_cols].sum(axis=1)
    df["total_incoming_dwt"] = df[inc_cols].sum(axis=1)
    df["total_outgoing_dwt"] = df[out_cols].sum(axis=1)

    df["container_ratio"] = (
        df.get("calls_container", pd.Series(0, index=df.index)) / (df["total_calls"] + 1)
    )

    # ── Flow features ─────────────────────────────────────────────────────────
    df["net_dwt_flow"] = df["total_incoming_dwt"] - df["total_outgoing_dwt"]

    # ── Congestion score — with idle-day fix ──────────────────────────────────
    # Step 1: raw ratio
    df["congestion_score_raw"] = (
        df["total_incoming_dwt"] / (df["total_outgoing_dwt"] + 1)
    )

    # Step 2: identify idle days (outgoing near zero = data artifact)
    idle_mask = df["total_outgoing_dwt"] < MIN_OUTGOING_DWT

    # Step 3: compute port median on NON-idle days (robust central value)
    non_idle_median = df.loc[~idle_mask, "congestion_score_raw"].median()
    if pd.isna(non_idle_median) or non_idle_median == 0:
        non_idle_median = 1.0   # fallback for ports with very sparse data

    # Step 4: replace idle days with median
    df["congestion_score"] = df["congestion_score_raw"].copy()
    df.loc[idle_mask, "congestion_score"] = non_idle_median

    # Step 5: cap at CONGESTION_CAP — removes remaining extreme outliers
    n_before_cap = (df["congestion_score"] > CONGESTION_CAP).sum()
    df["congestion_score"] = df["congestion_score"].clip(upper=CONGESTION_CAP)

    if n_before_cap > 0:
        log.info(
            f"  congestion_score: {n_before_cap} values capped at {CONGESTION_CAP} | "
            f"idle days fixed: {idle_mask.sum()} | "
            f"median used: {non_idle_median:.4f}"
        )

    # Drop the raw column (keep only clean version)
    df.drop(columns=["congestion_score_raw"], inplace=True)

    # ── Queue buildup ─────────────────────────────────────────────────────────
    df["queue_buildup_7d"] = df["net_dwt_flow"].rolling(7, min_periods=1).sum()

    # ── Year-over-year change ─────────────────────────────────────────────────
    ma_col       = next((c for c in df.columns if "7-day" in c.lower()
                         and "prior" not in c.lower()), None)
    prior_ma_col = next((c for c in df.columns if "prior" in c.lower()
                         and "7" in c.lower()), None)

    if ma_col and prior_ma_col:
        df["ma_7day"]        = df[ma_col]
        df["prior_year_ma"]  = df[prior_ma_col]
        prior_safe           = df["prior_year_ma"].replace(0, np.nan)
        df["yoy_pct_change"] = (
            (df["ma_7day"] - df["prior_year_ma"]) / prior_safe
        ).fillna(0) * 100
    else:
        log.warning("  YoY MA columns not found — yoy_pct_change set to 0")
        df["ma_7day"]        = df["total_calls"]
        df["prior_year_ma"]  = np.nan
        df["yoy_pct_change"] = 0.0

    # ── Anomaly flag ──────────────────────────────────────────────────────────
    # Computed AFTER capping so threshold is based on clean data
    mu        = df["congestion_score"].mean()
    sigma     = df["congestion_score"].std()
    threshold = mu + STATE_THRESHOLDS["anomaly_sigma"] * sigma
    df["anomaly_flag"] = (df["congestion_score"] > threshold).astype(int)

    log.debug(
        f"  Features added | anomaly_threshold={threshold:.3f} | "
        f"anomalies={df['anomaly_flag'].sum()} | "
        f"congestion range: [{df['congestion_score'].min():.3f}, {df['congestion_score'].max():.3f}]"
    )
    return df


# ── Main runner ───────────────────────────────────────────────────────────────

def run_portwatch_merger() -> dict:
    """
    Processes all 7 ports. Returns dict {port_name: DataFrame}.
    Saves per-port CSVs to data/processed/per_port/.
    """
    os.makedirs(PER_PORT_DIR, exist_ok=True)
    all_merged = {}

    log.info("=" * 60)
    log.info("STEP 1 — PortWatch Merger (with congestion cap fix)")
    log.info(f"  CONGESTION_CAP = {CONGESTION_CAP}")
    log.info(f"  MIN_OUTGOING_DWT = {MIN_OUTGOING_DWT}")
    log.info("=" * 60)

    for port_name, file_prefix in PORTS.items():
        log.info(f"\n── {port_name} ({file_prefix}) ──")
        try:
            df = load_and_merge_port(port_name, file_prefix)
            df = add_derived_features(df)

            out_path = os.path.join(PER_PORT_DIR, f"{file_prefix}_merged.csv")
            df.to_csv(out_path, index=False)

            log.info(
                f"  Saved {len(df):,} rows → {out_path} | "
                f"anomalies={df['anomaly_flag'].sum()} | "
                f"congestion max={df['congestion_score'].max():.3f} | "
                f"date: {df['date'].min().date()} → {df['date'].max().date()}"
            )
            all_merged[port_name] = df

        except FileNotFoundError as e:
            log.error(f"  {e}")
        except Exception as e:
            log.exception(f"  Unexpected error for {port_name}: {e}")

    log.info(f"\n✅ STEP 1 COMPLETE — {len(all_merged)}/7 ports processed")
    return all_merged


if __name__ == "__main__":
    run_portwatch_merger()