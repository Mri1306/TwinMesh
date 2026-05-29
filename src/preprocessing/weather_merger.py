import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.constants import (
    WEATHER_DIR, PORTS, WEATHER_COLS, PER_PORT_DIR,
)
from src.utils.logger import get_logger

log = get_logger("weather_merger")

WEATHER_FILL_DEFAULTS = {
    "temperature_mean":    np.nan,   
    "precipitation_sum":   0.0,
    "wind_speed_max":      0.0,
    "storm_alert":         0,
    "heavy_rain_alert":    0,
    "weather_risk_score":  0.0,
}

def load_weather() -> pd.DataFrame:
    """
    Loads maritime_weather_full.csv. Falls back to per-port individual
    files if the combined file is missing.
    """
    combined_path = os.path.join(WEATHER_DIR, "maritime_weather_full.csv")

    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path, encoding="utf-8-sig")
        df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        log.info(f"  Loaded combined weather: {len(df):,} rows")
        return df

    log.warning("  maritime_weather_full.csv not found — stitching per-port files")
    parts = []
    for port_name in PORTS.keys():
        fname = f"{port_name.replace(' ', '_')}_weather.csv"
        path  = os.path.join(WEATHER_DIR, fname)
        if os.path.exists(path):
            part = pd.read_csv(path, encoding="utf-8-sig")
            part.columns = [c.replace("\ufeff", "").strip() for c in part.columns]
            part["port"] = port_name
            parts.append(part)
        else:
            log.warning(f"  Missing per-port weather file: {fname}")

    if not parts:
        raise FileNotFoundError(
            "No weather files found. Expected:\n"
            "  data/raw/Weather/maritime_weather_full.csv\n"
            "  OR data/raw/Weather/<Port>_weather.csv for each port"
        )

    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    log.info(f"  Stitched {len(parts)} per-port files → {len(df):,} rows")
    return df


def validate_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """
    Checks required columns exist, clamps risk score to [0,1],
    and ensures storm/rain alerts are binary integers.
    """
    required = ["date", "port", "weather_risk_score", "storm_alert"]
    missing  = [c for c in required if c not in weather.columns]
    if missing:
        raise ValueError(f"Weather CSV missing required columns: {missing}")

    if "weather_risk_score" in weather.columns:
        weather["weather_risk_score"] = weather["weather_risk_score"].clip(0, 1)

    for col in ["storm_alert", "heavy_rain_alert"]:
        if col in weather.columns:
            weather[col] = (weather[col] > 0).astype(int)

    num_cols = ["temperature_mean", "precipitation_sum",
                "wind_speed_max", "weather_risk_score"]
    for col in num_cols:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")

    log.info(
        f"  Weather validated | "
        f"storm events={weather['storm_alert'].sum()} | "
        f"heavy rain events={weather.get('heavy_rain_alert', pd.Series([0])).sum()}"
    )
    return weather

def merge_weather_into_ports(
    all_merged: dict,
    weather: pd.DataFrame,
) -> dict:
    """
    Left-joins weather onto each port DataFrame by date.
    Rows with no matching weather (pre-2023) are filled with defaults.

    Returns the updated all_merged dict (in-place modification).
    """
    weather_cols_to_join = [c for c in WEATHER_COLS if c not in ("date", "port")]

    for port_name, df in all_merged.items():
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        port_wx = weather[weather["port"] == port_name][
            ["date"] + weather_cols_to_join
        ].copy()

        if len(port_wx) == 0:
            log.warning(
                f"  {port_name}: NO weather rows found — "
                f"check 'port' column values in weather CSV"
            )

        merged = df.merge(port_wx, on="date", how="left")

        for col, fill_val in WEATHER_FILL_DEFAULTS.items():
            if col in merged.columns:
                merged[col] = merged[col].fillna(fill_val)
            else:
                merged[col] = fill_val

        wx_coverage = merged["weather_risk_score"].notna().sum()
        log.info(
            f"  {port_name}: weather joined | "
            f"coverage={wx_coverage}/{len(merged)} rows | "
            f"storm alerts={int(merged['storm_alert'].sum())}"
        )
        all_merged[port_name] = merged

    return all_merged

def run_weather_merger(all_merged: dict) -> dict:
    log.info("=" * 60)
    log.info("STEP 2 — Weather Merger")
    log.info("=" * 60)

    weather = load_weather()
    weather = validate_weather(weather)
    all_merged = merge_weather_into_ports(all_merged, weather)

    log.info("\n✅ STEP 2 COMPLETE — weather merged into all ports")
    return all_merged


if __name__ == "__main__":
    from src.utils.constants import PER_PORT_DIR, PORTS
    all_merged = {}
    for port_name, prefix in PORTS.items():
        path = os.path.join(PER_PORT_DIR, f"{prefix}_merged.csv")
        if os.path.exists(path):
            all_merged[port_name] = pd.read_csv(path, parse_dates=["date"])
    run_weather_merger(all_merged)