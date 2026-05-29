import os
import sys
import ast
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.constants import (
    DISRUPTION_DIR, PROCESSED_DIR, PORTS, PORT_KEYWORDS,
)
from src.utils.logger import get_logger

log = get_logger("gdelt_parser")

GDELT_FILE = os.path.join(DISRUPTION_DIR, "gdelt_disruptions_2023_2026.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "gdelt_parsed.csv")

GDELT_VALIDATION_THRESHOLD_KEEP   = 0.4   
GDELT_VALIDATION_THRESHOLD_REMOVE = 0.3   

def _parse_gdelt_date(seendate_series: pd.Series) -> pd.Series:
    """
    GDELT seendate format: 20260203T110000Z
    We extract just the YYYYMMDD prefix.
    """
    parsed = pd.to_datetime(
        seendate_series.astype(str).str[:8],
        format="%Y%m%d",
        errors="coerce",
    )
    bad = parsed.isna().sum()
    if bad:
        log.warning(f"  {bad} GDELT rows with unparseable seendate — dropped")
    return parsed

def _find_port_mentions(title: str) -> list:
    """
    Scans article title for port keywords.
    Returns list of matching port names.
    If no port is mentioned → article is treated as global
    (assigned to ALL 7 ports with weight=1/7).
    """
    if pd.isna(title) or str(title).strip() == "":
        return []
    title_lower = str(title).lower()
    return [
        port for port, keywords in PORT_KEYWORDS.items()
        if any(kw in title_lower for kw in keywords)
    ]

def parse_gdelt(gdelt_path: str = GDELT_FILE) -> pd.DataFrame:
    """
    Parses raw GDELT CSV → flat daily disruption signal per port.
    """
    if not os.path.exists(gdelt_path):
        raise FileNotFoundError(
            f"GDELT file not found: {gdelt_path}\n"
            f"Expected: data/raw/Disruptions/gdelt_disruptions_2023_2026.csv"
        )

    raw = pd.read_csv(gdelt_path, encoding="utf-8-sig")
    raw.columns = [c.replace("\ufeff", "").strip() for c in raw.columns]
    log.info(f"  Loaded GDELT: {len(raw):,} rows | columns: {list(raw.columns)}")

    if "seendate" in raw.columns:
        raw["date"] = _parse_gdelt_date(raw["seendate"])
    elif "date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    else:
        raise ValueError("GDELT CSV has no 'seendate' or 'date' column")

    raw = raw.dropna(subset=["date"])

    title_col = next(
        (c for c in raw.columns if c.lower() in ("title", "headline", "articletitle")),
        None,
    )
    if title_col:
        raw["ports_mentioned"] = raw[title_col].apply(_find_port_mentions)
    else:
        log.warning("  No title column found — all articles treated as global")
        raw["ports_mentioned"] = [[] for _ in range(len(raw))]

    query_col = next(
        (c for c in raw.columns if c.lower() in ("query", "querytype", "category")),
        None,
    )

    all_port_names = list(PORTS.keys())
    records = []

    for _, row in raw.iterrows():
        ports = row["ports_mentioned"]
        is_global = len(ports) == 0
        target_ports = all_port_names if is_global else ports

        for port in target_ports:
            records.append({
                "date":           row["date"],
                "port":           port,
                "query_type":     row[query_col] if query_col else "unknown",
                "is_global":      int(is_global),
                "disruption_flag": 1,
            })

    if not records:
        log.warning("  No GDELT records generated — check port keyword matching")
        return pd.DataFrame(columns=["date", "port", "query_type",
                                     "disruption_flag", "disruption_count"])

    flat = pd.DataFrame(records)

    daily = (
        flat.groupby(["date", "port"])
        .agg(
            disruption_count=("disruption_flag", "sum"),
            disruption_flag=("disruption_flag", "max"),
            global_article_count=("is_global", "sum"),
        )
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])

    log.info(
        f"  GDELT parsed: {len(daily):,} port-day rows | "
        f"unique dates={daily['date'].nunique()} | "
        f"port coverage={daily['port'].value_counts().to_dict()}"
    )
    return daily

def merge_gdelt_into_ports(all_merged: dict, gdelt_daily: pd.DataFrame) -> dict:
    """
    Left-joins daily GDELT signal onto each port DataFrame.
    Missing days (no news) → disruption_flag=0, disruption_count=0.
    """
    for port_name, df in all_merged.items():
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        port_gdelt = gdelt_daily[gdelt_daily["port"] == port_name][
            ["date", "disruption_count", "disruption_flag"]
        ]

        merged = df.merge(port_gdelt, on="date", how="left")
        merged["disruption_flag"]  = merged["disruption_flag"].fillna(0).astype(int)
        merged["disruption_count"] = merged["disruption_count"].fillna(0).astype(int)

        disruption_days = merged["disruption_flag"].sum()
        log.info(f"  {port_name}: disruption days={disruption_days}/{len(merged)}")
        all_merged[port_name] = merged

    return all_merged

def validate_gdelt_correlation(all_merged: dict) -> dict:
    """
    Computes Pearson r between disruption_count and congestion_score
    for each port. Returns dict {port: r_value}.

    Prints decision: KEEP or REMOVE GDELT from paper.
    """
    log.info("\n── GDELT Validation (Pearson correlation with congestion) ──")
    correlations = {}

    for port_name, df in all_merged.items():
        df_2023 = df[df["date"] >= "2023-01-01"].copy()

        if "disruption_count" not in df_2023.columns:
            log.warning(f"  {port_name}: disruption_count missing — skipping")
            continue

        x = df_2023["disruption_count"].fillna(0)
        y = df_2023["congestion_score"].fillna(0)

        if x.std() == 0 or y.std() == 0:
            r = 0.0
            p = 1.0
        else:
            r, p = pearsonr(x, y)

        correlations[port_name] = r
        log.info(f"  {port_name}: r={r:.3f} (p={p:.4f})")

    if not correlations:
        return correlations

    r_values   = list(correlations.values())
    above_keep = sum(r > GDELT_VALIDATION_THRESHOLD_KEEP for r in r_values)
    above_rem  = sum(r < GDELT_VALIDATION_THRESHOLD_REMOVE for r in r_values)
    majority   = len(r_values) // 2 + 1

    log.info(f"\n  Ports with r > {GDELT_VALIDATION_THRESHOLD_KEEP}: {above_keep}/7")
    log.info(f"  Ports with r < {GDELT_VALIDATION_THRESHOLD_REMOVE}: {above_rem}/7")

    if above_keep >= majority:
        log.info("  ✅ DECISION: KEEP GDELT — correlation is meaningful")
        log.info("     Include GDELT in paper. Add correlation table.")
    elif above_rem >= majority:
        log.warning("  ⚠️  DECISION: REMOVE GDELT — correlation too weak")
        log.warning("     2 strong sources > 3 weak ones. Drop from paper.")
    else:
        log.info("  ⚠️  DECISION: MIXED — consider keeping as exploratory feature only")
        log.info("     Acknowledge limitation in Discussion section.")

    return correlations

def run_gdelt_parser(all_merged: dict) -> tuple:
    """
    Returns (all_merged with GDELT columns, gdelt_daily DataFrame,
             correlations dict).
    """
    log.info("=" * 60)
    log.info("STEP 3 — GDELT Parser")
    log.info("=" * 60)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    gdelt_daily = parse_gdelt()
    gdelt_daily.to_csv(OUTPUT_FILE, index=False)
    log.info(f"  Saved → {OUTPUT_FILE}")

    all_merged = merge_gdelt_into_ports(all_merged, gdelt_daily)
    correlations = validate_gdelt_correlation(all_merged)

    log.info("\n✅ STEP 3 COMPLETE — GDELT parsed and merged")
    return all_merged, gdelt_daily, correlations


if __name__ == "__main__":
    from src.utils.constants import PER_PORT_DIR, PORTS
    all_merged = {}
    for port_name, prefix in PORTS.items():
        path = os.path.join(PER_PORT_DIR, f"{prefix}_merged.csv")
        if os.path.exists(path):
            all_merged[port_name] = pd.read_csv(path, parse_dates=["date"])
    run_gdelt_parser(all_merged)