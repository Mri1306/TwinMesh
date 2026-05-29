import os
import sys
import argparse
import time
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger
from src.utils.constants import (
    PROCESSED_DIR, PER_PORT_DIR, UNIFIED_DIR, HISTORICAL_DIR,
    PORTS,
)

log = get_logger("run_preprocessing")

EXPECTED_OUTPUTS = {
    "Per-port merged CSVs": [
        os.path.join(PER_PORT_DIR, f"{prefix}_merged.csv")
        for prefix in PORTS.values()
    ],
    "all_ports_combined":   [os.path.join(PROCESSED_DIR, "all_ports_combined.csv")],
    "state_labeled":        [os.path.join(PROCESSED_DIR, "state_labeled.csv")],
    "markov_transitions":   [os.path.join(PROCESSED_DIR, "markov_transitions.csv")],
    "hawkes_events":        [os.path.join(PROCESSED_DIR, "hawkes_events.csv")],
    "gdelt_parsed":         [os.path.join(PROCESSED_DIR, "gdelt_parsed.csv")],
    "twinmesh_training":    [os.path.join(UNIFIED_DIR,   "twinmesh_training.csv")],
    "twinmesh_validation":  [os.path.join(UNIFIED_DIR,   "twinmesh_validation.csv")],
    "twinmesh_test":        [os.path.join(UNIFIED_DIR,   "twinmesh_test.csv")],
    "unified_json":         [os.path.join(UNIFIED_DIR,   "twinmesh_dataset_full.json")],
    "historical":           [os.path.join(HISTORICAL_DIR,"portwatch_2019_2022.csv")],
}


def check_outputs() -> bool:
    """Verifies all expected output files exist. Returns True if all present."""
    log.info("\n── Output File Check ──")
    all_ok = True
    for label, paths in EXPECTED_OUTPUTS.items():
        for path in paths:
            if os.path.exists(path):
                size_kb = os.path.getsize(path) / 1024
                log.info(f"  ✅ {os.path.basename(path):<45} ({size_kb:,.0f} KB)")
            else:
                log.warning(f"  ❌ MISSING: {path}")
                all_ok = False
    return all_ok


def load_per_port_csvs() -> dict:
    """Reloads per-port merged CSVs from disk (for resuming from Step 2+)."""
    all_merged = {}
    for port_name, prefix in PORTS.items():
        path = os.path.join(PER_PORT_DIR, f"{prefix}_merged.csv")
        if os.path.exists(path):
            all_merged[port_name] = pd.read_csv(path, parse_dates=["date"])
            log.info(f"  Loaded {port_name}: {len(all_merged[port_name]):,} rows")
        else:
            log.error(f"  ❌ Missing per-port file: {path} — run Step 1 first")
    return all_merged

def step1_portwatch(all_merged: dict) -> dict:
    from src.preprocessing.portwatch_merger import run_portwatch_merger
    t = time.time()
    result = run_portwatch_merger()
    log.info(f"  Step 1 done in {time.time()-t:.1f}s")
    return result


def step2_weather(all_merged: dict) -> dict:
    from src.preprocessing.weather_merger import run_weather_merger
    t = time.time()
    result = run_weather_merger(all_merged)
    log.info(f"  Step 2 done in {time.time()-t:.1f}s")
    return result


def step3_gdelt(all_merged: dict) -> tuple:
    from src.preprocessing.gdelt_parser import run_gdelt_parser
    t = time.time()
    result = run_gdelt_parser(all_merged)   # returns (all_merged, gdelt_daily, correlations)
    log.info(f"  Step 3 done in {time.time()-t:.1f}s")
    return result


def step4_states(all_merged: dict) -> tuple:
    from src.preprocessing.state_labeler import run_state_labeler
    t = time.time()
    result = run_state_labeler(all_merged)  # returns (all_merged, markov_df)
    log.info(f"  Step 4 done in {time.time()-t:.1f}s")
    return result


def step5_dataset(all_merged: dict, markov_df) -> pd.DataFrame:
    from src.preprocessing.dataset_builder import run_dataset_builder
    t = time.time()
    result = run_dataset_builder(all_merged, markov_df)
    log.info(f"  Step 5 done in {time.time()-t:.1f}s")
    return result

def save_per_port_final(all_merged: dict) -> None:
    """
    Overwrites per-port CSVs after all enrichment steps complete,
    so they contain the final columns (weather, GDELT, state).
    """
    os.makedirs(PER_PORT_DIR, exist_ok=True)
    for port_name, prefix in PORTS.items():
        if port_name in all_merged:
            path = os.path.join(PER_PORT_DIR, f"{prefix}_merged.csv")
            all_merged[port_name].to_csv(path, index=False)
    log.info("  Per-port CSVs updated with final enriched columns")

def main():
    parser = argparse.ArgumentParser(description="TwinMesh Preprocessing Pipeline")
    parser.add_argument("--step",  type=int, help="Run only this step (1–5)")
    parser.add_argument("--from",  type=int, dest="from_step",
                        help="Run from this step to Step 5")
    parser.add_argument("--check", action="store_true",
                        help="Check that all output files exist")
    args = parser.parse_args()

    if args.check:
        ok = check_outputs()
        sys.exit(0 if ok else 1)

    start_step = args.step or args.from_step or 1
    end_step   = args.step or 5

    log.info("=" * 60)
    log.info("T W I N M E S H  ·  Preprocessing Pipeline")
    log.info(f"Running steps {start_step} → {end_step}")
    log.info("=" * 60)

    t_total    = time.time()
    all_merged = {}
    markov_df  = None

    if start_step <= 1 <= end_step:
        all_merged = step1_portwatch(all_merged)
    elif start_step > 1:
        log.info("  Skipping Step 1 — loading per-port CSVs from disk")
        all_merged = load_per_port_csvs()
        if not all_merged:
            log.error("  Cannot continue — no per-port data loaded. Run Step 1 first.")
            sys.exit(1)

    if start_step <= 2 <= end_step:
        all_merged = step2_weather(all_merged)

    gdelt_correlations = {}
    if start_step <= 3 <= end_step:
        all_merged, _, gdelt_correlations = step3_gdelt(all_merged)

    if start_step <= 4 <= end_step:
        all_merged, markov_df = step4_states(all_merged)
        save_per_port_final(all_merged)

    if start_step <= 5 <= end_step:
        combined = step5_dataset(all_merged, markov_df)

    log.info("\n── Final Output Verification ──")
    check_outputs()

    elapsed = time.time() - t_total
    log.info(f"\n✅ PIPELINE COMPLETE in {elapsed:.1f}s")
    log.info("   Ready for Day 3: Hawkes standalone evaluation")
    log.info("   Next: python src/models/hawkes/hawkes_port_level.py")

    if gdelt_correlations:
        r_values = list(gdelt_correlations.values())
        r_mean   = sum(r_values) / len(r_values)
        log.info(f"\n── GDELT Decision ──")
        log.info(f"   Mean Pearson r across ports: {r_mean:.3f}")
        if r_mean > 0.4:
            log.info("   ✅ KEEP GDELT — include in paper")
        elif r_mean < 0.3:
            log.warning("   ⚠️  REMOVE GDELT — correlation too weak for paper claim")
        else:
            log.info("   ⚠️  BORDERLINE — keep as exploratory, acknowledge limitation")


if __name__ == "__main__":
    main()