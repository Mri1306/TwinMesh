import os
import sys
import argparse
import time
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger
from src.utils.constants import PROCESSED_DIR, UNIFIED_DIR

log = get_logger("run_day7")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "digital_twin")

EXPECTED_OUTPUTS = {
    "Replay log":          "twin_replay_log.csv",
    "Alert summary":       "twin_alert_summary.csv",
    "Dashboard":           "twin_dashboard.png",
    "Rotterdam replay":    "twin_replay_rotterdam.png",
    "Singapore replay":    "twin_replay_singapore.png",
    "Shanghai replay":     "twin_replay_shanghai.png",
}

def verify_day6_complete() -> bool:
    required = [
        os.path.join(UNIFIED_DIR, "twinmesh_training.csv"),
        os.path.join(UNIFIED_DIR, "twinmesh_test.csv"),
        os.path.join(PROCESSED_DIR, "all_ports_combined.csv"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        log.error("  Day 1-6 outputs missing:")
        for p in missing:
            log.error(f"    {p}")
        return False
    log.info("  Day 1-6 outputs verified")
    return True


def check_day7_outputs() -> bool:
    log.info("\n  ── Day 7 Output Verification ──")
    all_ok = True
    for label, fname in EXPECTED_OUTPUTS.items():
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            log.info(f"    {label:<25} {fname} ({size_kb:.0f} KB)")
        else:
            log.warning(f"    MISSING: {label:<25} {fname}")
            all_ok = False
    return all_ok

def run_morning() -> pd.DataFrame:
    log.info("\n" + "=" * 60)
    log.info("MORNING — Digital Twin Replay")
    log.info("=" * 60)
    t = time.time()
    from src.digital_twin.twin_orchestrator import run_digital_twin
    log_df = run_digital_twin()
    log.info(f"\n  Morning done in {time.time()-t:.1f}s")
    return log_df


def run_afternoon(log_df: pd.DataFrame = None) -> None:
    log.info("\n" + "=" * 60)
    log.info("AFTERNOON — Dashboard + Per-Port Plots")
    log.info("=" * 60)
    t = time.time()

    if log_df is None:
        path = os.path.join(RESULTS_DIR, "twin_replay_log.csv")
        if not os.path.exists(path):
            log.error("  Replay log not found — run morning first")
            return
        log_df = pd.read_csv(path, parse_dates=["date"])

    from src.digital_twin.twin_orchestrator import (
        plot_twin_dashboard, plot_per_port_replay
    )
    from src.utils.constants import UNIFIED_DIR, PROCESSED_DIR

    test     = pd.read_csv(
        os.path.join(UNIFIED_DIR, "twinmesh_test.csv"),
        parse_dates=["date"]
    )
    plot_twin_dashboard(log_df, test)

    for port in ["Rotterdam", "Singapore", "Shanghai", "Nhava Sheva", "Hamburg"]:
        plot_per_port_replay(log_df, port)

    log.info(f"\n  Afternoon done in {time.time()-t:.1f}s")


def run_evening() -> None:
    log.info("\n" + "=" * 60)
    log.info("EVENING — Live Prediction Demo + Paper Sentences")
    log.info("=" * 60)
    t = time.time()

    import pandas as _pd
    from src.digital_twin.twin_orchestrator import TwinMeshModel, demo_live_prediction
    from src.utils.constants import PROCESSED_DIR

    combined = _pd.read_csv(
        os.path.join(PROCESSED_DIR, "all_ports_combined.csv"),
        parse_dates=["date"]
    )
    twin_model = TwinMeshModel()
    if twin_model.load(combined):
        demo_live_prediction(twin_model, combined)

    log.info(f"\n  Evening done in {time.time()-t:.1f}s")

def print_day7_summary(log_df: pd.DataFrame = None) -> None:
    log.info("\n" + "=" * 70)
    log.info("DAY 7 SUMMARY — Digital Twin Complete")
    log.info("=" * 70)

    if log_df is not None and len(log_df):
        n_alerts = log_df["hawkes_alert"].sum() if "hawkes_alert" in log_df else 0
        log.info(f"\n  Replay decisions: {len(log_df):,}")
        log.info(f"  Alerts fired:     {int(n_alerts):,} ({n_alerts/len(log_df):.1%})")
        if "tier0_action" in log_df:
            log.info(f"  Tier-0 Strong:    {(log_df['tier0_action']=='Strong').mean():.1%}")

    log.info(
        f"\n  KEY OUTPUTS:\n"
        f"    Figure 9 (dashboard): results/digital_twin/twin_dashboard.png\n"
        f"    Replay log:           results/digital_twin/twin_replay_log.csv\n"
        f"    Alert summary:        results/digital_twin/twin_alert_summary.csv\n"
        f"\n  NEXT: Paper writing (Days 8-9)\n"
        f"    Abstract deadline: May 30\n"
        f"    Full paper deadline: June 6"
    )

def main():
    parser = argparse.ArgumentParser(description="TwinMesh Day 7 — Digital Twin")
    parser.add_argument("--morning",   action="store_true")
    parser.add_argument("--afternoon", action="store_true")
    parser.add_argument("--evening",   action="store_true")
    parser.add_argument("--check",     action="store_true")
    args = parser.parse_args()

    if args.check:
        sys.exit(0 if check_day7_outputs() else 1)

    log.info("=" * 60)
    log.info("T W I N M E S H  ·  DAY 7")
    log.info("Digital Twin Orchestrator")
    log.info("=" * 60)

    if not verify_day6_complete():
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    t_total      = time.time()
    run_all      = not (args.morning or args.afternoon or args.evening)
    do_morning   = run_all or args.morning
    do_afternoon = run_all or args.afternoon
    do_evening   = run_all or args.evening

    log_df = None

    if do_morning:
        log_df = run_morning()

    if do_afternoon:
        run_afternoon(log_df)

    if do_evening:
        run_evening()

    print_day7_summary(log_df)

    log.info(f"\n  Total Day 7 time: {time.time()-t_total:.1f}s")
    log.info("\n" + "=" * 60)
    log.info("  DAY 7 COMPLETE")
    log.info("  Digital twin running end-to-end")
    log.info("  All results/paper figures ready")
    log.info("=" * 60)


if __name__ == "__main__":
    main()