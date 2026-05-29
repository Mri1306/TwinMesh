import os
import sys
import argparse
import time

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger
from src.utils.constants import PROCESSED_DIR, UNIFIED_DIR

log = get_logger("run_day4")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "markov_standalone")

EXPECTED_OUTPUTS = {
    "Transition matrix":         "transition_matrix.csv",
    "Pure Markov metrics":       "markov_pure_metrics.csv",
    "Ablation row 3":            "ablation_row3_pure_markov.csv",
    "Transition heatmap":        "markov_transition_heatmap_pure_markov.png",
    "Confusion matrix":          "markov_confusion_matrix_pure_markov.png",
    "Multi-source metrics":      "markov_multisource_metrics.csv",
    "Ablation row 4":            "ablation_row4_portwatch_only.csv",
    "Ablation row 5":            "ablation_row5_portwatch_weather.csv",
    "Feature importance":        "feature_importance.png",
    "Calibration plot":          "calibration_plot.png",
    "Ablation rows 3-5":         "ablation_rows_3_5.csv",
    "Markov vs baseline":        "markov_vs_baseline.png",
}

def verify_day3_complete() -> bool:
    required = [
        os.path.join(UNIFIED_DIR,   "twinmesh_training.csv"),
        os.path.join(UNIFIED_DIR,   "twinmesh_test.csv"),
        os.path.join(PROCESSED_DIR, "all_ports_combined.csv"),
        os.path.join(PROCESSED_DIR, "markov_transitions.csv"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        log.error("  Day 1 outputs missing. Run first:")
        log.error("    python run_preprocessing.py")
        for p in missing:
            log.error(f"    Missing: {p}")
        return False
    log.info("  Day 1-3 outputs verified")
    return True


def check_day4_outputs() -> bool:
    log.info("\n  ── Day 4 Output Verification ──")
    all_ok = True
    for label, fname in EXPECTED_OUTPUTS.items():
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            log.info(f"    {label:<30} {fname} ({size_kb:.0f} KB)")
        else:
            log.warning(f"    MISSING: {label:<30} {fname}")
            all_ok = False
    return all_ok

def run_morning() -> tuple:
    """Model A: Pure 5-state Markov chain."""
    log.info("\n" + "=" * 60)
    log.info("MORNING — Markov Model A: Pure 5-State Chain")
    log.info("=" * 60)
    t = time.time()
    from src.models.markov.markov_chain import run_markov_chain
    result = run_markov_chain()
    log.info(f"\n  Morning done in {time.time()-t:.1f}s")
    return result


def run_afternoon() -> tuple:
    """Model B: Multi-source Markov."""
    log.info("\n" + "=" * 60)
    log.info("AFTERNOON — Markov Model B: Multi-Source")
    log.info("=" * 60)
    t = time.time()
    from src.models.markov.markov_multisource import run_markov_multisource
    result = run_markov_multisource()
    log.info(f"\n  Afternoon done in {time.time()-t:.1f}s")
    return result


def run_evening() -> object:
    """Evening: S3 lead time + ablation rows 3-5."""
    log.info("\n" + "=" * 60)
    log.info("EVENING — Markov Standalone Evaluation")
    log.info("=" * 60)
    t = time.time()
    from src.models.markov.markov_evaluator import run_markov_evaluator
    result = run_markov_evaluator()
    log.info(f"\n  Evening done in {time.time()-t:.1f}s")
    return result

def print_day4_summary() -> None:
    """
    Reads saved ablation rows 3-5 and prints the Day 4 summary.
    Shows what goes into paper Table 1 rows 3-5.
    """
    import pandas as pd

    rows_path = os.path.join(RESULTS_DIR, "ablation_rows_3_5.csv")
    if not os.path.exists(rows_path):
        return

    abl = pd.read_csv(rows_path)

    log.info("\n" + "=" * 70)
    log.info("DAY 4 SUMMARY — Ablation Rows 3-5 (for paper Table 1)")
    log.info("=" * 70)

    for _, row in abl.iterrows():
        acc_str = f"{row['state_accuracy']:.4f}" if pd.notna(row.get("state_accuracy")) else "  -"
        f1_str  = f"{row['macro_f1']:.4f}"       if pd.notna(row.get("macro_f1"))       else "  -"
        auc_str = f"{row['auprc']:.4f}"           if pd.notna(row.get("auprc"))          else "  -"
        log.info(
            f"\n  Row {int(row['row'])}: {row['model_variant']}\n"
            f"    Accuracy={acc_str}  Macro-F1={f1_str}  S3-AUPRC={auc_str}"
        )

    if len(abl) >= 2:
        acc_pure = abl.iloc[0]["state_accuracy"]
        acc_best = abl.iloc[-1]["state_accuracy"]
        lift     = acc_best - acc_pure if pd.notna(acc_pure) and pd.notna(acc_best) else 0
        log.info(
            f"\n  Lift (Row 3 → Row 5): +{lift:.4f} accuracy\n"
            f"  Pure Markov: {acc_pure:.4f}  →  Multi-Source: {acc_best:.4f}"
        )

    log.info(
        f"\n  NEXT STEP:\n"
        f"  Row 6 (Combined Hawkes + Markov) built on Day 5.\n"
        f"  python run_day5.py"
    )

def main():
    parser = argparse.ArgumentParser(description="TwinMesh Day 4 — Markov Standalone")
    parser.add_argument("--morning",   action="store_true")
    parser.add_argument("--afternoon", action="store_true")
    parser.add_argument("--evening",   action="store_true")
    parser.add_argument("--check",     action="store_true")
    args = parser.parse_args()

    if args.check:
        ok = check_day4_outputs()
        sys.exit(0 if ok else 1)

    log.info("=" * 60)
    log.info("T W I N M E S H  ·  DAY 4")
    log.info("Markov Standalone Evaluation")
    log.info("=" * 60)

    if not verify_day3_complete():
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    t_total      = time.time()
    run_all      = not (args.morning or args.afternoon or args.evening)
    do_morning   = run_all or args.morning
    do_afternoon = run_all or args.afternoon
    do_evening   = run_all or args.evening

    if do_morning:
        run_morning()

    if do_afternoon:
        run_afternoon()

    if do_evening:
        run_evening()

    print_day4_summary()

    log.info(f"\n  Total Day 4 time: {time.time()-t_total:.1f}s")
    log.info("\n" + "=" * 60)
    log.info("  DAY 4 COMPLETE")
    log.info("  Results locked in: results/markov_standalone/")
    log.info("  Ablation rows 3-5: results/markov_standalone/ablation_rows_3_5.csv")
    log.info("")
    log.info("  KEY OUTPUTS FOR PAPER:")
    log.info("  Figure 4: results/markov_standalone/markov_transition_heatmap_pure_markov.png")
    log.info("  Table 1 rows 3-5: results/markov_standalone/ablation_rows_3_5.csv")
    log.info("")
    log.info("  Next: python run_day5.py  (Combined TwinMesh + MDP)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()