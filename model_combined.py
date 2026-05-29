import os
import sys
import argparse
import time
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger
from src.utils.constants import (
    PROCESSED_DIR,
    UNIFIED_DIR,
    STATES,
    PORTS,
)

log = get_logger("run_day5")

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "combined"
)

AUDIT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "audit"
)

MARKOV_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "markov_standalone"
)

PORT_NAMES = list(PORTS.keys())

def verify_day4_complete():

    required = [
        os.path.join(
            UNIFIED_DIR,
            "twinmesh_training.csv"
        ),

        os.path.join(
            UNIFIED_DIR,
            "twinmesh_test.csv"
        ),

        os.path.join(
            PROCESSED_DIR,
            "all_ports_combined.csv"
        ),

        os.path.join(
            MARKOV_DIR,
            "ablation_rows_3_5.csv"
        ),
    ]

    missing = [
        p for p in required
        if not os.path.exists(p)
    ]

    if missing:

        log.error("  Day 1-4 outputs missing:")

        for p in missing:
            log.error(f"    {p}")

        return False

    log.info("  Day 1-4 outputs verified ✓")

    return True

def run_morning_v2():

    log.info("\n" + "=" * 60)
    log.info("MORNING — TwinMesh V2")
    log.info("XGBoost + Advanced Features")
    log.info("=" * 60)

    t = time.time()

    try:

        from src.models.combined.twinmesh_core_v2 import (
            run_twinmesh_v2
        )

        model, scaler, metrics = run_twinmesh_v2()

        log.info(
            f"\n  Morning done in "
            f"{time.time()-t:.1f}s"
        )

        return model, scaler, metrics

    except Exception as e:

        log.error(
            f"  ERROR running TwinMesh V2: {e}"
        )

        raise

def run_morning_v1():

    log.info("\n" + "=" * 60)
    log.info("MORNING — TwinMesh V1")
    log.info("Original LogisticRegression")
    log.info("=" * 60)

    t = time.time()

    try:

        from src.models.combined.twinmesh_core import (
            run_twinmesh_core
        )

        result = run_twinmesh_core()

        log.info(
            f"\n  Morning done in "
            f"{time.time()-t:.1f}s"
        )

        return result

    except Exception as e:

        log.error(
            f"  ERROR running TwinMesh V1: {e}"
        )

        raise

def run_afternoon():

    log.info("\n" + "=" * 60)
    log.info("AFTERNOON — MDP Safety Verification")
    log.info("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    policy_path = os.path.join(
        AUDIT_DIR,
        "mdp_policy_table.csv"
    )

    if not os.path.exists(policy_path):

        log.warning(
            "  Missing mdp_policy_table.csv"
        )

        return {}

    policy_df = pd.read_csv(policy_path)

    policy_map = {
        (row["state"], int(row["tier"])): row["optimal_action"]
        for _, row in policy_df.iterrows()
    }

    test_df = pd.read_csv(
        os.path.join(
            UNIFIED_DIR,
            "twinmesh_test.csv"
        ),
        parse_dates=["date"]
    )

    TIERS = [0, 1, 2, 3]

    decisions = []

    tier0_violations = 0

    total_policy_cost = 0.0
    total_strong_cost = 0.0

    ACTION_COSTS = {
        "Strong": 1.0,
        "Escrow": 0.6,
        "CRDT": 0.2,
        "Stale": 0.1,
        "Block": 2.0,
    }

    for _, row in test_df.iterrows():

        state = row["state"]

        for tier in TIERS:

            action = policy_map.get(
                (state, tier),
                "Strong"
            )

            if tier == 0 and action not in [
                "Strong",
                "Block"
            ]:

                tier0_violations += 1

            total_policy_cost += ACTION_COSTS.get(
                action,
                1.0
            )

            total_strong_cost += 1.0

            decisions.append({
                "date": str(row["date"]),
                "port": row["port"],
                "state": state,
                "tier": tier,
                "action": action,
            })

    cost_reduction = (
        (
            total_strong_cost
            - total_policy_cost
        )
        /
        total_strong_cost
    ) * 100

    safety_report = {
        "tier0_violations": tier0_violations,
        "total_decisions": len(decisions),
        "cost_reduction_pct": round(
            cost_reduction,
            2
        ),
        "safe": tier0_violations == 0,
    }

    log.info("\n" + "=" * 60)
    log.info("MDP SAFETY REPORT")
    log.info("=" * 60)

    log.info(
        f"  Tier-0 violations: "
        f"{tier0_violations}"
    )

    log.info(
        f"  Cost reduction: "
        f"{cost_reduction:.2f}%"
    )

    pd.DataFrame([safety_report]).to_csv(
        os.path.join(
            RESULTS_DIR,
            "mdp_safety_report.csv"
        ),
        index=False
    )

    pd.DataFrame(decisions[:1000]).to_csv(
        os.path.join(
            RESULTS_DIR,
            "pipeline_decisions.csv"
        ),
        index=False
    )

    return safety_report

def print_day5_summary(
    combined_metrics_v2,
    combined_metrics_v1,
    safety_report,
    run_v1=False,
):

    log.info("\n" + "=" * 80)
    log.info("DAY 5 SUMMARY")
    log.info("=" * 80)

    if combined_metrics_v2:

        log.info("\n  TwinMesh V2")

        for k, v in combined_metrics_v2.items():
            log.info(f"    {k}: {v}")

    if run_v1 and combined_metrics_v1:

        log.info("\n  TwinMesh V1")

        for k, v in combined_metrics_v1.items():
            log.info(f"    {k}: {v}")

    if safety_report:

        log.info("\n  MDP Safety")

        for k, v in safety_report.items():
            log.info(f"    {k}: {v}")

    log.info("\n  Outputs:")

    log.info(
        "    results/combined/twinmesh_v2_metrics.csv"
    )

    log.info(
        "    results/combined/mdp_safety_report.csv"
    )

    log.info(
        "    results/combined/pipeline_decisions.csv"
    )

def main():

    parser = argparse.ArgumentParser(
        description="TwinMesh Day 5"
    )

    parser.add_argument(
        "--v1",
        action="store_true",
        help="Run V1 only"
    )

    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both V1 and V2"
    )

    args = parser.parse_args()

    log.info("=" * 80)
    log.info("T W I N M E S H  ·  DAY 5")
    log.info("Combined Model + MDP Policy")

    if args.v1:
        log.info("Mode: V1")
    elif args.both:
        log.info("Mode: BOTH")
    else:
        log.info("Mode: V2")

    log.info("=" * 80)

    if not verify_day4_complete():
        sys.exit(1)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    t_total = time.time()

    combined_metrics_v2 = {}
    combined_metrics_v1 = {}
    safety_report = {}

    run_v1 = args.v1 or args.both

    if args.v1:

        log.info(
            "\nRunning V1 only...\n"
        )

        _, _, combined_metrics_v1, _ = (
            run_morning_v1()
        )

    elif args.both:

        log.info(
            "\nRunning V2 first...\n"
        )

        _, _, combined_metrics_v2 = (
            run_morning_v2()
        )

        log.info(
            "\nRunning V1 second...\n"
        )

        _, _, combined_metrics_v1, _ = (
            run_morning_v1()
        )

    else:

        log.info(
            "\nRunning V2...\n"
        )

        _, _, combined_metrics_v2 = (
            run_morning_v2()
        )

    safety_report = run_afternoon()

    print_day5_summary(
        combined_metrics_v2,
        combined_metrics_v1,
        safety_report,
        run_v1
    )

    log.info(
        f"\n  Total runtime: "
        f"{time.time()-t_total:.1f}s"
    )

    log.info("\n" + "=" * 80)
    log.info("DAY 5 COMPLETE")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

