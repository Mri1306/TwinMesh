import os
import sys
import time
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger
from src.utils.constants import UNIFIED_DIR

log = get_logger("run_baselines")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "combined")

EXPECTED_OUTPUTS = [
    "baseline_comparison.csv",
    "baseline_auprc_chart.png",
]

def verify_files():
    required = [
        os.path.join(UNIFIED_DIR, "twinmesh_training.csv"),
        os.path.join(UNIFIED_DIR, "twinmesh_test.csv"),
        os.path.join(RESULTS_DIR, "twinmesh_v2_metrics.csv"),
    ]

    missing = [p for p in required if not os.path.exists(p)]

    if missing:
        log.error("Missing required files:\n")
        for m in missing:
            log.error(f"  {m}")
        return False

    log.info("Baseline script verified ✓")
    log.info("Training/test datasets verified ✓")
    log.info("TwinMesh Day 5 metrics verified ✓")

    return True

def verify_outputs():
    ok = True

    log.info("\nGenerated outputs:\n")

    for fname in EXPECTED_OUTPUTS:

        path = os.path.join(RESULTS_DIR, fname)

        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            log.info(f"  ✓ {fname:<35} ({size_kb:.1f} KB)")
        else:
            ok = False
            log.warning(f"  ✗ Missing: {fname}")

    return ok

def main():

    t0 = time.time()

    log.info("=" * 80)
    log.info("T W I N M E S H  ·  BASELINE EXECUTION")
    log.info("Evaluating Day 6 Baselines (Without XGBoost)")
    log.info("=" * 80)

    if not verify_files():
        sys.exit(1)

    baseline_script = os.path.join(
        PROJECT_ROOT,
        "src",
        "baselines",
        "baselines.py"
    )

    if not os.path.exists(baseline_script):
        log.error(f"\nMissing baseline file:\n{baseline_script}")
        sys.exit(1)

    log.info("\nLaunching baselines.py ...\n")

    try:

        subprocess.run(
            [sys.executable, baseline_script],
            check=True
        )

    except subprocess.CalledProcessError as e:

        log.error("\nBaseline execution failed")
        log.error(str(e))
        sys.exit(1)

    runtime = time.time() - t0

    log.info("\n" + "=" * 80)
    log.info("BASELINE EXECUTION COMPLETE")
    log.info("=" * 80)

    log.info(f"\nRuntime: {runtime:.1f}s")

    ok = verify_outputs()

    if ok:
        log.info("\nAll baseline outputs generated successfully ✓")
    else:
        log.warning("\nSome outputs are missing")

    print()

if __name__ == "__main__":
    main()