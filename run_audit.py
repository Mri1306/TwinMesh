import os
import sys
import argparse
import time

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger
from src.utils.constants import PROCESSED_DIR

log = get_logger("run_day2")

AUDIT_DIR      = os.path.join(PROJECT_ROOT, "results", "audit")
COMBINED_FILE  = os.path.join(PROCESSED_DIR, "all_ports_combined.csv")

EXPECTED_OUTPUTS = {
    "State distribution":    "state_distribution.csv",
    "Red Sea check":         "check1_red_sea_signal.csv",
    "Cross-port lag":        "check2_lag_correlations.csv",
    "Vessel-type signal":    "check3_vessel_type_signal.csv",
    "Era comparison":        "check4_era_comparison.csv",
    "MDP cost matrix":       "mdp_cost_matrix.csv",
    "MDP policy table":      "mdp_policy_table.csv",
    "Audit report":          "audit_report.txt",
}

def verify_day1_complete() -> bool:
    if not os.path.exists(COMBINED_FILE):
        log.error("=" * 60)
        log.error("❌  Day 1 outputs not found.")
        log.error(f"    Missing: {COMBINED_FILE}")
        log.error("    Run Day 1 first:")
        log.error("      python run_preprocessing.py")
        log.error("=" * 60)
        return False

    import pandas as pd
    df = pd.read_csv(COMBINED_FILE, nrows=5, parse_dates=["date"])
    required_cols = [
        "date", "port", "congestion_score",
        "net_dwt_flow", "state", "anomaly_flag",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        log.error(f"❌  all_ports_combined.csv missing columns: {missing}")
        log.error("    Re-run Day 1: python run_preprocessing.py")
        return False

    log.info(f"✅  Day 1 outputs verified: {COMBINED_FILE}")
    return True


def check_day2_outputs() -> bool:
    log.info("\n── Day 2 Output Verification ──")
    all_ok = True
    for label, fname in EXPECTED_OUTPUTS.items():
        path = os.path.join(AUDIT_DIR, fname)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            log.info(f"  ✅  {label:<25} {fname} ({size_kb:.0f} KB)")
        else:
            log.warning(f"  ❌  {label:<25} MISSING: {fname}")
            all_ok = False
    return all_ok

def run_morning() -> dict:
    """
    Runs all 4 data audit checks in sequence.
    Returns dict of results for the audit report.
    """
    import pandas as pd

    log.info("\n" + "=" * 60)
    log.info("MORNING — Data Audit Checks")
    log.info("=" * 60)

    os.makedirs(AUDIT_DIR, exist_ok=True)

    df = pd.read_csv(COMBINED_FILE, parse_dates=["date"])
    log.info(f"  Loaded: {len(df):,} rows | "
             f"{df['date'].min().date()} → {df['date'].max().date()}")

    results = {}

    log.info("\n── State Distribution (pre-check) ──")
    dist = df.groupby("port")["state"].value_counts(normalize=True).unstack(fill_value=0)
    dist.to_csv(os.path.join(AUDIT_DIR, "state_distribution.csv"))
    log.info(f"\n{dist.round(3).to_string()}")
    from src.utils.constants import STATES
    global_dist = df["state"].value_counts(normalize=True).reindex(STATES, fill_value=0)
    log.info(f"\n  Global: {global_dist.round(3).to_dict()}")
    results["state_dist"] = dist

    log.info("\n" + "-" * 50)
    from src.audit.check1_red_sea import run_check1
    results["check1"] = run_check1()

    log.info("\n" + "-" * 50)
    from src.audit.check2_cross_port_lag import run_check2
    results["check2"] = run_check2()
    
    log.info("\n" + "-" * 50)
    from src.audit.check3_vessel_type_signal import run_check3
    results["check3"] = run_check3()

    log.info("\n" + "-" * 50)
    from src.audit.check4_yoy_era import run_check4
    results["check4_stats"], results["check4_pass"] = run_check4()

    return results

def run_afternoon() -> tuple:
    """
    Fits MDP cost matrix from real IMF data.
    Runs value iteration. Verifies Tier-0 safety.
    Returns (cost_matrix, policy_df, safe).
    """
    log.info("\n" + "=" * 60)
    log.info("AFTERNOON — MDP Cost Matrix (fitted from real IMF data)")
    log.info("=" * 60)

    from src.policy.mdp_cost_matrix import run_mdp_cost_matrix
    cost_matrix, policy_df, safe = run_mdp_cost_matrix()
    return cost_matrix, policy_df, safe

def write_audit_report(morning_results: dict, policy_df, safe: bool) -> None:
    """
    Writes results/audit/audit_report.txt.
    This is the file to READ after Day 2.
    Fill the X and Y numbers into your paper from this file.
    """
    import pandas as pd
    from src.utils.constants import STATES

    report_path = os.path.join(AUDIT_DIR, "audit_report.txt")
    lines = []

    lines += [
        "=" * 70,
        "  T W I N M E S H  ·  DAY 2 AUDIT REPORT",
        "  READ THIS FILE. Fill X and Y into your paper.",
        "=" * 70,
        "",
    ]

    lines += ["── STATE DISTRIBUTION ──", ""]
    dist = morning_results.get("state_dist")
    if dist is not None:
        lines.append(dist.round(3).to_string())
    lines.append("")

    lines += ["── CHECK 1: RED SEA CRISIS SIGNAL ──", ""]
    c1 = morning_results.get("check1", {})
    for port, vals in c1.items():
        if isinstance(vals, dict):
            status = "PASS ✅" if vals.get("pass") else "FAIL ❌"
            lines.append(
                f"  {port:<15} spike={vals.get('spike_sigma', 0):.3f}σ  "
                f"crisis_S3={vals.get('crisis_s3_rate', 0):.1%}  {status}"
            )
    lines.append("")

    lines += ["── CHECK 2: CROSS-PORT LAG (KEY FINDING FOR FIGURE 3) ──", ""]
    c2 = morning_results.get("check2")
    if c2 is not None and len(c2):
        key_pairs = [
            ("Shanghai", "Singapore"),
            ("Shanghai", "Rotterdam"),
            ("Singapore", "Rotterdam"),
        ]
        for src, tgt in key_pairs:
            row = c2[(c2["source"] == src) & (c2["target"] == tgt)]
            if len(row):
                r   = row.iloc[0]
                win = "IN WINDOW ✅" if r["in_expected_window"] else "outside window"
                lines.append(
                    f"  {src:<15} → {tgt:<15} "
                    f"peak_lag={int(r['peak_lag_days'])}d  "
                    f"r={r['peak_r']:+.3f}  {win}"
                )
    lines.append("")

    lines += ["── CHECK 3: VESSEL-TYPE SIGNAL (FILL INTO PAPER) ──", ""]
    c3 = morning_results.get("check3", {})
    con_drop = abs(c3.get("_container_drop_pct", 0))
    tan_drop = abs(c3.get("_tanker_drop_pct", 0))
    passed3  = c3.get("_check_passed", False)
    lines += [
        f"  Container DWT drop (S0→S3): {con_drop:.1f}%",
        f"  Tanker DWT drop    (S0→S3): {tan_drop:.1f}%",
        f"  Check result: {'PASS ✅' if passed3 else 'FAIL ❌'}",
        "",
        "  ── COPY THIS SENTENCE INTO PAPER SECTION 7 ──",
        f"  \"Container DWT net flow drops {con_drop:.0f}% during S3 (Disrupted)",
        f"   states while tanker flow changes only {tan_drop:.0f}%,",
        "   confirming vessel-type-specific cascade patterns",
        "   invisible to aggregate models.\"",
        "",
    ]

    lines += ["── CHECK 4: YoY ERA COMPARISON ──", ""]
    c4 = morning_results.get("check4_stats")
    c4_pass = morning_results.get("check4_pass", False)
    if c4 is not None and len(c4):
        pivot = c4.pivot_table(
            index="port", columns="era",
            values="mean_congestion"
        ).round(4)
        lines.append(pivot.to_string())
    lines += ["", f"  3 eras distinct: {'YES ✅' if c4_pass else 'NO ❌'}", ""]

    lines += ["── MDP POLICY TABLE (copy to paper Section 4.4) ──", ""]
    if policy_df is not None and len(policy_df):
        pivot_p = policy_df.pivot(
            index="state", columns="tier", values="optimal_action"
        )
        pivot_p.columns = [f"Tier-{t}" for t in sorted(pivot_p.columns)]
        lines.append(pivot_p.to_string())
    lines += [
        "",
        f"  Tier-0 safety (must = zero violations): "
        f"{'ZERO VIOLATIONS ✅' if safe else '❌ VIOLATIONS FOUND — FIX BEFORE DAY 5'}",
        "",
    ]

    all_pass = (
        all(v.get("pass", False) for v in c1.values() if isinstance(v, dict))
        and passed3
        and c4_pass
        and safe
    )
    lines += [
        "=" * 70,
        f"  OVERALL DAY 2 STATUS: {'✅ ALL CHECKS PASS — proceed to Day 3' if all_pass else '⚠️  SOME CHECKS FAILED — review above before Day 3'}",
        "  NEXT: python src/models/hawkes/hawkes_port_level.py",
        "=" * 70,
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info(f"\n  ✅ Audit report saved → {report_path}")
    log.info("     READ THIS FILE — fill X and Y into your paper")

def main():
    parser = argparse.ArgumentParser(description="TwinMesh Day 2")
    parser.add_argument("--morning",   action="store_true", help="Run checks only")
    parser.add_argument("--afternoon", action="store_true", help="Run MDP only")
    parser.add_argument("--check",     action="store_true", help="Verify outputs exist")
    args = parser.parse_args()

    if args.check:
        ok = check_day2_outputs()
        sys.exit(0 if ok else 1)

    log.info("=" * 60)
    log.info("T W I N M E S H  ·  DAY 2")
    log.info("Data Audit + MDP Cost Matrix")
    log.info("=" * 60)

    if not verify_day1_complete():
        sys.exit(1)

    os.makedirs(AUDIT_DIR, exist_ok=True)
    t_total = time.time()

    run_all_morning   = not args.afternoon
    run_all_afternoon = not args.morning

    morning_results = {}
    policy_df       = None
    safe            = False

    if run_all_morning:
        t = time.time()
        morning_results = run_morning()
        log.info(f"\n  Morning done in {time.time()-t:.1f}s")

    if run_all_afternoon:
        t = time.time()
        _, policy_df, safe = run_afternoon()
        log.info(f"\n  Afternoon done in {time.time()-t:.1f}s")

    if run_all_morning:
        write_audit_report(morning_results, policy_df, safe)

    log.info(f"\n  Total Day 2 time: {time.time()-t_total:.1f}s")
    log.info("\n" + "=" * 60)
    log.info("✅  DAY 2 COMPLETE")
    log.info("   → Read:  results/audit/audit_report.txt")
    log.info("   → Fill X and Y into paper Section 7")
    log.info("   → Fill policy table into paper Section 4.4")
    log.info("   → Next:  python run_day3.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()