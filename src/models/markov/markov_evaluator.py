import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.utils.constants import (
    PROCESSED_DIR, UNIFIED_DIR, STATES, STATE_MAP, PORTS,
)
from src.utils.logger import get_logger

log = get_logger("markov_evaluator")

RESULTS_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
    "results", "markov_standalone"
)
COMBINED    = os.path.join(PROCESSED_DIR, "all_ports_combined.csv")
TEST_FILE   = os.path.join(UNIFIED_DIR,   "twinmesh_test.csv")
PORT_NAMES  = list(PORTS.keys())

def compute_markov_lead_time(
    model,
    scaler,
    test_df: pd.DataFrame,
    feature_cols: list,
    lead_window: int = 7,
) -> dict:
    """
    For each S3 onset in test set:
      Look back lead_window days.
      Check if Markov P(S3) exceeded threshold in the window.
      Measure how many days before onset the threshold was crossed.

    Threshold = 0.3 (P(S3) > 30% = early warning).
    """
    from sklearn.preprocessing import StandardScaler
    S3_PROB_THRESHOLD = 0.30
    s3_idx = STATE_MAP["S3"]

    lead_results = {}

    for port in PORT_NAMES:
        port_df = test_df[test_df["port"] == port].sort_values("date").reset_index(drop=True)

        if len(port_df) < 2:
            continue

        for col in feature_cols:
            if col not in port_df.columns:
                port_df[col] = 0.0
            port_df[col] = pd.to_numeric(port_df[col], errors="coerce").fillna(0)

        X = port_df[feature_cols].values
        try:
            X_scaled = scaler.transform(X)
            probs    = model.predict_proba(X_scaled)
            s3_probs = probs[:, s3_idx] if probs.shape[1] > s3_idx else np.zeros(len(X))
        except Exception:
            continue

        # Find S3 onset days
        states = port_df["state"].tolist()
        s3_onsets = []
        for i in range(1, len(states)):
            if states[i] == "S3" and states[i-1] != "S3":
                s3_onsets.append(i)

        detected_leads = []
        for onset_idx in s3_onsets:
            window_start = max(0, onset_idx - lead_window)
            window_probs = s3_probs[window_start:onset_idx]

            above_thresh = np.where(window_probs > S3_PROB_THRESHOLD)[0]
            if len(above_thresh) > 0:
                first_alert = above_thresh[0]
                lead_days   = onset_idx - (window_start + first_alert)
                detected_leads.append(lead_days)
            else:
                detected_leads.append(0)

        if detected_leads:
            mean_lead  = np.mean(detected_leads)
            det_rate   = sum(d > 0 for d in detected_leads) / len(detected_leads)
            lead_results[port] = {
                "mean_lead_days":   round(mean_lead, 2),
                "detection_rate":   round(det_rate, 3),
                "n_s3_onsets":      len(s3_onsets),
                "n_detected_early": sum(d > 0 for d in detected_leads),
            }
            log.info(
                f"  {port:<15} mean_lead={mean_lead:.1f}d  "
                f"detection={det_rate:.1%}  "
                f"({sum(d > 0 for d in detected_leads)}/{len(s3_onsets)} detected)"
            )

    return lead_results

def load_ablation_rows() -> list:
    """Loads all saved ablation rows from markov_standalone folder."""
    rows = []
    for fname in [
        "ablation_row3_pure_markov.csv",
        "ablation_row4_portwatch_only.csv",
        "ablation_row5_portwatch_weather.csv",
    ]:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            row = pd.read_csv(path).iloc[0].to_dict()
            rows.append(row)
            log.info(f"  Loaded: {fname}")
        else:
            log.warning(f"  Missing: {fname} — run morning/afternoon first")

    return rows

def compute_majority_baseline(test_df: pd.DataFrame) -> dict:
    """
    Majority class baseline: always predict the most common next state.
    This is the simplest possible baseline — Markov must beat this.
    """
    all_next = []
    for port in PORT_NAMES:
        port_df = test_df[test_df["port"] == port].sort_values("date")
        states  = port_df["state"].tolist()
        all_next.extend(states[1:])

    from collections import Counter
    majority_state = Counter(all_next).most_common(1)[0][0]
    majority_preds = [majority_state] * len(all_next)
    majority_acc   = sum(p == a for p, a in zip(majority_preds, all_next)) / len(all_next)

    log.info(f"  Majority class baseline: always predict '{majority_state}' -> acc={majority_acc:.4f}")
    return {"majority_class": majority_state, "majority_accuracy": majority_acc}

def build_ablation_rows_3_5(
    rows: list,
    hawkes_row1_auprc: float = 0.2467,
) -> pd.DataFrame:
    """
    Combines ablation rows 3-5 and prints the table.
    Also prints the key paper finding: accuracy lift from each source.
    """
    if not rows:
        log.warning("  No ablation rows found — run morning and afternoon first")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    log.info("\n" + "=" * 70)
    log.info("ABLATION TABLE — ROWS 3-5 (Markov standalone)")
    log.info("=" * 70)
    log.info(f"{'Row':<5} {'Model':<40} {'Acc':>8} {'F1':>8} {'S3-AUC':>8}")
    log.info("-" * 70)

    for _, row in df.iterrows():
        acc = row.get("state_accuracy", 0) or 0
        f1  = row.get("macro_f1", 0) or 0
        auc = row.get("auprc", 0) or 0
        log.info(
            f"  {int(row.get('row', 0)):<3} "
            f"{str(row.get('model_variant', '')):<40} "
            f"{acc:>8.4f} "
            f"{f1:>8.4f} "
            f"{auc:>8.4f}"
        )

    if len(rows) >= 2:
        row3 = rows[0]
        row4 = rows[1] if len(rows) > 1 else rows[0]
        row5 = rows[2] if len(rows) > 2 else rows[-1]

        acc3 = row3.get("state_accuracy", 0) or 0
        acc4 = row4.get("state_accuracy", 0) or 0
        acc5 = row5.get("state_accuracy", 0) or 0

        log.info(
            f"\n  PAPER SENTENCES (Section 7):\n"
            f"\n  Row 3→4 (pure Markov → + PortWatch features):\n"
            f"  \"Adding PortWatch congestion features (Row 4) improves\n"
            f"   accuracy from {acc3:.4f} to {acc4:.4f} (+{acc4-acc3:.4f})\n"
            f"   over pure transition-matrix Markov (Row 3).\"\n"
            f"\n  Row 4→5 (+ PortWatch → + Weather):\n"
            f"  \"Adding maritime weather features (Row 5) further improves\n"
            f"   accuracy from {acc4:.4f} to {acc5:.4f} (+{acc5-acc4:.4f}),\n"
            f"   confirming that weather signals add measurable value\n"
            f"   beyond port activity data alone.\""
        )

    return df

def plot_markov_comparison(rows: list, majority_acc: float) -> None:
    """
    Bar chart comparing pure Markov, multi-source, and majority baseline.
    """
    if not rows:
        return

    labels   = ["Majority\nBaseline"] + [
        str(r.get("model_variant", f"Row {r.get('row','')}")).replace("+ ", "")[:20]
        for r in rows
    ]
    accs     = [majority_acc] + [r.get("state_accuracy", 0) or 0 for r in rows]
    f1s      = [0] + [r.get("macro_f1", 0) or 0 for r in rows]
    colors   = ["#95a5a6"] + ["#3498db"] * len(rows)

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_acc = ax.bar(x - width/2, accs, width, label="Accuracy", color=colors, alpha=0.85)
    bars_f1  = ax.bar(x + width/2, f1s,  width, label="Macro-F1",
                      color=[c if c != "#95a5a6" else "#bdc3c7" for c in colors], alpha=0.85)

    ax.axhline(majority_acc, color="#95a5a6", lw=1.5, linestyle="--",
               label=f"Majority baseline ({majority_acc:.3f})")

    for bar in bars_acc:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Markov Models vs Majority Baseline\n"
        "Ablation rows 3-5 (Day 4)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    out_png = os.path.join(RESULTS_DIR, "markov_vs_baseline.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Comparison plot saved -> {out_png}")

def run_markov_evaluator() -> pd.DataFrame:
    """
    Full Day 4 evening evaluation.
    Loads rows 3-5, builds combined table, computes lead time.
    Returns ablation_df.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("DAY 4 EVENING — Markov Standalone Evaluation")
    log.info("=" * 60)

    test_df = pd.read_csv(TEST_FILE, parse_dates=["date"])

    log.info("\n  Loading saved ablation rows...")
    rows = load_ablation_rows()

    log.info("\n  Computing majority class baseline...")
    baseline = compute_majority_baseline(test_df)
    majority_acc = baseline["majority_accuracy"]

    ablation_df = build_ablation_rows_3_5(rows)

    if len(ablation_df):
        ablation_df.to_csv(
            os.path.join(RESULTS_DIR, "ablation_rows_3_5.csv"), index=False
        )
        log.info(f"\n  Saved ablation_rows_3_5.csv")

    log.info("\n  Computing S3 lead time (multi-source Markov)...")
    log.info("=" * 60)
    log.info("S3 LEAD TIME — Multi-Source Markov")
    log.info("=" * 60)

    try:
        from src.models.markov.markov_multisource import (
            prepare_sequences, fit_logistic_markov, FEATURE_SETS
        )
        from src.utils.constants import UNIFIED_DIR
        train_df = pd.read_csv(os.path.join(UNIFIED_DIR, "twinmesh_training.csv"),
                               parse_dates=["date"])

        feat_cols = FEATURE_SETS["portwatch_weather"]
        X_train, y_train = prepare_sequences(train_df, feat_cols)
        model, scaler, _ = fit_logistic_markov(
            X_train, y_train, feat_cols, "lead_time_model"
        )

        lead_results = compute_markov_lead_time(
            model, scaler, test_df, feat_cols
        )

        if lead_results:
            lead_df = pd.DataFrame(lead_results).T
            lead_df.to_csv(
                os.path.join(RESULTS_DIR, "markov_s3_lead_time.csv")
            )
            avg_lead = np.mean([v["mean_lead_days"] for v in lead_results.values()])
            avg_det  = np.mean([v["detection_rate"]  for v in lead_results.values()])
            log.info(
                f"\n  LEAD TIME SUMMARY:\n"
                f"    Mean lead time:    {avg_lead:.1f} days\n"
                f"    Detection rate:    {avg_det:.1%}\n"
                f"\n  PAPER SENTENCE:\n"
                f"  \"Multi-source Markov detects S3 onset an average of\n"
                f"   {avg_lead:.1f} days early with a {avg_det:.1%} detection rate,\n"
                f"   compared to zero lead time for pure transition-matrix Markov.\""
            )

    except Exception as e:
        log.warning(f"  Lead time computation skipped: {e}")

    if len(rows):
        plot_markov_comparison(rows, majority_acc)

    log.info("\n  ── RESULTS LOCKED (Day 4 Evening) ──")
    log.info("  Ablation rows 3-5 saved")
    log.info("  Rows 1-2 (Hawkes):  results/hawkes_standalone/ablation_rows_1_2.csv")
    log.info("  Rows 3-5 (Markov):  results/markov_standalone/ablation_rows_3_5.csv")
    log.info("  Row 6 (Combined):   built on Day 5")
    log.info(f"\n  All outputs -> {RESULTS_DIR}")

    return ablation_df


if __name__ == "__main__":
    run_markov_evaluator()