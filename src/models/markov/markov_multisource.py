import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, average_precision_score,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.utils.constants import (
    PROCESSED_DIR, UNIFIED_DIR, STATES, STATE_MAP, PORTS,
)
from src.utils.logger import get_logger

log = get_logger("markov_multisource")

RESULTS_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
    "results", "markov_standalone"
)
TRAIN_FILE  = os.path.join(UNIFIED_DIR,   "twinmesh_training.csv")
TEST_FILE   = os.path.join(UNIFIED_DIR,   "twinmesh_test.csv")

PORT_NAMES  = list(PORTS.keys())

FEATURE_SETS = {
    "portwatch_only": [
        "congestion_score",
        "net_dwt_flow",
        "container_ratio",
        "queue_buildup_7d",
        "yoy_pct_change",
        "anomaly_flag",
    ],
    "portwatch_weather": [
        "congestion_score",
        "net_dwt_flow",
        "container_ratio",
        "queue_buildup_7d",
        "yoy_pct_change",
        "anomaly_flag",
        "weather_risk_score",
        "storm_alert",
        "heavy_rain_alert",
    ],
}

def prepare_sequences(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Builds (X, y) for next-state prediction.
    X[t] = feature vector at time t
    y[t] = state at time t+1

    Keeps temporal order strictly — no shuffling.
    Per-port sequences to avoid cross-port contamination.
    """
    X_list = []
    y_list = []

    for port in PORT_NAMES:
        port_df = df[df["port"] == port].sort_values("date").reset_index(drop=True)

        for col in feature_cols:
            if col not in port_df.columns:
                port_df[col] = 0.0
            port_df[col] = pd.to_numeric(port_df[col], errors="coerce").fillna(0)

        if len(port_df) < 2:
            continue

        X = port_df[feature_cols].values[:-1]     # features at t
        y = port_df["state"].values[1:]            # state at t+1

        # Map state strings to integers
        y_int = np.array([STATE_MAP.get(s, 0) for s in y])

        X_list.append(X)
        y_list.append(y_int)

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    return X_all, y_all

def fit_logistic_markov(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list,
    model_name: str,
) -> tuple:
    """
    Fits multinomial logistic regression as the Markov observation model.
    Returns (model, scaler).
    """
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        random_state=42,
    )
    model.fit(X_scaled, y_train)

    train_acc = model.score(X_scaled, y_train)
    log.info(f"  {model_name}: train_accuracy={train_acc:.4f}")

    # Feature importance (coefficient magnitude)
    coef_abs = np.abs(model.coef_).mean(axis=0)
    importance = pd.Series(coef_abs, index=feature_cols).sort_values(ascending=False)
    log.info(f"  Feature importance (mean |coef|):")
    for feat, imp in importance.head(5).items():
        log.info(f"    {feat:<30} {imp:.4f}")

    return model, scaler, importance

def evaluate_multisource(
    model,
    scaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict:
    """
    Evaluates multi-source Markov on test set.
    Returns metrics dict for ablation table.
    """
    X_scaled = scaler.transform(X_test)
    preds    = model.predict(X_scaled)
    probs    = model.predict_proba(X_scaled)

    pred_labels   = [STATES[p] if 0 <= p < len(STATES) else "S0" for p in preds]
    actual_labels = [STATES[y] if 0 <= y < len(STATES) else "S0" for y in y_test]

    accuracy  = accuracy_score(y_test, preds)
    macro_f1  = f1_score(y_test, preds, average="macro",
                         labels=list(range(len(STATES))), zero_division=0)
    per_state = f1_score(y_test, preds, average=None,
                         labels=list(range(len(STATES))), zero_division=0)

    s3_idx    = STATE_MAP["S3"]
    s3_true   = (y_test == s3_idx).astype(int)
    s3_probs  = probs[:, s3_idx] if probs.shape[1] > s3_idx else np.zeros(len(y_test))
    s3_recall = float(s3_true[preds == s3_idx].sum()) / max(s3_true.sum(), 1)
    s3_auprc  = average_precision_score(s3_true, s3_probs) if s3_true.sum() > 0 else 0.0

    brier_s3  = brier_score_loss(s3_true, s3_probs)

    metrics = {
        "model":        model_name,
        "accuracy":     round(accuracy,  4),
        "macro_f1":     round(macro_f1,  4),
        "s3_recall":    round(s3_recall, 4),
        "s3_auprc":     round(s3_auprc,  4),
        "brier_s3":     round(brier_s3,  4),
        "n_test_steps": len(y_test),
    }
    for i, state in enumerate(STATES):
        if i < len(per_state):
            metrics[f"f1_{state}"] = round(float(per_state[i]), 4)

    log.info(f"\n  {model_name} Results:")
    log.info(f"    Accuracy:  {accuracy:.4f}")
    log.info(f"    Macro-F1:  {macro_f1:.4f}")
    log.info(f"    S3 Recall: {s3_recall:.4f}")
    log.info(f"    S3 AUPRC:  {s3_auprc:.4f}")
    log.info(f"    Brier-S3:  {brier_s3:.4f}")
    log.info(f"    Per-state F1:")
    for i, state in enumerate(STATES):
        if i < len(per_state):
            log.info(f"      {state}: {per_state[i]:.4f}")

    return metrics, probs, pred_labels, actual_labels

def plot_feature_importance(importances: dict) -> None:
    """Bar chart of feature importance across both feature sets."""
    if not importances:
        return

    fig, axes = plt.subplots(1, len(importances), figsize=(7 * len(importances), 5))
    if len(importances) == 1:
        axes = [axes]

    for ax, (name, imp) in zip(axes, importances.items()):
        imp_sorted = imp.sort_values(ascending=True)
        colors     = ["#e74c3c" if i >= len(imp_sorted) - 3 else "#3498db"
                      for i in range(len(imp_sorted))]
        ax.barh(imp_sorted.index, imp_sorted.values, color=colors, alpha=0.85)
        ax.set_title(f"Feature Importance\n{name}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean |coefficient|")
        ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Multi-Source Markov — Feature Importance", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Feature importance plot saved -> {out_png}")


def plot_calibration(probs_dict: dict, y_test: np.ndarray) -> None:
    """
    Calibration curve for S3 probability across models.
    Well-calibrated model has diagonal plot.
    """
    s3_idx  = STATE_MAP["S3"]
    s3_true = (y_test == s3_idx).astype(int)

    if s3_true.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for (name, probs), color in zip(probs_dict.items(), colors):
        if probs is None:
            continue
        s3_probs = probs[:, s3_idx] if probs.shape[1] > s3_idx else None
        if s3_probs is None:
            continue
        try:
            frac_pos, mean_pred = calibration_curve(s3_true, s3_probs, n_bins=8)
            ax.plot(mean_pred, frac_pos, "s-", color=color,
                    label=name.replace("_", " ").title(), lw=2)
        except Exception:
            pass

    ax.set_xlabel("Mean predicted probability (S3)")
    ax.set_ylabel("Fraction of actual S3 events")
    ax.set_title(
        "Calibration Curve — S3 (Disrupted) State\n"
        "Diagonal = perfect calibration",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "calibration_plot.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Calibration plot saved -> {out_png}")


def plot_accuracy_progression(metrics_list: list) -> None:
    """
    Bar chart showing accuracy progression as each source is added.
    This is the core ablation story for the paper.
    """
    names  = [m["model"].replace("_", " ") for m in metrics_list]
    accs   = [m["accuracy"] for m in metrics_list]
    f1s    = [m["macro_f1"] for m in metrics_list]
    s3aupr = [m["s3_auprc"] for m in metrics_list]

    x     = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, accs,   width, label="Accuracy",   color="#3498db", alpha=0.85)
    ax.bar(x,          f1s,   width, label="Macro-F1",   color="#e74c3c", alpha=0.85)
    ax.bar(x + width, s3aupr, width, label="S3 AUPRC",   color="#2ecc71", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Multi-Source Markov — Accuracy Progression\n"
        "Each bar group = one ablation row in paper Table 1",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    for i, (acc, f1, s3) in enumerate(zip(accs, f1s, s3aupr)):
        ax.text(i - width, acc + 0.01, f"{acc:.3f}", ha="center", fontsize=8)
        ax.text(i,          f1 + 0.01, f"{f1:.3f}",  ha="center", fontsize=8)
        ax.text(i + width, s3 + 0.01, f"{s3:.3f}",  ha="center", fontsize=8)

    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "accuracy_progression.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Accuracy progression plot saved -> {out_png}")

def run_markov_multisource() -> tuple:
    """
    Full Model B pipeline — multi-source Markov.
    Fits models with increasing feature sets.
    Returns (all_metrics, all_models).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("DAY 4 AFTERNOON — Markov Model B: Multi-Source")
    log.info("=" * 60)

    train = pd.read_csv(TRAIN_FILE, parse_dates=["date"])
    test  = pd.read_csv(TEST_FILE,  parse_dates=["date"])

    all_metrics    = []
    all_models     = {}
    importances    = {}
    probs_dict     = {}
    ablation_rows  = []
    y_test_ref     = None

    for row_num, (feat_name, feat_cols) in enumerate(FEATURE_SETS.items(), start=4):
        log.info(f"\n{'='*60}")
        log.info(f"Feature set: {feat_name} ({len(feat_cols)} features)")
        log.info(f"Features: {feat_cols}")

        X_train, y_train = prepare_sequences(train, feat_cols)
        X_test,  y_test  = prepare_sequences(test,  feat_cols)
        if y_test_ref is None:
            y_test_ref = y_test

        log.info(f"  Train: {len(X_train):,} steps  |  Test: {len(X_test):,} steps")

        model, scaler, importance = fit_logistic_markov(
            X_train, y_train, feat_cols, feat_name
        )
        all_models[feat_name] = (model, scaler)
        importances[feat_name] = importance

        metrics, probs, pred_labels, actual_labels = evaluate_multisource(
            model, scaler, X_test, y_test, feat_name
        )
        all_metrics.append(metrics)
        probs_dict[feat_name] = probs

        abl_row = {
            "row":             row_num,
            "model_variant":   f"+ {feat_name.replace('_', ' ').title()}",
            "description":     f"LogReg on {len(feat_cols)} features",
            "auprc":           metrics["s3_auprc"],
            "poisson_auprc":   None,
            "lift_vs_poisson": None,
            "mean_lead_days":  None,
            "detection_rate":  None,
            "state_accuracy":  metrics["accuracy"],
            "macro_f1":        metrics["macro_f1"],
            "s3_recall":       metrics["s3_recall"],
        }
        ablation_rows.append(abl_row)

        pd.DataFrame([abl_row]).to_csv(
            os.path.join(RESULTS_DIR, f"ablation_row{row_num}_{feat_name}.csv"),
            index=False
        )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(
        os.path.join(RESULTS_DIR, "markov_multisource_metrics.csv"), index=False
    )

    log.info("\n" + "=" * 60)
    log.info("ABLATION SUMMARY — Rows 4-5 (Multi-Source Markov)")
    log.info("=" * 60)
    log.info(f"{'Row':<5} {'Model':<35} {'Acc':>8} {'F1':>8} {'S3-AUC':>8}")
    log.info("-" * 60)
    for row in ablation_rows:
        log.info(
            f"  {row['row']:<3} {row['model_variant']:<35} "
            f"{row['state_accuracy']:>8.4f} "
            f"{row['macro_f1']:>8.4f} "
            f"{row['auprc']:>8.4f}"
        )

    if len(all_metrics) >= 2:
        acc_pw  = all_metrics[0]["accuracy"]
        acc_wx  = all_metrics[1]["accuracy"]
        lift_wx = acc_wx - acc_pw
        log.info(
            f"\n  PAPER SENTENCE (Section 7):\n"
            f"  \"Adding maritime weather features (Row 5) improves\n"
            f"   state prediction accuracy from {acc_pw:.4f} to {acc_wx:.4f}\n"
            f"   (+{lift_wx:.4f}), demonstrating that weather signals\n"
            f"   add measurable predictive value beyond port activity alone.\""
        )

    plot_feature_importance(importances)
    if y_test_ref is not None:
        plot_calibration(probs_dict, y_test_ref)
    plot_accuracy_progression(all_metrics)

    log.info("\n  Model B (Multi-Source Markov) complete.")
    return all_metrics, all_models


if __name__ == "__main__":
    run_markov_multisource()