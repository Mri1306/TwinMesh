import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, average_precision_score
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.utils.constants import (
    PROCESSED_DIR, UNIFIED_DIR, STATES, STATE_MAP, INV_STATE, PORTS,
)
from src.utils.logger import get_logger

log = get_logger("markov_chain")

RESULTS_DIR       = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
    "results", "markov_standalone"
)
TRANSITIONS_FILE  = os.path.join(PROCESSED_DIR, "markov_transitions.csv")
COMBINED_FILE     = os.path.join(PROCESSED_DIR, "all_ports_combined.csv")
TRAIN_FILE        = os.path.join(UNIFIED_DIR,   "twinmesh_training.csv")
TEST_FILE         = os.path.join(UNIFIED_DIR,   "twinmesh_test.csv")

PORT_NAMES = list(PORTS.keys())
N_STATES   = len(STATES)

def build_transition_matrix(
    combined_df: pd.DataFrame,
    transitions_df: pd.DataFrame,
) -> np.ndarray:
    """
    Builds 5x5 empirical transition matrix from real state sequences.

    Two contributions:
      1. State-CHANGE transitions from markov_transitions.csv
         (rows where from_state != to_state)
      2. Self-transitions: days where state does NOT change
         (from all_ports_combined.csv consecutive rows per port)

    Normalises each row to sum to 1.0 (probability distribution).
    """
    T = np.zeros((N_STATES, N_STATES))

    for _, row in transitions_df.iterrows():
        i = STATE_MAP.get(row["from_state"], -1)
        j = STATE_MAP.get(row["to_state"],   -1)
        if 0 <= i < N_STATES and 0 <= j < N_STATES:
            T[i][j] += 1

    for port in PORT_NAMES:
        port_df = combined_df[combined_df["port"] == port].sort_values("date")
        states  = port_df["state"].tolist()
        for k in range(1, len(states)):
            prev = STATE_MAP.get(states[k-1], -1)
            curr = STATE_MAP.get(states[k],   -1)
            if prev == curr and 0 <= prev < N_STATES:
                T[prev][prev] += 1

    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1   # avoid division by zero
    T_norm = T / row_sums

    log.info("\n  Transition matrix (normalised):")
    T_df = pd.DataFrame(
        T_norm.round(3),
        index=[f"{s}-{n}" for s, n in zip(STATES, ["Normal","Delayed","Congested","Disrupted","Recovery"])],
        columns=STATES
    )
    log.info(f"\n{T_df.to_string()}")

    log.info("\n  Key transition probabilities:")
    log.info(f"  S2→S3 (Congested→Disrupted): {T_norm[2][3]:.3f}")
    log.info(f"  S3→S3 (Disrupted stays):     {T_norm[3][3]:.3f}")
    log.info(f"  S3→S4 (Disrupted→Recovery):  {T_norm[3][4]:.3f}")
    log.info(f"  S4→S0 (Recovery→Normal):     {T_norm[4][0]:.3f}")

    return T_norm

def predict_next_state_pure_markov(
    current_state: str,
    T: np.ndarray,
) -> tuple:
    """
    Predicts next state as argmax of transition row.
    Returns (predicted_state_str, probability_vector).
    """
    i    = STATE_MAP.get(current_state, 0)
    probs = T[i]
    pred  = STATES[np.argmax(probs)]
    return pred, probs


def evaluate_pure_markov(
    T: np.ndarray,
    test_df: pd.DataFrame,
) -> dict:
    """
    Evaluates pure Markov on test set.
    Predicts next state from current state using T.
    Computes: accuracy, macro-F1, per-state F1, S3 recall, AUPRC.
    """
    preds   = []
    actuals = []
    probs_s3 = []   

    for port in PORT_NAMES:
        port_df = test_df[test_df["port"] == port].sort_values("date").reset_index(drop=True)
        states  = port_df["state"].tolist()

        for k in range(len(states) - 1):
            curr_state  = states[k]
            true_next   = states[k + 1]
            pred_next, prob_vec = predict_next_state_pure_markov(curr_state, T)

            preds.append(pred_next)
            actuals.append(true_next)
            probs_s3.append(prob_vec[STATE_MAP["S3"]])

    accuracy = accuracy_score(actuals, preds)
    macro_f1 = f1_score(actuals, preds, average="macro",
                        labels=STATES, zero_division=0)
    per_state_f1 = f1_score(actuals, preds, average=None,
                             labels=STATES, zero_division=0)

    s3_true = [1 if a == "S3" else 0 for a in actuals]
    s3_pred = [1 if p == "S3" else 0 for p in preds]
    s3_recall = sum(t == 1 and p == 1 for t, p in zip(s3_true, s3_pred)) \
                / max(sum(s3_true), 1)

    if sum(s3_true) > 0:
        s3_auprc = average_precision_score(s3_true, probs_s3)
    else:
        s3_auprc = 0.0

    metrics = {
        "model":        "Pure_Markov",
        "accuracy":     round(accuracy, 4),
        "macro_f1":     round(macro_f1, 4),
        "s3_recall":    round(s3_recall, 4),
        "s3_auprc":     round(s3_auprc, 4),
        "n_test_steps": len(actuals),
    }
    for i, state in enumerate(STATES):
        metrics[f"f1_{state}"] = round(float(per_state_f1[i]), 4)

    log.info(f"\n  Pure Markov Results:")
    log.info(f"    Accuracy:  {accuracy:.4f}")
    log.info(f"    Macro-F1:  {macro_f1:.4f}")
    log.info(f"    S3 Recall: {s3_recall:.4f}")
    log.info(f"    S3 AUPRC:  {s3_auprc:.4f}")
    log.info(f"    Per-state F1:")
    for i, state in enumerate(STATES):
        log.info(f"      {state}: {per_state_f1[i]:.4f}")

    log.info(f"\n  Classification report:\n"
             f"{classification_report(actuals, preds, labels=STATES, zero_division=0)}")

    return metrics, actuals, preds, probs_s3

def plot_transition_heatmap(T: np.ndarray, suffix: str = "pure") -> None:
    """
    Heatmap of 5x5 transition matrix. Draft of Figure 4.
    Highlights S2→S3 (risk transition) and S3→S4 (recovery).
    """
    state_labels = [
        f"{s}\n({n})"
        for s, n in zip(STATES, ["Normal","Delayed","Congested","Disrupted","Recovery"])
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        T, annot=True, fmt=".3f",
        cmap="Blues", linewidths=0.5, ax=ax,
        annot_kws={"size": 10},
        xticklabels=state_labels,
        yticklabels=state_labels,
        cbar_kws={"label": "Transition probability P(S_t+1 | S_t)"}
    )
    ax.set_title(
        f"Markov Transition Matrix ({suffix})\n"
        "Row = current state  |  Column = next state\n"
        "(Draft of Figure 4)",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Next State (S_t+1)")
    ax.set_ylabel("Current State (S_t)")

    ax.add_patch(plt.Rectangle((3, 2), 1, 1, fill=False,
                                edgecolor="#e74c3c", lw=2.5, label="S2→S3 risk"))
    ax.add_patch(plt.Rectangle((4, 3), 1, 1, fill=False,
                                edgecolor="#27ae60", lw=2.5, label="S3→S4 recovery"))

    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, f"markov_transition_heatmap_{suffix}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Transition heatmap saved -> {out_png}")


def plot_confusion_matrix(actuals: list, preds: list, suffix: str = "pure") -> None:
    """Confusion matrix for state prediction."""
    cm     = confusion_matrix(actuals, preds, labels=STATES)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=STATES, yticklabels=STATES,
                ax=axes[0], annot_kws={"size": 10})
    axes[0].set_title("Confusion Matrix (counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=STATES, yticklabels=STATES,
                ax=axes[1], annot_kws={"size": 10})
    axes[1].set_title("Confusion Matrix (normalised)", fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.suptitle(f"Markov State Prediction ({suffix})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, f"markov_confusion_matrix_{suffix}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Confusion matrix saved -> {out_png}")


def plot_state_timeline(test_df: pd.DataFrame, actuals: list, preds: list) -> None:
    """
    Timeline plot: actual vs predicted state for one port.
    Shows where Markov gets it right and wrong.
    """
    port    = "Singapore"
    port_df = test_df[test_df["port"] == port].sort_values("date").reset_index(drop=True)
    n       = min(len(port_df) - 1, len(actuals) // len(list(PORTS.keys())))

    port_idx_start = list(PORTS.keys()).index(port)
    step           = len(PORTS)
    port_actuals   = actuals[port_idx_start::step][:n]
    port_preds     = preds[port_idx_start::step][:n]

    dates = port_df["date"].iloc[1:n+1].values
    act_num = [STATE_MAP.get(s, 0) for s in port_actuals]
    pred_num= [STATE_MAP.get(s, 0) for s in port_preds]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    colors = ["#2ecc71","#f1c40f","#e67e22","#e74c3c","#3498db"]

    for ax, values, label in zip(axes, [act_num, pred_num], ["Actual", "Predicted"]):
        for i, (d, v) in enumerate(zip(dates, values)):
            ax.bar(i, 1, color=colors[v], alpha=0.8, width=0.9)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel(label)
        ax.set_yticks([])

    axes[0].set_title(
        f"Singapore — Actual vs Predicted States (Pure Markov)\n"
        "Green=S0  Yellow=S1  Orange=S2  Red=S3  Blue=S4",
        fontsize=10, fontweight="bold"
    )
    axes[-1].set_xlabel("Test days (chronological)")
    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "markov_state_timeline.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  State timeline saved -> {out_png}")

def run_markov_chain() -> tuple:
    """
    Full Model A pipeline — pure Markov.
    Returns (T, metrics, actuals, preds).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("DAY 4 MORNING — Markov Model A: Pure 5-State Chain")
    log.info("=" * 60)

    for path in [TRANSITIONS_FILE, COMBINED_FILE, TRAIN_FILE, TEST_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                "Run Day 1 first: python run_preprocessing.py"
            )

    transitions = pd.read_csv(TRANSITIONS_FILE)
    combined    = pd.read_csv(COMBINED_FILE,  parse_dates=["date"])
    train       = pd.read_csv(TRAIN_FILE,     parse_dates=["date"])
    test        = pd.read_csv(TEST_FILE,      parse_dates=["date"])

    log.info(
        f"  Loaded | transitions={len(transitions):,} | "
        f"train={len(train):,} | test={len(test):,}"
    )

    log.info("\n  Building transition matrix (training data)...")
    T = build_transition_matrix(train, transitions[
        transitions["timestamp"] <= str(train["date"].max().date())
    ] if "timestamp" in transitions.columns else transitions)

    T_df = pd.DataFrame(T.round(4), index=STATES, columns=STATES)
    T_df.to_csv(os.path.join(RESULTS_DIR, "transition_matrix.csv"))
    log.info(f"  Saved transition_matrix.csv")

    log.info("\n  Evaluating on test set...")
    log.info("=" * 60)
    log.info("MARKOV MODEL A — Pure Markov Results")
    log.info("=" * 60)
    metrics, actuals, preds, probs_s3 = evaluate_pure_markov(T, test)

    pd.DataFrame([metrics]).to_csv(
        os.path.join(RESULTS_DIR, "markov_pure_metrics.csv"), index=False
    )

    ablation_row = {
        "row":             3,
        "model_variant":   "+ Pure Markov",
        "description":     "5-state transition matrix from real data",
        "auprc":           metrics["s3_auprc"],
        "poisson_auprc":   None,
        "lift_vs_poisson": None,
        "mean_lead_days":  None,
        "detection_rate":  None,
        "state_accuracy":  metrics["accuracy"],
        "macro_f1":        metrics["macro_f1"],
        "s3_recall":       metrics["s3_recall"],
    }
    pd.DataFrame([ablation_row]).to_csv(
        os.path.join(RESULTS_DIR, "ablation_row3_pure_markov.csv"), index=False
    )
    log.info(
        f"\n  ABLATION ROW 3 (Pure Markov):\n"
        f"    Accuracy={metrics['accuracy']:.4f}  "
        f"Macro-F1={metrics['macro_f1']:.4f}  "
        f"S3-Recall={metrics['s3_recall']:.4f}  "
        f"S3-AUPRC={metrics['s3_auprc']:.4f}"
    )

    plot_transition_heatmap(T, suffix="pure_markov")
    plot_confusion_matrix(actuals, preds, suffix="pure_markov")
    plot_state_timeline(test, actuals, preds)

    log.info("\n  Model A (Pure Markov) complete.")
    return T, metrics, actuals, preds


if __name__ == "__main__":
    run_markov_chain()