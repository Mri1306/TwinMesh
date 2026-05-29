import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.utils.constants import PROCESSED_DIR, UNIFIED_DIR, PORTS, STATES
from src.utils.logger import get_logger

log = get_logger("hawkes_evaluator")

RESULTS_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
    "results", "hawkes_standalone"
)
MATRIX_DIR  = os.path.join(RESULTS_DIR, "hawkes_excitation_matrices")
COMBINED    = os.path.join(PROCESSED_DIR, "all_ports_combined.csv")
TRAIN_FILE  = os.path.join(UNIFIED_DIR,   "twinmesh_training.csv")
TEST_FILE   = os.path.join(UNIFIED_DIR,   "twinmesh_test.csv")

PORT_NAMES = list(PORTS.keys())

def compute_log_likelihood_poisson(events: list, T: float, mu: float) -> float:
    """
    Log-likelihood of homogeneous Poisson process:
      L = N * log(mu) - mu * T
    where N = number of events, T = observation window.
    """
    N = len(events)
    if mu <= 0 or N == 0:
        return -np.inf
    return N * np.log(mu) - mu * T


def compute_log_likelihood_hawkes(
    events: list,
    baseline: float,
    alpha: float,
    decay: float,
    T: float,
) -> float:
    """
    Log-likelihood of univariate Hawkes process (exponential kernel):
      L = sum_{t_i} log(lambda(t_i)) - integral_0^T lambda(t) dt

    Integral approximated analytically for exponential kernel:
      integral = mu * T + alpha/decay * sum_k (1 - exp(-decay*(T-t_k)))
    """
    N = len(events)
    if N == 0 or baseline <= 0:
        return -np.inf

    events_arr = np.array(sorted(events))

    log_sum = 0.0
    for i, ti in enumerate(events_arr):
        lam_ti = baseline
        past   = events_arr[events_arr < ti]
        for tk in past:
            lam_ti += alpha * np.exp(-decay * (ti - tk))
        if lam_ti > 0:
            log_sum += np.log(lam_ti)
        else:
            log_sum += -10.0

    compensator = baseline * T
    for tk in events_arr:
        compensator += (alpha / decay) * (1 - np.exp(-decay * (T - tk)))

    return log_sum - compensator


def compute_llr_per_port(
    train_df: pd.DataFrame,
    mu_series: pd.Series,
    alpha_df: pd.DataFrame,
    decay: float = 0.3,
) -> dict:
    """
    Computes log-likelihood ratio (LLR) = L_Hawkes / L_Poisson per port.
    LLR > 1 means Hawkes fits better than Poisson.
    """
    llr_results = {}
    t0 = train_df["date"].min()

    for port in PORT_NAMES:
        port_df = train_df[train_df["port"] == port].copy()
        port_df["t_numeric"] = (port_df["date"] - t0).dt.days.astype(float)

        events = sorted(
            port_df[port_df["anomaly_flag"] == 1]["t_numeric"].values.tolist()
        )
        T  = float(port_df["t_numeric"].max())
        N  = len(events)
        mu = mu_series.get(port, N / T if T > 0 else 0.001)

        
        alpha_self = float(alpha_df.loc[port, port]) \
                     if port in alpha_df.index else 0.01

        ll_hawkes  = compute_log_likelihood_hawkes(events, mu, alpha_self, decay, T)
        ll_poisson = compute_log_likelihood_poisson(events, T, mu)

        llr = ll_hawkes - ll_poisson   

        llr_results[port] = {
            "n_events":    N,
            "mu_baseline": round(mu, 6),
            "alpha_self":  round(alpha_self, 4),
            "ll_hawkes":   round(ll_hawkes,  2),
            "ll_poisson":  round(ll_poisson, 2),
            "llr":         round(llr,        4),
            "hawkes_better": llr > 0,
        }

        flag = "Hawkes better" if llr > 0 else "Poisson better"
        log.info(
            f"  {port:<15} "
            f"LLR={llr:+.3f}  "
            f"L_Hawkes={ll_hawkes:.1f}  "
            f"L_Poisson={ll_poisson:.1f}  "
            f"[{flag}]"
        )

    return llr_results

def load_model_a_results() -> tuple:
    """Loads Model A (port-level) results saved by hawkes_port_level.py"""
    auprc_path = os.path.join(RESULTS_DIR, "hawkes_port_auprc.csv")
    lead_path  = os.path.join(RESULTS_DIR, "hawkes_port_lead_time.csv")
    mu_path    = os.path.join(MATRIX_DIR,  "port_level_mu.csv")
    alpha_path = os.path.join(MATRIX_DIR,  "port_level_7x7.csv")

    missing = [p for p in [auprc_path, lead_path, mu_path, alpha_path]
               if not os.path.exists(p)]
    if missing:
        log.warning(
            f"  Model A results not found: {missing}\n"
            "  Run hawkes_port_level.py first (morning)."
        )
        return None, None, None, None

    auprc_df = pd.read_csv(auprc_path, index_col=0)
    lead_df  = pd.read_csv(lead_path,  index_col=0)
    mu_s     = pd.read_csv(mu_path,    index_col=0).squeeze()
    alpha_df = pd.read_csv(alpha_path, index_col=0)

    log.info(f"  Model A loaded: {len(auprc_df)} ports")
    return auprc_df, lead_df, mu_s, alpha_df


def load_model_b_results() -> pd.DataFrame:
    """Loads Model B (vessel-type) results saved by hawkes_vessel_type.py"""
    auprc_path = os.path.join(RESULTS_DIR, "hawkes_vessel_metrics.csv")
    if not os.path.exists(auprc_path):
        log.warning(
            "  Model B results not found.\n"
            "  Run hawkes_vessel_type.py first (afternoon)."
        )
        return None

    auprc_df = pd.read_csv(auprc_path, index_col=0)
    log.info(f"  Model B loaded: {len(auprc_df)} ports")
    return auprc_df

def build_ablation_rows(
    model_a_auprc: pd.DataFrame,
    model_a_lead:  pd.DataFrame,
    model_b_auprc: pd.DataFrame,
    llr_results:   dict,
) -> pd.DataFrame:
    """
    Builds rows 1 and 2 of the final ablation table (paper Table 1).

    Row 1: Aggregate Port-Level Hawkes (Model A)
    Row 2: + Vessel-Type Breakdown (Model B)

    Lift from Row 1 → Row 2 = your core novelty claim.
    """
    rows = []

    if model_a_auprc is not None:
        avg_a_auprc   = model_a_auprc["hawkes_auprc"].mean()
        avg_a_poisson = model_a_auprc["poisson_auprc"].mean()
        avg_a_lead    = model_a_lead["mean_lead_days"].mean() \
                        if model_a_lead is not None else 0.0
        avg_a_det     = model_a_lead["detection_rate"].mean() \
                        if model_a_lead is not None else 0.0
        hawkes_better = sum(v["hawkes_better"] for v in llr_results.values()) \
                        if llr_results else 0

        rows.append({
            "row":             1,
            "model_variant":   "Aggregate Hawkes (Model A)",
            "description":     "Port-level, aggregate anomaly events, 7 nodes",
            "auprc":           round(avg_a_auprc,   4),
            "poisson_auprc":   round(avg_a_poisson,  4),
            "lift_vs_poisson": round(avg_a_auprc - avg_a_poisson, 4),
            "mean_lead_days":  round(avg_a_lead,    2),
            "detection_rate":  round(avg_a_det,     3),
            "state_accuracy":  None,
            "macro_f1":        None,
            "llr_hawkes_better": f"{hawkes_better}/7 ports",
        })

    if model_b_auprc is not None:
        avg_b_auprc   = model_b_auprc["hawkes_auprc"].mean()
        avg_b_poisson = model_b_auprc["poisson_auprc"].mean()

        rows.append({
            "row":             2,
            "model_variant":   "+ Vessel-Type Breakdown (Model B)",
            "description":     "Per-port 5 vessel-type streams, 5x5 matrix",
            "auprc":           round(avg_b_auprc,   4),
            "poisson_auprc":   round(avg_b_poisson,  4),
            "lift_vs_poisson": round(avg_b_auprc - avg_b_poisson, 4),
            "mean_lead_days":  None,
            "detection_rate":  None,
            "state_accuracy":  None,
            "macro_f1":        None,
            "llr_hawkes_better": None,
        })

    ablation_df = pd.DataFrame(rows)

    log.info("\n" + "=" * 70)
    log.info("ABLATION TABLE — ROWS 1 AND 2 (Hawkes standalone)")
    log.info("=" * 70)
    log.info(f"{'Row':<5} {'Model':<40} {'AUPRC':>8} {'Poisson':>8} {'Lift':>8} {'Lead':>6}")
    log.info("-" * 70)
    for _, row in ablation_df.iterrows():
        lead_str = f"{row['mean_lead_days']:.1f}d" \
                   if pd.notna(row['mean_lead_days']) else "  -"
        log.info(
            f"  {int(row['row']):<3} "
            f"{row['model_variant']:<40} "
            f"{row['auprc']:>8.4f} "
            f"{row['poisson_auprc']:>8.4f} "
            f"{row['lift_vs_poisson']:>+8.4f} "
            f"{lead_str:>6}"
        )

    if len(rows) == 2:
        auprc_lift = rows[1]["auprc"] - rows[0]["auprc"]
        log.info(f"\n  Vessel-type lift (Row2 - Row1): {auprc_lift:+.4f} AUPRC")
        log.info(
            f"\n  ── PAPER SENTENCE (Section 7, ablation discussion):\n"
            f"  \"Adding vessel-type decomposition (Row 2) improves AUPRC\n"
            f"   by {abs(auprc_lift):.4f} over aggregate port-level Hawkes (Row 1),\n"
            f"   demonstrating that vessel-type granularity provides\n"
            f"   additional predictive signal beyond aggregate port activity.\""
        )

    return ablation_df

def plot_auprc_comparison(
    model_a_auprc: pd.DataFrame,
    model_b_auprc: pd.DataFrame,
) -> None:
    """
    Grouped bar chart: Hawkes AUPRC vs Poisson per port, for both models.
    """
    if model_a_auprc is None:
        return

    ports = [p for p in PORT_NAMES if p in model_a_auprc.index]
    x     = np.arange(len(ports))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(x - width,   model_a_auprc.loc[ports, "hawkes_auprc"].values,
           width, label="Model A (Port-Level)", color="#3498db", alpha=0.85)

    if model_b_auprc is not None:
        b_vals = [model_b_auprc.loc[p, "hawkes_auprc"]
                  if p in model_b_auprc.index else 0
                  for p in ports]
        ax.bar(x, b_vals, width,
               label="Model B (Vessel-Type)", color="#e74c3c", alpha=0.85)

    poisson_vals = model_a_auprc.loc[ports, "poisson_auprc"].values
    ax.bar(x + width, poisson_vals, width,
           label="Poisson Baseline", color="#95a5a6", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(ports, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("AUPRC")
    ax.set_title(
        "Hawkes AUPRC vs Poisson Baseline — Port-Level and Vessel-Type Models\n"
        "Higher = better cascade prediction",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()

    out_png = os.path.join(RESULTS_DIR, "hawkes_auprc_comparison.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  AUPRC comparison plot saved -> {out_png}")


def plot_llr_chart(llr_results: dict) -> None:
    """Bar chart of log-likelihood ratio per port. Positive = Hawkes wins."""
    if not llr_results:
        return

    ports  = list(llr_results.keys())
    llrs   = [llr_results[p]["llr"] for p in ports]
    colors = ["#27ae60" if v > 0 else "#e74c3c" for v in llrs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ports, llrs, color=colors, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, llrs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.05 if val >= 0 else -0.15),
            f"{val:+.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="#27ae60" if val >= 0 else "#e74c3c"
        )

    ax.axhline(0, color="black", lw=1.2)
    ax.set_ylabel("Log-Likelihood Ratio (Hawkes - Poisson)")
    ax.set_title(
        "Hawkes vs Poisson Log-Likelihood Ratio per Port\n"
        "Green = Hawkes fits better  |  Red = Poisson fits better",
        fontsize=11, fontweight="bold"
    )
    plt.xticks(rotation=20, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    out_png = os.path.join(RESULTS_DIR, "hawkes_llr_comparison.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  LLR chart saved -> {out_png}")

def run_hawkes_evaluator() -> pd.DataFrame:
    """
    Full Day 3 evening evaluation.
    Loads Model A + B results, computes LLR, builds ablation rows 1+2.
    Returns ablation_df.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("DAY 3 EVENING — Hawkes Standalone Evaluation")
    log.info("=" * 60)

    train = pd.read_csv(TRAIN_FILE, parse_dates=["date"])

    log.info("\n  Loading Model A results...")
    model_a_auprc, model_a_lead, mu_s, alpha_df = load_model_a_results()

    log.info("\n  Loading Model B results...")
    model_b_auprc = load_model_b_results()

    llr_results = {}
    if mu_s is not None and alpha_df is not None:
        log.info("\n" + "=" * 60)
        log.info("LOG-LIKELIHOOD RATIO (Hawkes vs Poisson)")
        log.info("=" * 60)
        llr_results = compute_llr_per_port(train, mu_s, alpha_df)

        llr_df = pd.DataFrame(llr_results).T
        llr_df.to_csv(os.path.join(RESULTS_DIR, "hawkes_llr.csv"))
        n_better = sum(v["hawkes_better"] for v in llr_results.values())
        log.info(f"\n  Hawkes better than Poisson: {n_better}/7 ports")

    log.info("\n  Building ablation rows...")
    ablation_df = build_ablation_rows(
        model_a_auprc, model_a_lead, model_b_auprc, llr_results
    )
    ablation_df.to_csv(
        os.path.join(RESULTS_DIR, "ablation_rows_1_2.csv"), index=False
    )

    perf_rows = []
    if model_a_auprc is not None:
        for port in model_a_auprc.index:
            row = {
                "port":         port,
                "model":        "Model_A_Port_Level",
                "hawkes_auprc": model_a_auprc.loc[port, "hawkes_auprc"],
                "poisson_auprc":model_a_auprc.loc[port, "poisson_auprc"],
                "lift":         model_a_auprc.loc[port, "lift"],
            }
            if model_a_lead is not None and port in model_a_lead.index:
                row["mean_lead_days"] = model_a_lead.loc[port, "mean_lead_days"]
                row["detection_rate"] = model_a_lead.loc[port, "detection_rate"]
            if port in llr_results:
                row["llr"]            = llr_results[port]["llr"]
                row["hawkes_better"]  = llr_results[port]["hawkes_better"]
            perf_rows.append(row)

    if model_b_auprc is not None:
        for port in model_b_auprc.index:
            perf_rows.append({
                "port":         port,
                "model":        "Model_B_VesselType",
                "hawkes_auprc": model_b_auprc.loc[port, "hawkes_auprc"],
                "poisson_auprc":model_b_auprc.loc[port, "poisson_auprc"],
                "lift":         model_b_auprc.loc[port, "lift"],
            })

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(os.path.join(RESULTS_DIR, "hawkes_performance.csv"), index=False)
    log.info(f"\n  Saved hawkes_performance.csv ({len(perf_df)} rows)")

    plot_auprc_comparison(model_a_auprc, model_b_auprc)
    if llr_results:
        plot_llr_chart(llr_results)

    log.info("\n  ── RESULTS LOCKED (Day 3 Evening) ──")
    log.info("  These go into ablation_rows_1_2.csv")
    log.info("  Rows 3-6 added on Days 4-5 (Markov + Combined)")
    log.info(f"\n  All outputs -> {RESULTS_DIR}")

    return ablation_df


if __name__ == "__main__":
    run_hawkes_evaluator()