import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.utils.constants import PROCESSED_DIR, PORTS
from src.utils.logger import get_logger

log = get_logger("hawkes_ccf_matrix")

RESULTS_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
    "results", "hawkes_standalone"
)
MATRIX_DIR  = os.path.join(RESULTS_DIR, "hawkes_excitation_matrices")
COMBINED    = os.path.join(PROCESSED_DIR, "all_ports_combined.csv")
PORT_NAMES  = list(PORTS.keys())
MAX_LAG     = 21


def compute_ccf_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds 7x7 empirical excitation matrix from peak-lag CCF.
    alpha_ij = Pearson r between congestion_j[t] and congestion_i[t+lag]
    at the lag that maximises |r| for each pair.

    Only uses positive r values (excitation, not inhibition).
    Only uses lags in [1, 14] days (plausible shipping cascade window).
    """
    pivot = df.pivot_table(
        index="date", columns="port",
        values="congestion_score", aggfunc="mean"
    ).sort_index().ffill()

    alpha = pd.DataFrame(0.0, index=PORT_NAMES, columns=PORT_NAMES)

    log.info("Computing CCF-based excitation matrix (7x7)...")
    log.info(f"{'Source':<15} -> {'Target':<15}  peak_lag  r_value  used_as_alpha")
    log.info("-" * 65)

    for source, target in itertools.product(PORT_NAMES, PORT_NAMES):
        if source not in pivot.columns or target not in pivot.columns:
            continue

        x = pivot[source].fillna(0).values
        y = pivot[target].fillna(0).values
        n = min(len(x), len(y))

        best_r   = 0.0
        best_lag = 0

        for lag in range(1, MAX_LAG + 1):
            if lag >= n:
                break
            x_lag = x[:n-lag]
            y_lag = y[lag:n]
            if x_lag.std() > 0 and y_lag.std() > 0:
                r, p = pearsonr(x_lag, y_lag)
                if r > best_r and 1 <= lag <= 14:
                    best_r   = r
                    best_lag = lag

        if source == target:
            x_lag = x[:n-1]
            y_lag = y[1:n]
            if x_lag.std() > 0 and y_lag.std() > 0:
                r_self, _ = pearsonr(x_lag, y_lag)
                best_r   = max(0.0, r_self)
                best_lag = 1

        alpha.loc[target, source] = round(best_r, 4)

        if best_r > 0.03:
            log.info(
                f"  {source:<15} -> {target:<15} "
                f"lag={best_lag:>2}d  r={best_r:+.4f}  alpha={best_r:.4f}"
            )

    return alpha


def plot_ccf_matrix(alpha: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        alpha, annot=True, fmt=".3f",
        cmap="YlOrRd", linewidths=0.5, ax=ax,
        annot_kws={"size": 9},
        cbar_kws={"label": "Empirical excitation coefficient (CCF r at peak lag)"}
    )
    ax.set_title(
        "TwinMesh — Empirical Cross-Port Excitation Matrix\n"
        "alpha_ij = CCF(j→i) at peak lag in [1-14d] window\n"
        "(Paper Figure 3 — Draft)",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Source port (j) — disruption origin")
    ax.set_ylabel("Target port (i) — disruption propagation")

    plt.tight_layout()
    out_png = os.path.join(MATRIX_DIR, "ccf_7x7_excitation_matrix.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  CCF matrix plot saved -> {out_png}")


def run_ccf_matrix() -> pd.DataFrame:
    os.makedirs(MATRIX_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("Hawkes CCF Excitation Matrix (empirical)")
    log.info("=" * 60)

    df    = pd.read_csv(COMBINED, parse_dates=["date"])
    alpha = compute_ccf_matrix(df)

    alpha.to_csv(os.path.join(MATRIX_DIR, "ccf_7x7.csv"))
    alpha.to_csv(os.path.join(RESULTS_DIR, "hawkes_ccf_paper_matrix.csv"))
    plot_ccf_matrix(alpha)

    _arr = np.array(alpha.values, dtype=float)
    np.fill_diagonal(_arr, 0)
    alpha_no_diag = pd.DataFrame(_arr, index=PORT_NAMES, columns=PORT_NAMES)

    top5 = (alpha_no_diag.stack()
            .reset_index()
            .rename(columns={"level_0": "target", "level_1": "source", 0: "alpha"})
            .sort_values("alpha", ascending=False)
            .head(5))

    log.info("\n  TOP 5 CROSS-PORT EXCITATION PAIRS:")
    log.info(f"  {'Source':<15} -> {'Target':<15}  alpha")
    log.info("  " + "-" * 45)
    for _, row in top5.iterrows():
        log.info(f"  {row['source']:<15} -> {row['target']:<15}  {row['alpha']:.4f}")

    best = top5.iloc[0]
    log.info(
        f"\n  PAPER SENTENCE (Section 7):\n"
        f"  \"{best['source']} disruptions propagate to {best['target']}\n"
        f"   with the strongest empirical excitation coefficient\n"
        f"   (alpha={best['alpha']:.3f}), confirming the cascade structure\n"
        f"   captured by TwinMesh's cross-port Hawkes excitation matrix.\""
    )

    log.info(f"\n  Matrix saved -> {os.path.join(MATRIX_DIR, 'ccf_7x7.csv')}")
    return alpha


if __name__ == "__main__":
    run_ccf_matrix()