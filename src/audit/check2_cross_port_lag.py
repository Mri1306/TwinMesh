import os
import sys
import itertools
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.constants import PROCESSED_DIR, PORTS
from src.utils.logger import get_logger

log = get_logger("check2_cross_port_lag")

AUDIT_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")),
    "results", "audit"
)
COMBINED = os.path.join(PROCESSED_DIR, "all_ports_combined.csv")

MAX_LAG_DAYS   = 21    # test lags 0–21 days
EXPECTED_LAG_MIN = 7
EXPECTED_LAG_MAX = 14

PORT_NAMES = list(PORTS.keys())

KEY_PAIRS = [
    ("Shanghai",    "Singapore"),
    ("Shanghai",    "Rotterdam"),
    ("Singapore",   "Rotterdam"),
    ("Singapore",   "Hamburg"),
    ("Nhava Sheva", "Rotterdam"),
    ("Busan",       "Singapore"),
]

def compute_ccf(x: pd.Series, y: pd.Series, max_lag: int) -> dict:
    """
    Cross-correlation function: for each lag k (0..max_lag),
    compute Pearson r between x[t] and y[t+k].
    Returns dict {lag: r_value}.
    """
    x = x.fillna(0).values
    y = y.fillna(0).values
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    ccf_vals = {}
    for lag in range(0, max_lag + 1):
        if lag == 0:
            r, _ = pearsonr(x, y)
        else:
            r, _ = pearsonr(x[:-lag], y[lag:])
        ccf_vals[lag] = round(r, 4)

    return ccf_vals


def find_peak_lag(ccf_vals: dict) -> tuple:
    """Returns (peak_lag, peak_r) — the lag with highest absolute correlation."""
    peak_lag = max(ccf_vals, key=lambda k: abs(ccf_vals[k]))
    return peak_lag, ccf_vals[peak_lag]

def check_cross_port_lag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes CCF for all port pairs. Records peak lag and r value.
    Flags pairs where peak lag falls in expected 7–14 day window.
    """
    log.info("\n" + "=" * 60)
    log.info("CHECK 2 — Cross-Port Lag Correlation")
    log.info("=" * 60)
    log.info(f"  Expected lag window: {EXPECTED_LAG_MIN}–{EXPECTED_LAG_MAX} days")
    log.info(f"  Max lag tested:      {MAX_LAG_DAYS} days")

    os.makedirs(AUDIT_DIR, exist_ok=True)

    
    pivot = df.pivot_table(
        index="date", columns="port",
        values="congestion_score", aggfunc="mean"
    ).sort_index()

    records      = []
    ccf_store    = {}   # for plotting

    all_pairs = list(itertools.permutations(PORT_NAMES, 2))

    for source, target in all_pairs:
        if source not in pivot.columns or target not in pivot.columns:
            continue

        x = pivot[source]
        y = pivot[target]

        ccf_vals        = compute_ccf(x, y, MAX_LAG_DAYS)
        peak_lag, peak_r = find_peak_lag(ccf_vals)
        in_window       = EXPECTED_LAG_MIN <= peak_lag <= EXPECTED_LAG_MAX

        records.append({
            "source":       source,
            "target":       target,
            "peak_lag_days": peak_lag,
            "peak_r":        peak_r,
            "r_at_lag_7":   ccf_vals.get(7, None),
            "r_at_lag_14":  ccf_vals.get(14, None),
            "in_expected_window": in_window,
        })

        ccf_store[(source, target)] = ccf_vals

        pair_label = f"{source} → {target}"
        flag = "✅" if in_window else "  "
        log.info(
            f"  {flag} {pair_label:<35} "
            f"peak_lag={peak_lag:>2}d  r={peak_r:+.3f}"
            + (" ← IN WINDOW" if in_window else "")
        )

    results_df = pd.DataFrame(records)

    
    out_csv = os.path.join(AUDIT_DIR, "check2_lag_correlations.csv")
    results_df.to_csv(out_csv, index=False)
    log.info(f"\n  Saved → {out_csv}")

    
    n_in_window = results_df["in_expected_window"].sum()
    n_total     = len(results_df)
    log.info(
        f"\n  Pairs with peak lag in {EXPECTED_LAG_MIN}–{EXPECTED_LAG_MAX}d window: "
        f"{n_in_window}/{n_total}"
    )

    
    key_results = results_df[
        results_df.apply(
            lambda r: (r["source"], r["target"]) in KEY_PAIRS, axis=1
        )
    ].sort_values("peak_r", ascending=False)

    if len(key_results):
        best = key_results.iloc[0]
        log.info(
            f"\n  ── PAPER SENTENCE (fill this in):\n"
            f"  \"{best['source']} disruptions propagate to {best['target']}\n"
            f"   with a peak lag of {best['peak_lag_days']} days (r={best['peak_r']:+.3f}),\n"
            f"   confirming the cascade structure captured by TwinMesh's\n"
            f"   Hawkes excitation matrix.\""
        )

    return results_df, ccf_store

def plot_lag_heatmap(results_df: pd.DataFrame) -> None:
    """
    Heatmap of peak_r values for all port pairs.
    Draft of Figure 3 — Hawkes excitation proxy.
    """
    pivot = results_df.pivot(
        index="source", columns="target", values="peak_r"
    ).reindex(index=PORT_NAMES, columns=PORT_NAMES)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, vmin=-0.5, vmax=0.8,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 9}
    )
    ax.set_title(
        "Cross-Port Peak Lag Correlation (r values)\n"
        "Draft proxy for Hawkes excitation matrix β\n"
        f"(Peak lag computed over 0–{MAX_LAG_DAYS} day window)",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Target Port (y[t + lag])", fontsize=10)
    ax.set_ylabel("Source Port (x[t])", fontsize=10)

    plt.tight_layout()
    out_png = os.path.join(AUDIT_DIR, "check2_lag_heatmap.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Lag heatmap saved → {out_png}")


def plot_key_pair_ccf(ccf_store: dict) -> None:
    """
    CCF plots for the key shipping lane pairs only.
    """
    valid_pairs = [p for p in KEY_PAIRS if p in ccf_store]
    n = len(valid_pairs)
    if n == 0:
        return

    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    fig.suptitle(
        "Check 2 — Cross-Port CCF (Key Shipping Lane Pairs)",
        fontsize=12, fontweight="bold"
    )

    for ax, (source, target) in zip(axes, valid_pairs):
        ccf_vals = ccf_store[(source, target)]
        lags     = list(ccf_vals.keys())
        r_vals   = list(ccf_vals.values())
        peak_lag, peak_r = find_peak_lag(ccf_vals)

        bars = ax.bar(lags, r_vals,
                      color=["#e74c3c" if l == peak_lag else "#3498db" for l in lags],
                      alpha=0.8, width=0.7)

        
        ax.axvspan(EXPECTED_LAG_MIN - 0.5, EXPECTED_LAG_MAX + 0.5,
                   alpha=0.12, color="#2ecc71", label=f"Expected window ({EXPECTED_LAG_MIN}–{EXPECTED_LAG_MAX}d)")

        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(peak_lag, color="#e74c3c", lw=1.5, linestyle="--", alpha=0.7)

        ax.set_title(
            f"{source}  →  {target}\n"
            f"Peak lag: {peak_lag}d  |  r={peak_r:+.3f}",
            fontsize=10, fontweight="bold"
        )
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Pearson r")
        ax.set_xticks(lags)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    out_png = os.path.join(AUDIT_DIR, "check2_ccf_plots.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  CCF plots saved → {out_png}")

def run_check2() -> pd.DataFrame:
    if not os.path.exists(COMBINED):
        raise FileNotFoundError(f"Run Day 1 first. Missing: {COMBINED}")

    df = pd.read_csv(COMBINED, parse_dates=["date"])
    df = df[df["date"] >= "2023-01-01"]

    results_df, ccf_store = check_cross_port_lag(df)
    plot_lag_heatmap(results_df)
    plot_key_pair_ccf(ccf_store)
    return results_df


if __name__ == "__main__":
    run_check2()