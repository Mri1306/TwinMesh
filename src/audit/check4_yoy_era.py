import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.constants import PROCESSED_DIR, PER_PORT_DIR, PORTS, STATES
from src.utils.logger import get_logger

log = get_logger("check4_yoy_era")

AUDIT_DIR   = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")),
    "results", "audit"
)
COMBINED    = os.path.join(PROCESSED_DIR, "all_ports_combined.csv")
HISTORICAL  = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")),
    "data", "historical", "portwatch_2019_2022.csv"
)

ERAS = {
    "Pre-COVID (2019)":       ("2019-01-01", "2019-12-31"),
    "COVID + Suez (2020–22)": ("2020-01-01", "2022-12-31"),
    "Red Sea (2023–26)":      ("2023-01-01", "2026-12-31"),
}

ERA_COLORS = {
    "Pre-COVID (2019)":       "#2ecc71",
    "COVID + Suez (2020–22)": "#e67e22",
    "Red Sea (2023–26)":      "#e74c3c",
}

PORT_NAMES = list(PORTS.keys())

def load_all_eras() -> pd.DataFrame:
    """
    Loads all available PortWatch data across all eras.
    Tries combined first, then falls back to historical + combined stitch.
    """
    frames = []

    if os.path.exists(COMBINED):
        df_2023 = pd.read_csv(COMBINED, parse_dates=["date"])
        frames.append(df_2023)
        log.info(f"  Loaded 2023+ data: {len(df_2023):,} rows")

    if os.path.exists(HISTORICAL):
        df_hist = pd.read_csv(HISTORICAL, parse_dates=["date"])
        frames.append(df_hist)
        log.info(f"  Loaded 2019–2022 data: {len(df_hist):,} rows")
    else:
        log.warning(
            "  portwatch_2019_2022.csv not found — "
            "loading per-port merged CSVs for full history"
        )
        for port_name, prefix in PORTS.items():
            path = os.path.join(PER_PORT_DIR, f"{prefix}_merged.csv")
            if os.path.exists(path):
                port_df = pd.read_csv(path, parse_dates=["date"])
                port_df["port"] = port_name
                frames.append(port_df)

    if not frames:
        raise FileNotFoundError(
            "No data found. Run Day 1 pipeline first:\n"
            "  python run_preprocessing.py"
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "port"])
    combined = combined.sort_values(["date", "port"])

    log.info(
        f"  Full dataset: {len(combined):,} rows | "
        f"{combined['date'].min().date()} → {combined['date'].max().date()}"
    )
    return combined

def label_era(date: pd.Timestamp) -> str:
    for era_name, (start, end) in ERAS.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return era_name
    return "Other"

def compute_era_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each port × era: mean congestion, S3 rate, total_calls mean,
    net_dwt_flow mean, disruption days (if available).
    """
    df = df.copy()
    df["era"] = df["date"].apply(label_era)
    df = df[df["era"] != "Other"]

    records = []
    for port in PORT_NAMES:
        for era_name in ERAS.keys():
            subset = df[(df["port"] == port) & (df["era"] == era_name)]
            if len(subset) == 0:
                continue

            s3_rate = (subset.get("state", pd.Series([])) == "S3").mean() \
                      if "state" in subset.columns else np.nan

            records.append({
                "port":               port,
                "era":                era_name,
                "n_days":             len(subset),
                "mean_congestion":    round(subset["congestion_score"].mean(), 4),
                "max_congestion":     round(subset["congestion_score"].max(), 4),
                "mean_total_calls":   round(subset["total_calls"].mean(), 2)
                                      if "total_calls" in subset.columns else np.nan,
                "mean_net_dwt_flow":  round(subset["net_dwt_flow"].mean(), 2)
                                      if "net_dwt_flow" in subset.columns else np.nan,
                "s3_rate":            round(s3_rate, 4)
                                      if not np.isnan(s3_rate) else np.nan,
                "anomaly_days":       int(subset["anomaly_flag"].sum())
                                      if "anomaly_flag" in subset.columns else 0,
            })

    return pd.DataFrame(records)


def check_era_distinction(era_stats: pd.DataFrame) -> bool:
    """
    PASS if Red Sea era shows higher mean congestion than Pre-COVID
    for a majority of ports.
    """
    log.info("\n" + "=" * 60)
    log.info("CHECK 4 — YoY Era Comparison")
    log.info("=" * 60)

    passed_ports = 0
    for port in PORT_NAMES:
        port_data = era_stats[era_stats["port"] == port]
        pre_row   = port_data[port_data["era"] == "Pre-COVID (2019)"]
        rs_row    = port_data[port_data["era"] == "Red Sea (2023–26)"]

        if pre_row.empty or rs_row.empty:
            continue

        pre_cong = pre_row["mean_congestion"].values[0]
        rs_cong  = rs_row["mean_congestion"].values[0]
        pre_s3   = pre_row["s3_rate"].values[0]
        rs_s3    = rs_row["s3_rate"].values[0]

        port_pass = rs_cong > pre_cong
        if port_pass:
            passed_ports += 1

        flag = "✅" if port_pass else "⚠️ "
        log.info(
            f"\n  {flag} {port}:"
            f"\n    Pre-COVID congestion: {pre_cong:.4f}  |  S3 rate: {pre_s3:.1%}"
            f"\n    Red Sea congestion:   {rs_cong:.4f}  |  S3 rate: {rs_s3:.1%}"
            f"\n    Lift: {rs_cong - pre_cong:+.4f}"
        )

    majority = len(PORT_NAMES) // 2 + 1
    overall_pass = passed_ports >= majority

    log.info(
        f"\n  Ports where Red Sea > Pre-COVID: {passed_ports}/{len(PORT_NAMES)}"
    )
    log.info(f"  Overall: {'✅ PASS — 3 eras are distinct' if overall_pass else '❌ FAIL — eras not clearly separated'}")

    if not overall_pass:
        log.warning(
            "  ACTION: Check date range coverage in portwatch_2019_2022.csv\n"
            "          and congestion threshold in constants.py"
        )

    return overall_pass

def plot_era_heatmap(era_stats: pd.DataFrame) -> None:
    """
    Heatmap: ports × eras, colored by mean_congestion.
    """
    pivot = era_stats.pivot(
        index="port", columns="era", values="mean_congestion"
    )

    era_order = list(ERAS.keys())
    pivot = pivot.reindex(columns=[e for e in era_order if e in pivot.columns])
    pivot = pivot.reindex(PORT_NAMES)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".3f",
        cmap="YlOrRd", linewidths=0.5, ax=ax,
        annot_kws={"size": 10, "weight": "bold"},
        cbar_kws={"label": "Mean Congestion Score"}
    )
    ax.set_title(
        "Check 4 — Mean Congestion Score by Port × Era\n"
        "Red = higher congestion  |  3 eras must be visually distinct",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Era", fontsize=10)
    ax.set_ylabel("Port", fontsize=10)
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_png = os.path.join(AUDIT_DIR, "check4_era_heatmap.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Era heatmap saved → {out_png}")


def plot_era_timeseries(df: pd.DataFrame) -> None:
    """
    Full congestion timeline per port (2019–2026).
    Era background shading + 30-day rolling mean.
    """
    fig, axes = plt.subplots(
        len(PORT_NAMES), 1,
        figsize=(16, 3 * len(PORT_NAMES)),
        sharex=True
    )
    fig.suptitle(
        "Check 4 — Congestion Score Timeline (Full History)\n"
        "Shading: Pre-COVID / COVID / Red Sea eras",
        fontsize=12, fontweight="bold", y=1.01
    )

    for ax, port in zip(axes, PORT_NAMES):
        port_df = df[df["port"] == port].sort_values("date")

        ax.plot(port_df["date"], port_df["congestion_score"],
                color="#bdc3c7", lw=0.6, alpha=0.7)

        roll = port_df["congestion_score"].rolling(30, min_periods=1).mean()
        ax.plot(port_df["date"], roll,
                color="#2c3e50", lw=1.8, label="30d MA")

        for era_name, (start, end) in ERAS.items():
            ax.axvspan(
                pd.Timestamp(start), pd.Timestamp(end),
                alpha=0.10,
                color=ERA_COLORS[era_name],
                label=era_name
            )

        if "state" in port_df.columns:
            s3 = port_df[port_df["state"] == "S3"]
            ax.scatter(s3["date"], s3["congestion_score"],
                       color="#e74c3c", s=8, zorder=5, alpha=0.7)

        ax.set_title(port, fontsize=9, fontweight="bold", loc="left", pad=2)
        ax.set_ylabel("Cong.", fontsize=8)
        ax.grid(True, alpha=0.2)

        if port == PORT_NAMES[0]:
            ax.legend(fontsize=7, loc="upper right", ncol=4)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()

    out_png = os.path.join(AUDIT_DIR, "check4_era_timeseries.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Era timeseries saved → {out_png}")


def plot_era_boxplot(df: pd.DataFrame) -> None:
    """
    Boxplot: congestion_score distribution per era per port.
    Shows spread + outliers across eras.
    """
    df = df.copy()
    df["era"] = df["date"].apply(label_era)
    df = df[df["era"] != "Other"]

    fig, axes = plt.subplots(
        2, 4, figsize=(16, 8), sharey=False
    )
    axes = axes.flatten()

    for i, port in enumerate(PORT_NAMES):
        ax = axes[i]
        port_df = df[df["port"] == port]

        era_groups = [
            port_df[port_df["era"] == era]["congestion_score"].dropna()
            for era in ERAS.keys()
        ]
        ax.boxplot(
            era_groups,
            labels=[e.split("(")[0].strip() for e in ERAS.keys()],
            patch_artist=True,
            boxprops=dict(facecolor="#d5e8d4"),
            medianprops=dict(color="#e74c3c", lw=2),
            whiskerprops=dict(color="#7f8c8d"),
            capprops=dict(color="#7f8c8d"),
            flierprops=dict(marker=".", color="#e74c3c", alpha=0.4, ms=3)
        )
        ax.set_title(port, fontsize=10, fontweight="bold")
        ax.set_ylabel("Congestion Score")
        ax.grid(True, alpha=0.3, axis="y")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right", fontsize=8)

    for ax in axes[len(PORT_NAMES):]:
        ax.set_visible(False)

    fig.suptitle(
        "Check 4 — Congestion Distribution by Era\n"
        "Boxes must shift rightward Pre-COVID → COVID → Red Sea",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()

    out_png = os.path.join(AUDIT_DIR, "check4_era_boxplot.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Era boxplot saved → {out_png}")

def run_check4() -> tuple:
    os.makedirs(AUDIT_DIR, exist_ok=True)

    df       = load_all_eras()
    era_stats = compute_era_stats(df)
    passed   = check_era_distinction(era_stats)

    out_csv = os.path.join(AUDIT_DIR, "check4_era_comparison.csv")
    era_stats.to_csv(out_csv, index=False)
    log.info(f"\n  Saved → {out_csv}")

    log.info(f"\n  Era stats summary:\n{era_stats.to_string(index=False)}")

    plot_era_heatmap(era_stats)
    plot_era_timeseries(df)
    plot_era_boxplot(df)

    return era_stats, passed


if __name__ == "__main__":
    run_check4()