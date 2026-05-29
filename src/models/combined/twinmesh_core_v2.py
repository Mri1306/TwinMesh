import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    average_precision_score,
    brier_score_loss,
)

import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../")
    )
)

from src.utils.constants import (
    PROCESSED_DIR,
    UNIFIED_DIR,
    STATES,
    STATE_MAP,
    PORTS,
)

from src.utils.logger import get_logger

log = get_logger("twinmesh_core_v2")


RESULTS_DIR = os.path.join(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../")
    ),
    "results",
    "combined"
)

HAWKES_DIR = os.path.join(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../")
    ),
    "results",
    "hawkes_standalone"
)

PORT_NAMES = list(PORTS.keys())

FEATURES_BASE = [
    "congestion_score",
    "net_dwt_flow",
    "container_ratio",
    "queue_buildup_7d",
    "yoy_pct_change",
    "anomaly_flag",
    "weather_risk_score",
    "storm_alert",
    "heavy_rain_alert",
    "hawkes_lambda",
    "hawkes_delta_lambda",
]

FEATURES_TEMPORAL = [
    "state_is_s0_prev",
    "state_is_s3_prev",
    "state_changed_last_day",
    "days_in_state",
    "congestion_score_7d",
    "congestion_score_14d",
    "congestion_accel_7d_vs_14d",
    "queue_buildup_14d",
    "queue_buildup_30d",
    "queue_accel",
]

FEATURES_HAWKES_ADVANCED = [
    "hawkes_cascade_velocity",
    "hawkes_cascade_acceleration",
    "hawkes_intensity_pct",
    "hawkes_port_baseline_dev",
    "hawkes_cross_port_strength",
    "hawkes_confidence_score",
]

FEATURES_INTERACTIONS = [
    "cascade_x_congestion",
    "cascade_x_weather",
    "anomaly_x_rising_lambda",
    "queue_x_economic_shock",
    "storm_x_cascade",
]

FEATURES_SHOCK = [
    "economic_shock_flag",
    "weather_trend",
    "combined_risk_score",
]

FEATURES_COMBINED_V2 = (
    FEATURES_BASE
    + FEATURES_TEMPORAL
    + FEATURES_HAWKES_ADVANCED
    + FEATURES_INTERACTIONS
    + FEATURES_SHOCK
)

log.info(f"Feature set size: {len(FEATURES_COMBINED_V2)} features")

def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy().sort_values(["port", "date"])

    for port in PORT_NAMES:

        port_mask = df["port"] == port

        df.loc[port_mask, "state_is_s0_prev"] = (
            df.loc[port_mask, "state"].shift(1) == "S0"
        ).astype(int).fillna(0)

        df.loc[port_mask, "state_is_s3_prev"] = (
            df.loc[port_mask, "state"].shift(1) == "S3"
        ).astype(int).fillna(0)

        df.loc[port_mask, "state_changed_last_day"] = (
            df.loc[port_mask, "state"]
            != df.loc[port_mask, "state"].shift(1)
        ).astype(int).fillna(0)

        state_col = df.loc[port_mask, "state"].values

        days_counter = np.ones(len(state_col))

        for i in range(1, len(state_col)):
            if state_col[i] == state_col[i - 1]:
                days_counter[i] = days_counter[i - 1] + 1
            else:
                days_counter[i] = 1

        df.loc[port_mask, "days_in_state"] = days_counter

        df.loc[port_mask, "congestion_score_7d"] = (
            df.loc[port_mask, "congestion_score"]
            .rolling(window=7, min_periods=1)
            .mean()
            .values
        )

        df.loc[port_mask, "congestion_score_14d"] = (
            df.loc[port_mask, "congestion_score"]
            .rolling(window=14, min_periods=1)
            .mean()
            .values
        )

        df.loc[port_mask, "congestion_accel_7d_vs_14d"] = (
            (
                df.loc[port_mask, "congestion_score_7d"]
                - df.loc[port_mask, "congestion_score_14d"]
            )
            /
            (
                df.loc[port_mask, "congestion_score_14d"] + 0.001
            )
        ).fillna(0)

        df.loc[port_mask, "queue_buildup_14d"] = (
            df.loc[port_mask, "queue_buildup_7d"]
            .rolling(window=14, min_periods=1)
            .mean()
            .values
        )

        df.loc[port_mask, "queue_buildup_30d"] = (
            df.loc[port_mask, "queue_buildup_7d"]
            .rolling(window=30, min_periods=1)
            .mean()
            .values
        )

        df.loc[port_mask, "queue_accel"] = (
            (
                df.loc[port_mask, "queue_buildup_7d"]
                - df.loc[port_mask, "queue_buildup_14d"]
            )
            /
            (
                df.loc[port_mask, "queue_buildup_14d"] + 0.001
            )
        ).fillna(0)

    return df

def engineer_hawkes_advanced_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy().sort_values(["port", "date"])

    for port in PORT_NAMES:

        port_mask = df["port"] == port

        df.loc[port_mask, "hawkes_cascade_velocity"] = (
            df.loc[port_mask, "hawkes_delta_lambda"]
        )

        df.loc[port_mask, "hawkes_cascade_acceleration"] = (
            df.loc[port_mask, "hawkes_cascade_velocity"]
            .diff()
            .fillna(0)
        )

        port_lambdas = df.loc[port_mask, "hawkes_lambda"]

        df.loc[port_mask, "hawkes_intensity_pct"] = (
            port_lambdas.rank(pct=True).values
        )

        port_mean = port_lambdas.mean()
        port_std = port_lambdas.std() + 1e-6

        df.loc[port_mask, "hawkes_port_baseline_dev"] = (
            (port_lambdas - port_mean) / port_std
        ).values

    df["hawkes_cross_port_strength"] = (
        df.groupby("date")["hawkes_lambda"]
        .transform("sum")
    )

    port_groups = (
        df.groupby("date")["hawkes_lambda"]
        .agg(["std", "mean"])
    )

    cv = (
        port_groups["std"]
        /
        (port_groups["mean"] + 1e-6)
    )

    confidence = 1 / (1 + cv)

    df["hawkes_confidence_score"] = (
        df["date"]
        .map(confidence)
        .fillna(0.5)
    )

    return df

def engineer_interaction_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["cascade_x_congestion"] = (
        df["hawkes_lambda"]
        * df["congestion_score"]
    )

    df["cascade_x_weather"] = (
        df["hawkes_lambda"]
        * df["weather_risk_score"]
    )

    df["anomaly_x_rising_lambda"] = (
        df["anomaly_flag"]
        *
        (df["hawkes_delta_lambda"] > 0).astype(int)
    )

    economic_shock = np.abs(df["yoy_pct_change"]) > 20

    df["queue_x_economic_shock"] = (
        df["queue_buildup_7d"]
        * economic_shock.astype(int)
    )

    df["storm_x_cascade"] = (
        df["storm_alert"]
        * df["hawkes_lambda"]
    )

    df["economic_shock_flag"] = economic_shock.astype(int)

    df["weather_trend"] = (
        df.groupby("port")["storm_alert"]
        .shift(1)
        .fillna(0)
    )

    df["combined_risk_score"] = (
        df["hawkes_lambda"]
        + df["weather_risk_score"]
    ) / 2

    return df

def prepare_combined_sequences_v2(df):

    df = engineer_temporal_features(df)
    df = engineer_hawkes_advanced_features(df)
    df = engineer_interaction_features(df)

    for col in FEATURES_COMBINED_V2:
        if col not in df.columns:
            df[col] = 0.0

        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .fillna(0)
        )

    X_list = []
    y_list = []

    for port in PORT_NAMES:

        port_df = (
            df[df["port"] == port]
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(port_df) < 2:
            continue

        X = port_df[FEATURES_COMBINED_V2].values[:-1]

        y = np.array([
            STATE_MAP.get(s, 0)
            for s in port_df["state"].values[1:]
        ])

        X_list.append(X)
        y_list.append(y)

    return np.vstack(X_list), np.concatenate(y_list)

def fit_xgboost_model(X_train, y_train):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_train)

    n_samples = len(y_train)

    class_counts = np.bincount(
        y_train,
        minlength=len(STATES)
    )

    s3_idx = STATE_MAP["S3"]

    n_s3 = max(class_counts[s3_idx], 1)

    n_others = n_samples - n_s3

    scale_pos_weight = n_others / n_s3

    log.info(f"  Class distribution: {class_counts}")
    log.info(f"  S3 class weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        max_depth=7,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    model.fit(X_scaled, y_train)

    train_acc = model.score(X_scaled, y_train)

    log.info(
        f"  TwinMesh V2 train_accuracy={train_acc:.4f}"
    )

    return model, scaler

def evaluate_combined_v2(
    model,
    scaler,
    X_test,
    y_test
):

    X_scaled = scaler.transform(X_test)

    preds = model.predict(X_scaled)

    probs = model.predict_proba(X_scaled)

    accuracy = accuracy_score(y_test, preds)

    macro_f1 = f1_score(
        y_test,
        preds,
        average="macro",
        zero_division=0
    )

    s3_idx = STATE_MAP["S3"]

    s3_true = (y_test == s3_idx).astype(int)

    s3_probs = probs[:, s3_idx]

    s3_recall = (
        float(
            s3_true[preds == s3_idx].sum()
        )
        /
        max(s3_true.sum(), 1)
    )

    s3_auprc = average_precision_score(
        s3_true,
        s3_probs
    )

    brier_s3 = brier_score_loss(
        s3_true,
        s3_probs
    )

    metrics = {
        "model": "TwinMesh_V2",
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "s3_recall": round(s3_recall, 4),
        "s3_auprc": round(s3_auprc, 4),
        "brier_s3": round(brier_s3, 4),
    }

    log.info("\n" + "=" * 60)
    log.info("TWINMESH V2 — Results")
    log.info("=" * 60)

    for k, v in metrics.items():
        log.info(f"  {k}: {v}")

    return metrics, probs, s3_probs

def run_twinmesh_v2():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("DAY 5 MORNING (IMPROVED) — TwinMesh V2")
    log.info("XGBoost + Advanced Feature Engineering")
    log.info("=" * 60)

    train = pd.read_csv(
        os.path.join(UNIFIED_DIR, "twinmesh_training.csv"),
        parse_dates=["date"]
    )

    test = pd.read_csv(
        os.path.join(UNIFIED_DIR, "twinmesh_test.csv"),
        parse_dates=["date"]
    )

    log.info("\n  Loading Hawkes features...")

    combined_all = pd.read_csv(
        os.path.join(PROCESSED_DIR, "all_ports_combined.csv"),
        parse_dates=["date"]
    )

    combined_with_lambda = None

    try:

        from src.models.combined.twinmesh_core import (
            compute_hawkes_intensity
        )

        log.info(
            "  Imported compute_hawkes_intensity successfully"
        )

        combined_with_lambda = compute_hawkes_intensity(
            combined_all
        )

    except Exception as e:

        log.warning(
            f"  Could not compute Hawkes intensities: {e}"
        )

        hawkes_file = os.path.join(
            HAWKES_DIR,
            "hawkes_cascade_timeseries.csv"
        )

        if os.path.exists(hawkes_file):

            hawkes_df = pd.read_csv(
                hawkes_file,
                parse_dates=["date"]
            )

            lambda_cols = [
                "date",
                "port",
                "hawkes_lambda",
                "hawkes_delta_lambda",
            ]

            if all(
                c in hawkes_df.columns
                for c in lambda_cols
            ):

                lambda_df = hawkes_df[lambda_cols]

                combined_with_lambda = (
                    combined_all.merge(
                        lambda_df,
                        on=["date", "port"],
                        how="left"
                    )
                )

    if combined_with_lambda is None:

        log.warning(
            "  Using default Hawkes values (0.08)"
        )

        combined_all["hawkes_lambda"] = 0.08
        combined_all["hawkes_delta_lambda"] = 0.0

        combined_with_lambda = combined_all

    lambda_cols = [
        "date",
        "port",
        "hawkes_lambda",
        "hawkes_delta_lambda",
    ]

    lambda_df = combined_with_lambda[lambda_cols]

    train = train.merge(
        lambda_df,
        on=["date", "port"],
        how="left"
    )

    test = test.merge(
        lambda_df,
        on=["date", "port"],
        how="left"
    )

    train["hawkes_lambda"] = (
        train["hawkes_lambda"]
        .fillna(0.08)
    )

    train["hawkes_delta_lambda"] = (
        train["hawkes_delta_lambda"]
        .fillna(0.0)
    )

    test["hawkes_lambda"] = (
        test["hawkes_lambda"]
        .fillna(0.08)
    )

    test["hawkes_delta_lambda"] = (
        test["hawkes_delta_lambda"]
        .fillna(0.0)
    )

    log.info(
        f"  Lambda merged | "
        f"train lambda range: "
        f"[{train['hawkes_lambda'].min():.4f}, "
        f"{train['hawkes_lambda'].max():.4f}]"
    )

    log.info("\n  Step 1: Preparing sequences...")

    X_train, y_train = prepare_combined_sequences_v2(
        train
    )

    X_test, y_test = prepare_combined_sequences_v2(
        test
    )

    log.info(
        f"  Train: {len(X_train):,} | "
        f"Test: {len(X_test):,}"
    )

    log.info(
        f"  Features: {X_train.shape[1]}"
    )

    log.info("\n  Step 2: Fitting XGBoost...")

    model, scaler = fit_xgboost_model(
        X_train,
        y_train
    )

    log.info("\n  Step 3: Evaluating...")

    metrics, probs, s3_probs = evaluate_combined_v2(
        model,
        scaler,
        X_test,
        y_test
    )

    pd.DataFrame([metrics]).to_csv(
        os.path.join(
            RESULTS_DIR,
            "twinmesh_v2_metrics.csv"
        ),
        index=False
    )

    log.info(
        "\n  Metrics saved → twinmesh_v2_metrics.csv"
    )

    return model, scaler, metrics


if __name__ == "__main__":
    run_twinmesh_v2()

