# =============================================================================
# TwinMesh App · pages/replay_analysis.py
# Screen 3 — Replay Simulation
# Shows actual vs predicted state, Hawkes lambda, alerts over time
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os

from app.backend.replay_service import (
    get_port_replay,
    get_replay_summary,
    get_hawkes_series_all
)

STATE_LABELS = {
    "S0": "Normal",
    "S1": "Delayed",
    "S2": "Congested",
    "S3": "Disrupted",
    "S4": "Recovery",
}

STATE_NUM = {
    "S0": 0,
    "S1": 1,
    "S2": 2,
    "S3": 3,
    "S4": 4,
}

STATE_COLOR_HEX = {
    "S0": "#10b981",
    "S1": "#f59e0b",
    "S2": "#f97316",
    "S3": "#ef4444",
    "S4": "#3b82f6",
}

PORTS = [
    "Rotterdam",
    "Singapore",
    "Shanghai",
    "Nhava Sheva",
    "Busan",
    "Hamburg",
    "Antwerp"
]


# =============================================================================
# MAIN RENDER
# =============================================================================

def render():

    st.title("📼 TwinMesh — Replay Simulation")

    st.caption(
        "Day-by-day digital twin replay: actual state vs predicted state, "
        "Hawkes cascade intensity, and early warning alerts."
    )

    # =========================================================================
    # LOAD SUMMARY
    # =========================================================================

    with st.spinner("Loading replay data..."):
        summary = get_replay_summary()

    if summary:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total decisions",
            f"{summary.get('total_decisions', 0):,}"
        )

        c2.metric(
            "Alerts fired",
            f"{summary.get('n_alerts', 0):,}",
            delta=f"{summary.get('alert_rate', 0):.1%} of days"
        )

        c3.metric(
            "S3 days",
            f"{summary.get('n_s3_days', 0):,}"
        )

        c4.metric(
            "State accuracy",
            f"{summary.get('accuracy', 0):.1%}"
        )

        st.divider()

    # =========================================================================
    # DYNAMIC DATE RANGE
    # =========================================================================

    replay_csv = os.path.join(
        "results",
        "digital_twin",
        "twin_replay_log.csv"
    )

    if not os.path.exists(replay_csv):
        st.error("Replay log not found.")
        return

    replay_df = pd.read_csv(
        replay_csv,
        parse_dates=["date"]
    )

    if replay_df.empty:
        st.warning("Replay log is empty.")
        return

    replay_df["date"] = pd.to_datetime(replay_df["date"])

    min_date = replay_df["date"].min().date()
    max_date = replay_df["date"].max().date()

    # =========================================================================
    # PORT + DATE SELECTOR
    # =========================================================================

    col_sel, col_date = st.columns([1, 1])

    with col_sel:

        port = st.selectbox(
            "Select port",
            PORTS,
            index=0
        )

    with col_date:

        st.caption(f"Replay period: {min_date} → {max_date}")

        start_d, end_d = st.slider(
            "Date range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
        )

    start_d = str(start_d)
    end_d = str(end_d)

    # =========================================================================
    # LOAD PORT REPLAY
    # =========================================================================

    with st.spinner(f"Loading {port} replay..."):

        data = get_port_replay(
            port,
            start_d,
            end_d
        )

    if not data or data.get("n_rows", 0) == 0:

        st.warning(
            f"No replay data found for {port} "
            f"between {start_d} and {end_d}"
        )

        return

    # =========================================================================
    # EXTRACT SERIES
    # =========================================================================

    dates = data["dates"]

    actual = data["actual_state"]

    predicted = data["predicted_state"]

    p_s3 = data["p_s3"]

    lam = data["hawkes_lambda"]

    alerts = data["alerts"]

    tier0 = data.get(
        "tier0_action",
        ["Strong"] * len(dates)
    )

    act_num = [STATE_NUM.get(s, 0) for s in actual]

    pred_num = [STATE_NUM.get(s, 0) for s in predicted]

    # =========================================================================
    # MAIN REPLAY FIGURE
    # =========================================================================

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.25, 0.25, 0.25, 0.25],
        vertical_spacing=0.04,
        subplot_titles=[
            f"{port} — Actual State",
            f"{port} — Predicted State",
            "P(S3) — Disruption Probability",
            "Hawkes λ(t) — Cascade Intensity",
        ],
    )

    # =========================================================================
    # ROW 1 — ACTUAL STATE
    # =========================================================================

    for s, color in [

        ("S0", "#10b981"),
        ("S1", "#f59e0b"),
        ("S2", "#f97316"),
        ("S3", "#ef4444"),
        ("S4", "#3b82f6"),

    ]:

        mask = [i for i, a in enumerate(actual) if a == s]

        if mask:

            fig.add_trace(

                go.Bar(
                    x=[dates[i] for i in mask],
                    y=[1] * len(mask),

                    name=f"{s} {STATE_LABELS[s]}",

                    marker_color=color,

                    marker_line_width=0,

                    showlegend=True if s == "S3" else False,

                    legendgroup=s,

                    hovertemplate=
                    f"<b>{port}</b><br>"
                    f"Date: %{{x}}<br>"
                    f"State: {s}<extra></extra>",
                ),

                row=1,
                col=1
            )

    # =========================================================================
    # ROW 2 — PREDICTED STATE
    # =========================================================================

    for s, color in [

        ("S0", "#10b981"),
        ("S1", "#f59e0b"),
        ("S2", "#f97316"),
        ("S3", "#ef4444"),
        ("S4", "#3b82f6"),

    ]:

        mask = [i for i, p in enumerate(predicted) if p == s]

        if mask:

            fig.add_trace(

                go.Bar(
                    x=[dates[i] for i in mask],
                    y=[1] * len(mask),

                    name=f"Pred {s}",

                    marker_color=color,

                    marker_line_width=0,

                    showlegend=False,

                    opacity=0.7,
                ),

                row=2,
                col=1
            )

    # =========================================================================
    # ROW 3 — P(S3)
    # =========================================================================

    fig.add_trace(

        go.Scatter(
            x=dates,
            y=p_s3,

            fill="tozeroy",

            fillcolor="rgba(239,68,68,0.12)",

            line=dict(
                color="#ef4444",
                width=1.5
            ),

            connectgaps=True,

            name="P(S3)",

            hovertemplate=
            "%{x}: P(S3)=%{y:.1%}<extra></extra>",
        ),

        row=3,
        col=1
    )

    fig.add_hline(
        y=0.30,

        line_dash="dash",

        line_color="#f59e0b",

        annotation_text="Alert threshold",

        row=3,
        col=1
    )

    # =========================================================================
    # ALERT MARKERS
    # =========================================================================

    alert_dates = [
        dates[i]
        for i, a in enumerate(alerts)
        if a
    ]

    alert_ps3 = [
        p_s3[i]
        for i, a in enumerate(alerts)
        if a
    ]

    if alert_dates:

        fig.add_trace(

            go.Scatter(
                x=alert_dates,
                y=alert_ps3,

                mode="markers",

                marker=dict(
                    color="#ef4444",
                    size=7,
                    symbol="triangle-up"
                ),

                name="Alert fired",

                hovertemplate=
                "⚠ Alert: %{x}<extra></extra>",
            ),

            row=3,
            col=1
        )

    # =========================================================================
    # ROW 4 — HAWKES LAMBDA
    # =========================================================================

    fig.add_trace(

        go.Scatter(
            x=dates,
            y=lam,

            fill="tozeroy",

            fillcolor="rgba(59,130,246,0.12)",

            line=dict(
                color="#3b82f6",
                width=1.5
            ),

            connectgaps=True,

            name="λ(t)",

            hovertemplate=
            "%{x}: λ=%{y:.4f}<extra></extra>",
        ),

        row=4,
        col=1
    )

    fig.add_hline(
        y=0.08,

        line_dash="dot",

        line_color="#94a3b8",

        annotation_text="μ baseline",

        row=4,
        col=1
    )

    # =========================================================================
    # FIGURE LAYOUT
    # =========================================================================

    fig.update_layout(

        height=700,

        barmode="stack",

        showlegend=True,

        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10
        ),

        plot_bgcolor="rgba(0,0,0,0)",

        paper_bgcolor="rgba(0,0,0,0)",

        font=dict(size=10),

        legend=dict(
            orientation="h",
            y=-0.05,
            x=0,
        ),
    )

    for row in range(1, 5):

        fig.update_yaxes(
            showgrid=True,
            gridcolor="#f1f5f9",
            row=row,
            col=1
        )

        fig.update_xaxes(
            showgrid=False,
            row=row,
            col=1
        )

    fig.update_yaxes(
        tickformat=".0%",
        row=3,
        col=1
    )

    fig.update_yaxes(
        row=1,
        col=1,
        showticklabels=False,
        range=[0, 1.1]
    )

    fig.update_yaxes(
        row=2,
        col=1,
        showticklabels=False,
        range=[0, 1.1]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================================================================
    # PER PORT STATS
    # =========================================================================

    st.divider()

    st.subheader(f"{port} — Replay Statistics")

    per_port = summary.get(
        "per_port",
        {}
    ).get(port, {})

    sc1, sc2, sc3, sc4 = st.columns(4)

    sc1.metric(
        "S3 rate",
        f"{per_port.get('s3_rate', 0):.1%}"
    )

    sc2.metric(
        "Alerts fired",
        str(per_port.get("n_alerts", sum(alerts)))
    )

    sc3.metric(
        "Avg P(S3)",
        f"{per_port.get('mean_p_s3', np.mean(p_s3)):.1%}"
    )

    sc4.metric(
        "Avg λ(t)",
        f"{per_port.get('mean_lambda', np.mean(lam)):.4f}"
    )

    # =========================================================================
    # CROSS PORT LAMBDA
    # =========================================================================

    st.divider()

    st.subheader("Cross-Port Hawkes Intensity λ(t)")

    st.caption(
        "How cascade pressure builds and propagates "
        "across all 7 ports simultaneously"
    )

    with st.spinner("Loading cross-port data..."):

        series = get_hawkes_series_all(
            start_d,
            end_d
        )

    if series and "dates" in series:

        fig2 = go.Figure()

        colors_line = {
            "Rotterdam": "#ef4444",
            "Singapore": "#10b981",
            "Shanghai": "#f59e0b",
            "Nhava Sheva": "#f97316",
            "Busan": "#8b5cf6",
            "Hamburg": "#06b6d4",
            "Antwerp": "#64748b"
        }

        for p_name in PORTS:

            if p_name in series:

                vals = [
                    v if v is not None else 0
                    for v in series[p_name]
                ]

                fig2.add_trace(

                    go.Scatter(
                        x=series["dates"],
                        y=vals,

                        name=p_name,

                        line=dict(
                            color=colors_line.get(
                                p_name,
                                "#94a3b8"
                            ),
                            width=1.5
                        ),

                        connectgaps=True,

                        hovertemplate=
                        f"{p_name}: %{{y:.4f}}<extra></extra>",
                    )
                )

        fig2.update_layout(

            height=280,

            margin=dict(
                t=10,
                b=10,
                l=10,
                r=10
            ),

            plot_bgcolor="rgba(0,0,0,0)",

            paper_bgcolor="rgba(0,0,0,0)",

            legend=dict(
                orientation="h",
                y=-0.15,
                font=dict(size=10)
            ),

            xaxis=dict(showgrid=False),

            yaxis=dict(
                showgrid=True,
                gridcolor="#f1f5f9",
                title="λ(t)",
                title_font_size=11
            ),
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # =========================================================================
    # TIER-0 POLICY TIMELINE
    # =========================================================================

    if tier0 and len(tier0) == len(dates):

        st.divider()

        st.subheader("Tier-0 Policy Timeline")

        action_color_map = {
            "Strong": "#ef4444",
            "Escrow": "#f59e0b",
            "CRDT": "#10b981",
            "Stale": "#94a3b8",
            "Block": "#1e293b"
        }

        fig3 = go.Figure()

        for action in [

            "Strong",
            "Escrow",
            "CRDT",
            "Stale"

        ]:

            mask = [
                i
                for i, a in enumerate(tier0)
                if a == action
            ]

            if mask:

                fig3.add_trace(

                    go.Bar(
                        x=[dates[i] for i in mask],
                        y=[1] * len(mask),

                        name=action,

                        marker_color=
                        action_color_map[action],

                        marker_line_width=0,

                        hovertemplate=
                        f"{action}: %{{x}}<extra></extra>",
                    )
                )

        fig3.update_layout(

            barmode="stack",

            height=100,

            margin=dict(
                t=5,
                b=5,
                l=10,
                r=10
            ),

            yaxis=dict(
                showticklabels=False,
                range=[0, 1.1]
            ),

            xaxis=dict(showgrid=False),

            plot_bgcolor="rgba(0,0,0,0)",

            paper_bgcolor="rgba(0,0,0,0)",

            legend=dict(
                orientation="h",
                y=-0.4,
                font=dict(size=10)
            ),

            showlegend=True,
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        st.caption(
            "Strong = full consistency · "
            "Escrow = bounded · "
            "CRDT = eventual"
        )