# =============================================================================
# TwinMesh App · pages/live_prediction.py
# Screen 2 — Live Prediction Panel
# User inputs observation → TwinMesh outputs prediction + MDP actions
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from app.backend.prediction_service import predict
from app.backend.policy_service import get_action_explanation

STATE_LABELS = {
    "S0":"Normal","S1":"Delayed","S2":"Congested",
    "S3":"Disrupted","S4":"Recovery",
}
STATE_COLOR = {
    "S0":"#10b981","S1":"#f59e0b","S2":"#f97316",
    "S3":"#ef4444","S4":"#3b82f6",
}
ACTION_COLOR = {
    "Strong":"#ef4444","Escrow":"#f59e0b",
    "CRDT":"#10b981","Stale":"#6b7280","Block":"#1e293b",
}
PORTS = ["Singapore","Rotterdam","Shanghai","Nhava Sheva","Busan","Hamburg","Antwerp"]
TIER_NAMES = {0:"Tier-0 Financial",1:"Tier-1 Inventory",
              2:"Tier-2 Routing",3:"Tier-3 Analytics"}


def render():
    st.title("🔮 TwinMesh — Live Prediction")
    st.caption(
        "Configure a port observation and run TwinMesh's full pipeline: "
        "Hawkes intensity → Markov prediction → MDP action."
    )

    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("Configure observation")

        port  = st.selectbox("Port", PORTS, index=1)
        state = st.selectbox(
            "Current disruption state",
            ["S0","S1","S2","S3","S4"],
            format_func=lambda s: f"{s} — {STATE_LABELS[s]}",
            index=2,
        )

        st.markdown("**PortWatch features**")
        cong  = st.slider("Congestion score", 0.0, 10.0, 2.8, 0.1)
        queue = st.slider("Queue buildup (7-day)", 0.0, 10.0, 3.2, 0.1)
        yoy   = st.slider("YoY % change", -50.0, 50.0, -8.5, 0.5)
        anom  = st.toggle("Anomaly flag", value=True)

        st.markdown("**Weather features**")
        weather = st.slider("Weather risk score", 0.0, 1.0, 0.6, 0.05)
        storm   = st.toggle("Storm alert", value=False)
        rain    = st.toggle("Heavy rain alert", value=True)

        st.markdown("**DWT flow**")
        dwt     = st.number_input("Net DWT flow (thousands)", value=-50.0, step=10.0)

        run_btn = st.button("🚀 Run TwinMesh prediction", type="primary",
                            use_container_width=True)

    with right:
        if run_btn or "last_pred" in st.session_state:
            with st.spinner("Running TwinMesh pipeline..."):
                if run_btn:
                    result = predict(
                        port               = port,
                        current_state      = state,
                        congestion_score   = cong,
                        net_dwt_flow       = dwt * 1000,
                        queue_buildup_7d   = queue,
                        yoy_pct_change     = yoy,
                        anomaly_flag       = int(anom),
                        weather_risk_score = weather,
                        storm_alert        = int(storm),
                        heavy_rain_alert   = int(rain),
                    )
                    st.session_state["last_pred"] = result
                else:
                    result = st.session_state["last_pred"]

            p_s3     = result["p_s3"]
            pred_st  = result["predicted_state"]
            alert    = result["alert"]
            lam      = result["hawkes_lambda"]
            delta    = result["hawkes_delta"]
            probs    = result["state_probs"]
            actions  = result["actions"]
            conf     = result["confidence"]

            # Alert box
            if alert:
                st.markdown(
                    f'<div class="alert-box">'
                    f'<b>⚠ Early Warning Alert Fired</b><br>'
                    f'P(S3) = <b>{p_s3:.1%}</b> exceeds threshold (30%).<br>'
                    f'Estimated lead time: ~4.2 days before S3 onset.'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="ok-box">✅ No alert — P(S3) = {p_s3:.1%} (below 30%)</div>',
                    unsafe_allow_html=True
                )

            # Key metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted next state", pred_st,
                      delta=STATE_LABELS.get(pred_st,""),
                      delta_color="off")
            m2.metric("P(S3)", f"{p_s3:.1%}",
                      delta="ALERT" if alert else "OK",
                      delta_color="inverse" if alert else "off")
            m3.metric("Hawkes λ(t)", f"{lam:.4f}",
                      delta=f"Δ {delta:+.4f}",
                      delta_color="inverse" if delta > 0 else "normal")

            # State probability chart
            st.markdown("**State probability distribution**")
            fig = go.Figure(go.Bar(
                x      = list(probs.values()),
                y      = [f"{s} {STATE_LABELS[s]}" for s in probs],
                orientation = "h",
                marker_color= [STATE_COLOR.get(s,"#94a3b8") for s in probs],
                text   = [f"{v:.1%}" for v in probs.values()],
                textposition = "outside",
            ))
            fig.update_layout(
                height=200, margin=dict(l=0,r=60,t=5,b=5),
                xaxis=dict(range=[0,1.1], showgrid=False),
                yaxis=dict(showgrid=False),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=11),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Hawkes intensity gauge
            st.markdown("**Hawkes cascade intensity**")
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = lam,
                delta = {"reference": 0.08, "valueformat": ".4f"},
                gauge = {
                    "axis": {"range": [0, 0.45], "tickformat": ".2f"},
                    "bar":  {"color": "#3b82f6"},
                    "steps": [
                        {"range": [0, 0.15],  "color": "#f0fdf4"},
                        {"range": [0.15, 0.30],"color": "#fef9c3"},
                        {"range": [0.30, 0.45],"color": "#fef2f2"},
                    ],
                    "threshold": {
                        "line":  {"color": "#ef4444", "width": 2},
                        "thickness": 0.75,
                        "value": 0.30,
                    },
                },
                title={"text":"λ(t) — Baseline μ=0.08"},
                number={"valueformat":".4f"},
            ))
            fig_gauge.update_layout(
                height=220, margin=dict(t=30,b=0,l=20,r=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # MDP actions
            st.markdown("**MDP consistency policy recommendations**")
            for tier_idx, rec in actions.items():
                action = rec["action"]
                cost   = rec["cost"]
                acolor = ACTION_COLOR.get(action, "#64748b")
                tier_name = TIER_NAMES.get(tier_idx, f"Tier-{tier_idx}")
                exp    = get_action_explanation(action)

                with st.expander(f"Tier-{tier_idx}: **{action}** — {tier_name}"):
                    c1, c2 = st.columns([2,1])
                    c1.markdown(f"**{exp['description']}**")
                    c1.caption(exp["cost_label"])
                    c2.metric("Relative cost", f"{cost:.2f}")

        else:
            st.info(
                "👈 Configure an observation in the left panel and click "
                "**Run TwinMesh prediction** to see the output.",
                icon="ℹ️"
            )
            st.markdown("**What this screen demonstrates:**")
            st.markdown("""
- **Hawkes intensity λ(t)** — how much cascade pressure is building
- **State prediction** — where the port is headed (Markov)
- **P(S3)** — probability of imminent disruption
- **Early warning alert** — fires 4.2 days before S3 onset on average
- **MDP policy** — what consistency level each tier of operations should use
""")