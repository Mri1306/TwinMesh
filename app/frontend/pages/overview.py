# =============================================================================
# TwinMesh App · pages/overview.py
# Screen 1 — Overview Dashboard
# Shows all 7 ports, current state, P(S3), alerts, key metrics
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from app.backend.prediction_service import get_all_port_status, ModelStore
from app.backend.replay_service import get_alert_feed, get_replay_summary

STATE_LABELS = {
    "S0":"Normal","S1":"Delayed","S2":"Congested",
    "S3":"Disrupted","S4":"Recovery",
}
STATE_COLOR  = {
    "S0":"#10b981","S1":"#f59e0b","S2":"#f97316",
    "S3":"#ef4444","S4":"#3b82f6",
}
PORT_COORDS  = {
    "Singapore":   (1.3521,  103.8198),
    "Rotterdam":   (51.9244,   4.4777),
    "Shanghai":    (31.2304,  121.4737),
    "Nhava Sheva": (18.9500,   72.9500),
    "Busan":       (35.1796,  129.0756),
    "Hamburg":     (53.5753,   9.8689),
    "Antwerp":     (51.2213,   4.4051),
}


def render():
    st.title("🌐 TwinMesh — Global Port Overview")
    st.caption("Real-time disruption intelligence across 7 global supply chain nodes")

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading port status..."):
        port_status = get_all_port_status()
        summary     = get_replay_summary()
        alerts      = get_alert_feed(limit=8)

    # ── Key metrics row ───────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    n_disrupted = sum(1 for p in port_status if p["current_state"] == "S3")
    n_alerts    = sum(1 for p in port_status if p["alert"])
    avg_p_s3    = np.mean([p["p_s3"] for p in port_status])
    avg_lambda  = np.mean([p["hawkes_lambda"] for p in port_status])

    col1.metric("Ports Disrupted (S3)", f"{n_disrupted}/7",
                delta="Active cascades" if n_disrupted else None,
                delta_color="inverse")
    col2.metric("Active Alerts", str(n_alerts),
                delta="P(S3) > 30%",
                delta_color="inverse" if n_alerts else "off")
    col3.metric("Avg P(S3)", f"{avg_p_s3:.1%}")
    col4.metric("Avg Hawkes λ(t)", f"{avg_lambda:.4f}")
    col5.metric("Cost Reduction", "52.4%", delta="vs AlwaysStrong",
                delta_color="normal")

    st.divider()

    # ── Map + alerts two-column ───────────────────────────────────────────────
    map_col, alert_col = st.columns([2, 1])

    with map_col:
        st.subheader("Port Status Map")
        lats, lons, names, colors, sizes, hover = [], [], [], [], [], []
        for p in port_status:
            port = p["port"]
            coord = PORT_COORDS.get(port, (0, 0))
            lats.append(coord[0])
            lons.append(coord[1])
            names.append(port)
            colors.append(STATE_COLOR.get(p["current_state"], "#94a3b8"))
            sizes.append(30 if p["alert"] else 18)
            hover.append(
                f"<b>{port}</b><br>"
                f"State: {p['current_state']} {STATE_LABELS.get(p['current_state'],'')}<br>"
                f"P(S3): {p['p_s3']:.1%}<br>"
                f"λ(t): {p['hawkes_lambda']:.4f}<br>"
                f"Alert: {'⚠ YES' if p['alert'] else 'No'}"
            )

        fig = go.Figure(go.Scattergeo(
            lat            = lats,
            lon            = lons,
            text           = names,
            customdata     = hover,
            hovertemplate  = "%{customdata}<extra></extra>",
            mode           = "markers+text",
            textposition   = "top center",
            textfont       = dict(size=10, color="#1e293b"),
            marker         = dict(
                size        = sizes,
                color       = colors,
                line        = dict(width=1.5, color="#ffffff"),
                symbol      = "circle",
            ),
        ))
        fig.update_layout(
            geo=dict(
                showland=True, landcolor="#f8fafc",
                showocean=True, oceancolor="#dbeafe",
                showcountries=True, countrycolor="#cbd5e1",
                showcoastlines=True, coastlinecolor="#94a3b8",
                projection_type="natural earth",
                showframe=False,
            ),
            margin=dict(l=0,r=0,t=0,b=0),
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Legend
        lcols = st.columns(5)
        for i, (s, label) in enumerate(STATE_LABELS.items()):
            lcols[i].markdown(
                f'<div style="display:inline-block;width:10px;height:10px;'
                f'border-radius:50%;background:{STATE_COLOR[s]};margin-right:4px"></div>'
                f'<span style="font-size:11px">{s} {label}</span>',
                unsafe_allow_html=True
            )

    with alert_col:
        st.subheader("🚨 Alert Centre")
        active_alerts = [p for p in port_status if p["alert"]]
        if active_alerts:
            for p in active_alerts:
                urgency = "🔴" if p["p_s3"] >= 0.5 else "🟡"
                st.markdown(
                    f'<div class="alert-box">'
                    f'<b>{urgency} {p["port"]}</b><br>'
                    f'State: {p["current_state"]} · P(S3): <b>{p["p_s3"]:.1%}</b><br>'
                    f'λ(t)={p["hawkes_lambda"]:.4f} · Tier-0: <b>{p["actions"][0]["action"]}</b>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="ok-box">✅ <b>No active alerts</b><br>'
                'All ports operating within normal thresholds.</div>',
                unsafe_allow_html=True
            )

        st.caption("Alert threshold: P(S3) > 30%")

    st.divider()

    # ── Port cards grid ───────────────────────────────────────────────────────
    st.subheader("Port Status Detail")
    cols = st.columns(4)
    for i, p in enumerate(port_status):
        with cols[i % 4]:
            state = p["current_state"]
            color = STATE_COLOR.get(state, "#94a3b8")
            p_s3  = p["p_s3"]
            bar_color = "#ef4444" if p_s3>=0.5 else "#f59e0b" if p_s3>=0.3 else "#10b981"

            st.markdown(
                f'<div class="metric-card" style="border-left:3px solid {color}">'
                f'<div style="font-weight:600;margin-bottom:4px">{p["port"]}</div>'
                f'<div style="font-size:11px;color:#64748b">State: '
                f'<span style="font-weight:600;color:{color}">{state} {STATE_LABELS.get(state,"")}</span></div>'
                f'<div style="font-size:11px;margin-top:2px">Predicted: '
                f'<b style="color:{STATE_COLOR.get(p["predicted_state"],color)}">'
                f'{p["predicted_state"]}</b></div>'
                f'<div style="margin-top:6px">'
                f'<div style="font-size:10px;color:#94a3b8;margin-bottom:2px">P(S3): {p_s3:.1%}</div>'
                f'<div style="height:5px;background:#f1f5f9;border-radius:3px">'
                f'<div style="width:{p_s3*100:.0f}%;height:100%;background:{bar_color};border-radius:3px"></div>'
                f'</div></div>'
                f'<div style="font-size:10px;color:#94a3b8;margin-top:4px">λ(t)={p["hawkes_lambda"]:.4f}</div>'
                f'{"<div style=\"color:#ef4444;font-size:10px;margin-top:2px\">⚠ ALERT</div>" if p["alert"] else ""}'
                f'</div>',
                unsafe_allow_html=True
            )

    st.divider()

    # ── AUPRC comparison chart ────────────────────────────────────────────────
    st.subheader("Model Performance — S3 AUPRC")
    auprc_data = pd.DataFrame({
        "Model": ["Twinmesh","RTTThreshold","StaticPACELC","Prophet","AlwaysStrong","AlwaysEventual","PoissonTrigger"],
        "AUPRC": [0.5167, 0.4249, 0.3621, 0.2833, 0.2463, 0.2463, 0.2463],
        "Type":  ["TwinMesh","Baseline","Baseline","Baseline","Baseline","Baseline","Baseline"],
    })
    colors_map = {"TwinMesh":"#10b981","Baseline":"#94a3b8"}
    fig2 = px.bar(
        auprc_data, x="Model", y="AUPRC", color="Type",
        color_discrete_map=colors_map,
        text="AUPRC",
        height=300,
    )
    fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig2.update_layout(
        showlegend=True,
        margin=dict(t=10,b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0.2, 0.62]),
        font=dict(size=11),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "★ TwinMesh additionally provides: 4.2d early warning · 52.4% cost reduction · "
        "Zero Tier-0 violations — capabilities absent from all baselines."
    )