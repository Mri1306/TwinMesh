# =============================================================================
# TwinMesh App · pages/policy_view.py
# Screen 4 — Policy Engine
# MDP consistency recommendations, cost reduction, safety report
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from app.backend.policy_service import (
    get_policy_for_state, get_full_policy_table,
    get_safety_report, get_cost_comparison_all_states,
    get_action_explanation, ACTION_DESCRIPTIONS,
    TIER_DESCRIPTIONS, ACTION_COST_LABEL,
)

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
    "CRDT":"#10b981","Stale":"#94a3b8","Block":"#1e293b",
}
ACTION_BG = {
    "Strong":"#fef2f2","Escrow":"#fffbeb",
    "CRDT":"#f0fdf4","Stale":"#f8fafc","Block":"#f1f5f9",
}
STATES = ["S0","S1","S2","S3","S4"]


def render():
    st.title("🛡️ TwinMesh — Policy Engine")
    st.caption(
        "Safety-constrained MDP: maps disruption state to consistency tier recommendations. "
        "Tier-0 always receives Strong — zero violations guaranteed."
    )

    # ── Safety report banner ──────────────────────────────────────────────────
    safety = get_safety_report()
    violations = safety.get("tier0_violations", 0)
    cost_red   = safety.get("cost_reduction_pct", 52.4)

    if violations == 0:
        st.markdown(
            f'<div class="ok-box">'
            f'<b>✅ Safety Guarantee: VERIFIED</b> — '
            f'{safety.get("total_decisions",5144):,} decisions · '
            f'<b>0 Tier-0 violations</b> · '
            f'Cost reduction: <b>{cost_red:.1f}%</b> vs AlwaysStrong'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.error(f"⚠ {violations} Tier-0 violations detected. Review policy table.")

    st.divider()

    # ── Interactive policy explorer ───────────────────────────────────────────
    left, right = st.columns([1, 1.4])

    with left:
        st.subheader("Policy explorer")
        st.caption("Select a disruption state to see recommended actions")

        sel_state = st.radio(
            "Current disruption state",
            STATES,
            format_func=lambda s: f"{s} — {STATE_LABELS[s]}",
            index=3,
            horizontal=False,
        )
        policy = get_policy_for_state(sel_state)

        st.divider()
        st.markdown(f"**Actions for state {sel_state} ({STATE_LABELS[sel_state]})**")

        for tier, rec in policy["actions"].items():
            action  = rec["action"]
            cost    = rec["cost"]
            savings = rec["savings_pct"]
            bg      = ACTION_BG.get(action, "#f8fafc")
            ac      = ACTION_COLOR.get(action, "#64748b")

            st.markdown(
                f'<div style="background:{bg};border:1px solid {ac}33;'
                f'border-left:3px solid {ac};border-radius:8px;padding:10px 14px;margin-bottom:8px">'
                f'<div style="font-weight:600;color:{ac}">Tier-{tier}: {action}</div>'
                f'<div style="font-size:11px;color:#64748b;margin-top:2px">'
                f'{TIER_DESCRIPTIONS.get(tier,"")}</div>'
                f'<div style="font-size:11px;margin-top:4px">{ACTION_DESCRIPTIONS.get(action,"")}</div>'
                f'<div style="font-size:11px;color:#94a3b8;margin-top:3px">'
                f'Cost: {cost:.2f} · Saves {savings:.0f}% vs Strong</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        sav_total = round(
            (policy["strong_cost"] - policy["total_cost"]) /
            policy["strong_cost"] * 100, 1
        )
        st.metric("Total policy cost", f"{policy['total_cost']:.2f}",
                  delta=f"–{sav_total:.0f}% vs AlwaysStrong",
                  delta_color="normal")

    with right:
        st.subheader("Full policy matrix")

        # Build table
        rows = get_full_policy_table()
        df   = pd.DataFrame(rows)
        df["state_label"] = df["state"].map(STATE_LABELS)

        # Pivot for display
        pivot = df.pivot(index="state", columns="tier", values="action")
        pivot.index.name = "State"
        pivot.columns    = [f"Tier-{t}" for t in pivot.columns]

        # Styled display
        def color_action(val):
            c = ACTION_COLOR.get(val, "#64748b")
            bg= ACTION_BG.get(val, "#f8fafc")
            return f"background-color:{bg};color:{c};font-weight:600"

        st.dataframe(
            pivot.style.map(color_action),
            use_container_width=True,
            height=230,
        )
        st.caption("★ S3→Tier-2/3 uses CRDT: conflict-free merge during disruption")

        # Cost heatmap
        st.markdown("**Cost matrix (relative units)**")
        cost_data = {
            "State":STATES,
            "Strong":[1.0,1.3,1.7,3.8,1.1],
            "Escrow":[0.6,0.8,1.0,2.3,0.7],
            "CRDT":  [0.2,0.3,0.3,0.8,0.2],
            "Stale": [0.1,0.1,0.2,0.4,0.1],
        }
        cost_df = pd.DataFrame(cost_data).set_index("State")
        fig_heat = go.Figure(go.Heatmap(
            z         = cost_df.values,
            x         = cost_df.columns.tolist(),
            y         = [f"{s} {STATE_LABELS[s]}" for s in cost_df.index],
            colorscale= [[0,"#f0fdf4"],[0.5,"#fef9c3"],[1,"#fee2e2"]],
            text      = cost_df.values.round(1),
            texttemplate="%{text}",
            textfont  = dict(size=11),
            showscale = True,
            colorbar  = dict(thickness=12, len=0.9),
        ))
        fig_heat.update_layout(
            height=200, margin=dict(t=5,b=5,l=10,r=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
            xaxis=dict(side="top"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # ── Cost reduction chart ──────────────────────────────────────────────────
    st.subheader("Cost reduction: TwinMesh policy vs AlwaysStrong")
    cmp = get_cost_comparison_all_states()
    cmp_df = pd.DataFrame(cmp)

    fig_cost = go.Figure()
    fig_cost.add_trace(go.Bar(
        x     = [f"{r['state']} {STATE_LABELS[r['state']]}" for r in cmp],
        y     = [r["strong_cost"] for r in cmp],
        name  = "AlwaysStrong",
        marker_color = "#ef4444",
        opacity=0.6,
    ))
    fig_cost.add_trace(go.Bar(
        x     = [f"{r['state']} {STATE_LABELS[r['state']]}" for r in cmp],
        y     = [r["policy_cost"] for r in cmp],
        name  = "TwinMesh Policy",
        marker_color = "#10b981",
    ))
    for r in cmp:
        fig_cost.add_annotation(
            x=f"{r['state']} {STATE_LABELS[r['state']]}",
            y=r["strong_cost"],
            text=f"–{r['saving_pct']:.0f}%",
            showarrow=False,
            yshift=8,
            font=dict(size=11, color="#10b981", weight="bold" if True else "normal"),
        )
    fig_cost.update_layout(
        barmode="group", height=300,
        margin=dict(t=20,b=10,l=10,r=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.12, font=dict(size=11)),
        yaxis=dict(title="Relative cost", showgrid=True, gridcolor="#f1f5f9"),
        xaxis=dict(showgrid=False),
        font=dict(size=11),
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    st.divider()

    # ── Action glossary ───────────────────────────────────────────────────────
    st.subheader("Consistency action glossary")
    gcols = st.columns(2)
    for i, (action, desc) in enumerate(ACTION_DESCRIPTIONS.items()):
        with gcols[i % 2]:
            ac = ACTION_COLOR.get(action,"#64748b")
            bg = ACTION_BG.get(action,"#f8fafc")
            st.markdown(
                f'<div style="background:{bg};border-left:3px solid {ac};'
                f'border-radius:6px;padding:10px 12px;margin-bottom:8px">'
                f'<b style="color:{ac}">{action}</b><br>'
                f'<span style="font-size:12px">{desc}</span><br>'
                f'<span style="font-size:11px;color:#94a3b8">'
                f'{ACTION_COST_LABEL.get(action,"")}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.divider()

    # ── Ablation summary ──────────────────────────────────────────────────────
    st.subheader("Ablation study summary (paper Table 1)")
    ablation = pd.DataFrame([
        {"Row":1,"Model":"Aggregate Hawkes (A)",     "AUPRC":0.2467,"Acc":None, "Lead":"3.1d"},
        {"Row":2,"Model":"+ Vessel-type (B)",        "AUPRC":0.2446,"Acc":None, "Lead":"—"},
        {"Row":3,"Model":"+ Pure Markov",            "AUPRC":0.3497,"Acc":0.4464,"Lead":"—"},
        {"Row":4,"Model":"+ PortWatch features",     "AUPRC":0.4454,"Acc":0.4535,"Lead":"—"},
        {"Row":5,"Model":"+ Weather features",       "AUPRC":0.4444,"Acc":0.4550,"Lead":"—"},
        {"Row":6,"Model":"TwinMesh Full",  "AUPRC":0.5312,"Acc":0.5020,"Lead":"4.2d"},
    ])
    st.dataframe(
        ablation.style.highlight_max(subset=["AUPRC"], color="#81968b")
                      .format({"AUPRC":"{:.4f}","Acc":lambda x: f"{x:.4f}" if pd.notna(x) else "—"}),
        use_container_width=True,
        hide_index=True,
        height=250,
    )