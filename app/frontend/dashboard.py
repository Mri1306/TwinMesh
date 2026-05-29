# =============================================================================
# TwinMesh App · dashboard.py
# Main Streamlit entry point
# =============================================================================

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)

import streamlit as st

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="TwinMesh — Digital Twin Operations",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SHARED CSS
# =============================================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}

.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 8px;
}

.metric-label {
    font-size: 12px;
    color: #64748b;
    margin-bottom: 2px;
}

.metric-value {
    font-size: 24px;
    font-weight: 600;
}

.state-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
}

.s0 {
    background:#d1fae5;
    color:#065f46;
}

.s1 {
    background:#fef3c7;
    color:#92400e;
}

.s2 {
    background:#fed7aa;
    color:#9a3412;
}

.s3 {
    background:#fee2e2;
    color:#991b1b;
}

.s4 {
    background:#dbeafe;
    color:#1e40af;
}

.alert-box {
    background:#fef2f2;
    border:1px solid #fca5a5;
    border-radius:8px;
    padding:12px 14px;
    margin-bottom:8px;
}

.warn-box {
    background:#fffbeb;
    border:1px solid #fcd34d;
    border-radius:8px;
    padding:12px 14px;
    margin-bottom:8px;
}

.ok-box {
    background:#f0fdf4;
    border:1px solid #86efac;
    border-radius:8px;
    padding:12px 14px;
    margin-bottom:8px;
}

.action-strong {
    color:#dc2626;
    font-weight:600;
}

.action-escrow {
    color:#d97706;
    font-weight:600;
}

.action-crdt {
    color:#16a34a;
    font-weight:600;
}

.action-stale {
    color:#6b7280;
    font-weight:600;
}

/* Hide Streamlit default page navigation */
div[data-testid="stSidebarNav"] {
    display: none;
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPERS
# =============================================================================

STATE_LABELS = {
    "S0": "Normal",
    "S1": "Delayed",
    "S2": "Congested",
    "S3": "Disrupted",
    "S4": "Recovery",
}

STATE_EMOJI = {
    "S0": "🟢",
    "S1": "🟡",
    "S2": "🟠",
    "S3": "🔴",
    "S4": "🔵",
}

ACTION_COLOR = {
    "Strong": "action-strong",
    "Escrow": "action-escrow",
    "CRDT": "action-crdt",
    "Stale": "action-stale",
}


def state_badge(state: str) -> str:
    label = STATE_LABELS.get(state, state)
    return (
        f'<span class="state-badge {state.lower()}">'
        f'{state} {label}'
        f'</span>'
    )


def p_s3_color(p: float) -> str:
    if p >= 0.50:
        return "#ef4444"
    if p >= 0.30:
        return "#f59e0b"
    return "#10b981"


# =============================================================================
# LOAD MODELS
# =============================================================================

@st.cache_resource(show_spinner="Loading TwinMesh models...")
def load_all():

    from app.backend.prediction_service import load_models

    ok = load_models()
    return ok


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown("## 🚢 TwinMesh")

    st.caption(
        "Hawkes-Markov Digital Twin Framework "
        "for Predictive Supply Chain Intelligence"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📖 TwinMesh Overview",
            "🌐 Overview Dashboard",
            "🔮 Live Prediction",
            "📼 Replay Simulation",
            "🛡️ Policy Engine",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### System Status")

    loaded = load_all()

    if loaded:
        st.success("Models Loaded", icon="✅")
    else:
        st.error("Model Load Failed", icon="❌")

    st.markdown("")

    st.caption("Dataset")
    st.markdown("""
• 7 Global Ports  
• 2023–2026  
• IMF PortWatch  
• Multi-Source Intelligence
""")

    st.markdown("---")

    st.caption(
        "TwinMesh\n"
    )


# =============================================================================
# PAGE ROUTING
# =============================================================================

if "TwinMesh Overview" in page:

    from app.frontend.pages.info import render

    render()

elif "Overview" in page:

    from app.frontend.pages.overview import render

    render()

elif "Prediction" in page:

    from app.frontend.pages.live_prediction import render

    render()

elif "Replay" in page:

    from app.frontend.pages.replay_analysis import render

    render()

elif "Policy" in page:

    from app.frontend.pages.policy_view import render

    render()