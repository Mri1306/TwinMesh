# =============================================================================
# TwinMesh App · pages/info.py
# Research Overview Page
# =============================================================================

import streamlit as st


def render():

    st.title("🚢 TwinMesh")
    st.subheader(
        "Hawkes-Markov Predictive Consistency Steering "
        "for Cross-Border Supply Chain Digital Twins"
    )

    st.markdown("---")

    st.markdown("""
### 🎯 Research Motivation

Modern maritime supply chains face increasingly frequent disruptions caused by:

- Port congestion
- Extreme weather
- Geopolitical conflicts
- Vessel delays
- Infrastructure bottlenecks

Most existing systems react after disruptions have already propagated.

TwinMesh provides:

- Early disruption intelligence
- Risk forecasting
- Digital twin orchestration
- Safety-aware operational recommendations
""")

    st.markdown("---")

    # =========================================================================
    # Architecture CSS
    # =========================================================================

    st.markdown("""
    <style>

    .arch-box{
        padding:16px;
        border-radius:12px;
        margin-bottom:8px;
        box-shadow:0 2px 8px rgba(0,0,0,0.08);
    }

    .arch-title{
        font-size:20px;
        font-weight:700;
        color:#111827 !important;
    }

    .arch-desc{
        font-size:15px;
        color:#374151 !important;
        margin-top:5px;
    }

    .arch-arrow{
        text-align:center;
        font-size:28px;
        font-weight:bold;
        margin:8px 0;
    }

    </style>
    """, unsafe_allow_html=True)

    st.subheader("🏗 TwinMesh Research Architecture")

    st.markdown("""

    <div class="arch-box"
         style="border-left:5px solid #3b82f6;background:#eff6ff;">
        <div class="arch-title">
            Layer 1 · Multi-Source Data Ingestion
        </div>
        <div class="arch-desc">
            IMF PortWatch · Weather Signals · Maritime Indicators ·
            Disruption Events
        </div>
    </div>

    <div class="arch-arrow">⬇</div>

    <div class="arch-box"
         style="border-left:5px solid #10b981;background:#f0fdf4;">
        <div class="arch-title">
            Layer 2 · Temporal Feature Engineering
        </div>
        <div class="arch-desc">
            Congestion Scores · Delay Proxies · Queue Buildup ·
            Throughput Ratios · State Labels (S0–S4)
        </div>
    </div>

    <div class="arch-arrow">⬇</div>

    <div class="arch-box"
         style="border-left:5px solid #f59e0b;background:#fffbeb;">
        <div class="arch-title">
            Layer 3 · Hawkes Cascade Intelligence
        </div>
        <div class="arch-desc">
            Cross-Port Influence Matrix · Event Intensities λ(t) ·
            Cascade Detection
        </div>
    </div>

    <div class="arch-arrow">⬇</div>

    <div class="arch-box"
         style="border-left:5px solid #ef4444;background:#fef2f2;">
        <div class="arch-title">
            Layer 4 · Markov State Intelligence
        </div>
        <div class="arch-desc">
            State Transition Modeling · Future State Forecasting ·
            Recovery Estimation
        </div>
    </div>

    <div class="arch-arrow">⬇</div>

    <div class="arch-box"
         style="border-left:5px solid #8b5cf6;background:#f5f3ff;">
        <div class="arch-title">
            Layer 5 · TwinMesh Predictive Core
        </div>
        <div class="arch-desc">
            Feature Fusion · XGBoost Predictor ·
            Risk Probability Estimation
        </div>
    </div>

    <div class="arch-arrow">⬇</div>

    <div class="arch-box"
         style="border-left:5px solid #06b6d4;background:#ecfeff;">
        <div class="arch-title">
            Layer 6 · Safety-Constrained MDP Policy Engine
        </div>
        <div class="arch-desc">
            Strong · Escrow · CRDT · Stale Reads ·
            Cost Optimization
        </div>
    </div>

    <div class="arch-arrow">⬇</div>

    <div class="arch-box"
         style="border-left:5px solid #64748b;background:#f8fafc;">
        <div class="arch-title">
            Layer 7 · Digital Twin Orchestration
        </div>
        <div class="arch-desc">
            Decision Execution · Consistency Steering ·
            Recovery Coordination
        </div>
    </div>

    <div class="arch-arrow">⬇</div>

    <div class="arch-box"
         style="border-left:5px solid #16a34a;background:#f0fdf4;">
        <div class="arch-title">
            Outputs
        </div>
        <div class="arch-desc">
            Early Alerts · Lead-Time Forecasts · Cost Reduction ·
            Zero Tier-0 Violations
        </div>
    </div>

    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🧠 Research Contributions")

    st.success(
        "1. Hawkes-Based Cross-Port Disruption Cascade Modeling"
    )

    st.success(
        "2. Markov-Based Supply Chain State Forecasting"
    )

    st.success(
        "3. Multi-Source Temporal Intelligence Fusion"
    )

    st.success(
        "4. Safety-Constrained MDP Decision Optimization"
    )

    st.success(
        "5. Digital Twin Orchestration Framework"
    )

    st.markdown("---")

    st.subheader("📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Ports", "7")
    c2.metric("Years", "2023–2026")
    c3.metric("States", "5")
    c4.metric("Sources", "Multi")

    st.info("""
**Data Sources**

• IMF PortWatch

• Maritime Activity Indicators

• Weather Signals

• Historical Disruption Events
""")

    st.markdown("---")

    st.subheader("🖥 Dashboard Guide")

    st.info("""
### 🌐 Overview Dashboard

Provides a high-level operational view of all ports.

Features:
- Port health monitoring
- Congestion analysis
- Global disruption overview
- System status
""")

    st.warning("""
### 🔮 Live Prediction

Runs the TwinMesh predictive pipeline.

Features:
- State prediction
- S3 disruption probability
- Hawkes intensity
- Model confidence
- Risk estimation
""")

    st.success("""
### 📼 Replay Simulation

Replay historical disruption events.

Features:
- Historical playback
- Timeline analysis
- Event evolution
- State transitions
""")

    st.error("""
### 🛡️ Policy Engine

Safety-constrained decision support.

Features:
- Consistency recommendations
- Cost reduction analysis
- Safety guarantees
- Policy matrix
""")

    st.markdown("---")

    st.subheader("⭐ Key Highlights")

    left, right = st.columns(2)

    with left:
        st.success("✓ Hawkes Cascade Modeling")
        st.success("✓ Markov Forecasting")
        st.success("✓ Multi-Source Intelligence")
        st.success("✓ XGBoost Risk Prediction")

    with right:
        st.success("✓ MDP Optimization")
        st.success("✓ Digital Twin Orchestration")
        st.success("✓ Cost Reduction")
        st.success("✓ Zero Tier-0 Violations")

    st.markdown("---")

    st.caption(
        "TwinMesh • Hawkes-Markov Digital Twin Framework"
    )