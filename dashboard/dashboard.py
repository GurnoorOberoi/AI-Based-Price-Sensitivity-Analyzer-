import sys
import os
sys.path.append(os.path.abspath("."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data_loader import load_data
from src.modeling import train_model, calculate_elasticity
from src.simulation import simulate_prices


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Price Sensitivity Analyzer",
    page_icon="💰",
    layout="wide"
)

# =====================================================
# PREMIUM ANIMATED BACKGROUND (SUBTLE + CLASSY)
# =====================================================
st.markdown("""
<style>

/* ===============================
   AURORA GLASS BACKGROUND
   =============================== */
.stApp {
    background:
        radial-gradient(circle at 20% 20%, rgba(0,255,255,0.18), transparent 40%),
        radial-gradient(circle at 80% 30%, rgba(255,0,255,0.15), transparent 45%),
        radial-gradient(circle at 50% 80%, rgba(0,150,255,0.18), transparent 50%),
        linear-gradient(180deg, #05060a, #02030a);
    animation: auroraMove 30s ease-in-out infinite;
    color: white;
}

/* Smooth floating motion */
@keyframes auroraMove {
    0% { background-position: 0% 0%, 100% 0%, 50% 100%, 0% 0%; }
    50% { background-position: 20% 30%, 80% 20%, 60% 70%, 0% 0%; }
    100% { background-position: 0% 0%, 100% 0%, 50% 100%, 0% 0%; }
}

/* ===============================
   SIDEBAR (GLASS EFFECT)
   =============================== */
section[data-testid="stSidebar"] {
    background: rgba(10, 12, 20, 0.75);
    backdrop-filter: blur(14px);
    border-right: 1px solid rgba(0,255,255,0.2);
}

/* ===============================
   HEADINGS
   =============================== */
h1, h2, h3 {
    color: #e8faff;
    text-shadow: 0 0 12px rgba(0,255,255,0.35);
}

/* ===============================
   KPI METRIC CARDS
   =============================== */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 0 30px rgba(0,255,255,0.18);
}

/* ===============================
   PLOT BACKGROUND (CONSISTENT)
   =============================== */
.js-plotly-plot .plotly {
    background: rgba(0,0,0,0.9) !important;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HERO
# =====================================================
st.markdown("""
<h1 style="text-align:center;font-size:48px;">💰 AI-Based Price Sensitivity Analyzer</h1>
<p style="text-align:center;color:#c7ecee;font-size:18px;">
Strategic pricing intelligence • Demand modeling • Revenue optimization
</p>
""", unsafe_allow_html=True)

st.divider()

# =====================================================
# FEATURES
# =====================================================
FEATURES = [
    "PRICE",
    "IS_WEEKEND",
    "IS_SCHOOLBREAK",
    "AVERAGE_TEMPERATURE",
    "IS_OUTDOOR"
]

# =====================================================
# LOAD & TRAIN
# =====================================================
with st.spinner("Training pricing intelligence model..."):
    df = load_data()
    model, X_test = train_model(df, FEATURES)
    elasticity = calculate_elasticity(df)
    sim_df = simulate_prices(df, model, X_test)

best = sim_df.loc[sim_df["REVENUE"].idxmax()]

# =====================================================
# STRATEGIC KPIs (STATIC – BUSINESS CORRECT)
# =====================================================
st.subheader("📌 Strategic Pricing Benchmarks")

k1, k2, k3 = st.columns(3)
k1.metric("📉 Price Elasticity", round(elasticity, 2))
k2.metric("💰 Optimal Price (₹)", round(best["PRICE"], 2))
k3.metric("📈 Maximum Revenue (₹)", round(best["REVENUE"], 2))

st.divider()

# =====================================================
# LIVE PRICE IMPACT (MOVED UP 🔥)
# =====================================================
st.subheader("🎯 Live Price Impact Analysis")

price = st.slider(
    "Simulate Product Price (₹)",
    float(df["PRICE"].min()),
    float(df["PRICE"].max()),
    float(best["PRICE"])
)

base = X_test.mean().to_dict()
base["PRICE"] = price

pred_demand = model.predict(pd.DataFrame([base]))[0]
pred_revenue = price * pred_demand
revenue_gap = pred_revenue - best["REVENUE"]

c1, c2, c3 = st.columns(3)
c1.metric("Predicted Demand", round(pred_demand, 1))
c2.metric("Predicted Revenue (₹)", round(pred_revenue, 2))
c3.metric(
    "Revenue Gap vs Optimal (₹)",
    round(revenue_gap, 2),
    delta=round(revenue_gap, 2)
)

st.divider()

# =====================================================
# CUSTOMER BEHAVIOR
# =====================================================
st.subheader("🧠 Customer Demand Behavior")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(
        df,
        x="QUANTITY",
        nbins=30,
        title="Distribution of Customer Demand",
        template="plotly_dark"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(
        df,
        x="PRICE",
        y="QUANTITY",
        opacity=0.6,
        title="Price vs Demand Relationship",
        template="plotly_dark"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# =====================================================
# AI EXPLAINABILITY
# =====================================================
st.subheader("🤖 How the AI Makes Decisions")

fi_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)

fig_fi = px.bar(
    fi_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance (Random Forest)",
    template="plotly_dark"
)

st.plotly_chart(fig_fi, use_container_width=True)

st.divider()

# =====================================================
# REVENUE OPTIMIZATION CURVE
# =====================================================
st.subheader("💸 Revenue Optimization Curve")

fig_rev = px.line(
    sim_df,
    x="PRICE",
    y="REVENUE",
    markers=True,
    title="Price vs Revenue Curve",
    template="plotly_dark"
)
st.plotly_chart(fig_rev, use_container_width=True)

# =====================================================
# DATA TABLE
# =====================================================
with st.expander("📄 View Full Price Simulation Table"):
    st.dataframe(sim_df)

st.success("🚀 Enterprise-Grade AI Pricing Dashboard Ready")

