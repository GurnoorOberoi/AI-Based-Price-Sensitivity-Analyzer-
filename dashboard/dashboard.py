# import sys
# import os
# sys.path.append(os.path.abspath("."))

# import streamlit as st
# import pandas as pd
# import plotly.express as px

# from src.data_loader import load_data
# from src.modeling import train_model, calculate_elasticity
# from src.simulation import simulate_prices


# # ----------------------------------
# # PAGE CONFIG
# # ----------------------------------
# st.set_page_config(
#     page_title="AI Price Sensitivity Dashboard",
#     page_icon="💰",
#     layout="wide"
# )

# st.markdown("""
# <style>

# /* ----------- GLOBAL BACKGROUND ----------- */
# .stApp {
#     background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
#     color: #ffffff;
# }

# /* ----------- SIDEBAR ----------- */
# section[data-testid="stSidebar"] {
#     background: rgba(0, 0, 0, 0.65);
#     backdrop-filter: blur(10px);
# }

# /* ----------- HEADERS ----------- */
# h1, h2, h3 {
#     color: #00f2fe;
#     font-weight: 700;
# }

# /* ----------- METRIC CARDS ----------- */
# div[data-testid="metric-container"] {
#     background: rgba(255, 255, 255, 0.1);
#     border-radius: 18px;
#     padding: 18px;
#     box-shadow: 0 0 25px rgba(0, 242, 254, 0.35);
#     border: 1px solid rgba(255, 255, 255, 0.15);
# }

# /* ----------- DATAFRAMES ----------- */
# div[data-testid="stDataFrame"] {
#     background: rgba(255,255,255,0.05);
#     border-radius: 12px;
# }

# /* ----------- BUTTONS ----------- */
# .stButton>button {
#     background: linear-gradient(135deg, #00f2fe, #4facfe);
#     color: black;
#     font-weight: bold;
#     border-radius: 12px;
#     padding: 10px 20px;
#     border: none;
# }

# /* ----------- SLIDERS ----------- */
# .stSlider > div {
#     color: #00f2fe;
# }

# </style>
# """, unsafe_allow_html=True)


# # st.title("💰 AI-Based Price Sensitivity Analyzer")
# st.markdown("""
# <h1 style='text-align:center; font-size:48px;'>
# 💰 AI-Based Price Sensitivity Analyzer
# </h1>
# <p style='text-align:center; font-size:18px; color:#c7ecee;'>
# Data-driven pricing intelligence powered by Machine Learning
# </p>
# """, unsafe_allow_html=True)


# st.markdown("### Smart pricing decisions powered by Machine Learning")

# # ----------------------------------
# # FEATURE LIST (MUST MATCH MODEL)
# # ----------------------------------
# FEATURES = [
#     "PRICE",
#     "IS_WEEKEND",
#     "IS_SCHOOLBREAK",
#     "AVERAGE_TEMPERATURE",
#     "IS_OUTDOOR"
# ]

# # ----------------------------------
# # LOAD DATA & TRAIN MODEL
# # ----------------------------------
# with st.spinner("Loading data & training model..."):
#     df = load_data()
#     model, X_test = train_model(df, FEATURES)
#     elasticity = calculate_elasticity(df)
#     sim_df = simulate_prices(df, model, X_test)

# # ----------------------------------
# # KPI SECTION
# # ----------------------------------
# best = sim_df.loc[sim_df["REVENUE"].idxmax()]

# col1, col2, col3 = st.columns(3)
# col1.metric("📉 Price Elasticity", round(elasticity, 2))
# col2.metric("💰 Optimal Price (₹)", round(best["PRICE"], 2))
# col3.metric("📈 Max Revenue (₹)", round(best["REVENUE"], 2))

# st.divider()

# # ----------------------------------
# # LIVE PRICE SIMULATION
# # ----------------------------------
# st.subheader("🎯 Live Price Simulation")

# price_input = st.slider(
#     "Select Product Price (₹)",
#     float(df["PRICE"].min()),
#     float(df["PRICE"].max()),
#     float(best["PRICE"])
# )

# base = X_test.mean().to_dict()
# base["PRICE"] = price_input

# predicted_demand = model.predict(pd.DataFrame([base]))[0]
# predicted_revenue = price_input * predicted_demand

# col4, col5 = st.columns(2)
# col4.metric("Predicted Demand", round(predicted_demand, 1))
# col5.metric("Predicted Revenue (₹)", round(predicted_revenue, 2))

# st.divider()

# # ----------------------------------
# # INTERACTIVE CHARTS
# # ----------------------------------
# st.subheader("📊 Revenue Optimization Curve")

# fig1 = px.line(
#     sim_df,
#     x="PRICE",
#     y="REVENUE",
#     markers=True,
#     title="Price vs Revenue Curve"
# )
# st.plotly_chart(fig1, use_container_width=True)

# st.subheader("📦 Price vs Demand")

# fig2 = px.scatter(
#     df,
#     x="PRICE",
#     y="QUANTITY",
#     opacity=0.6,
#     title="Customer Demand Sensitivity"
# )
# st.plotly_chart(fig2, use_container_width=True)

# # ----------------------------------
# # DATA VIEW
# # ----------------------------------
# with st.expander("📄 View Price Simulation Results"):
#     st.dataframe(sim_df)

# st.success("Dashboard loaded successfully 🚀")

# import sys
# import os
# sys.path.append(os.path.abspath("."))

# import streamlit as st
# import pandas as pd
# import plotly.express as px

# from src.data_loader import load_data
# from src.modeling import train_model, calculate_elasticity
# from src.simulation import simulate_prices


# # ======================================================
# # PAGE CONFIG
# # ======================================================
# st.set_page_config(
#     page_title="AI Price Sensitivity Analyzer",
#     page_icon="💰",
#     layout="wide"
# )

# # ======================================================
# # PREMIUM UI STYLING
# # ======================================================
# st.markdown("""
# <style>

# .stApp {
#     background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
#     color: #ffffff;
# }

# section[data-testid="stSidebar"] {
#     background: rgba(0,0,0,0.65);
#     backdrop-filter: blur(12px);
# }

# h1, h2, h3 {
#     color: #00f2fe;
#     font-weight: 700;
# }

# div[data-testid="metric-container"] {
#     background: rgba(255,255,255,0.08);
#     border-radius: 18px;
#     padding: 18px;
#     box-shadow: 0 0 25px rgba(0,242,254,0.35);
#     border: 1px solid rgba(255,255,255,0.15);
# }

# .stButton>button {
#     background: linear-gradient(135deg, #00f2fe, #4facfe);
#     color: black;
#     font-weight: bold;
#     border-radius: 12px;
#     border: none;
#     padding: 10px 20px;
# }

# </style>
# """, unsafe_allow_html=True)

# # ======================================================
# # HEADER
# # ======================================================
# st.markdown("""
# <h1 style='text-align:center; font-size:48px;'>
# 💰 AI-Based Price Sensitivity Analyzer
# </h1>
# <p style='text-align:center; font-size:18px; color:#c7ecee;'>
# Machine-Learning powered pricing intelligence for revenue optimization
# </p>
# """, unsafe_allow_html=True)

# st.divider()

# # ======================================================
# # FEATURE LIST
# # ======================================================
# FEATURES = [
#     "PRICE",
#     "IS_WEEKEND",
#     "IS_SCHOOLBREAK",
#     "AVERAGE_TEMPERATURE",
#     "IS_OUTDOOR"
# ]

# # ======================================================
# # LOAD DATA & TRAIN MODEL
# # ======================================================
# with st.spinner("🔄 Loading data & training model..."):
#     df = load_data()
#     model, X_test = train_model(df, FEATURES)
#     elasticity = calculate_elasticity(df)
#     sim_df = simulate_prices(df, model, X_test)

# # ======================================================
# # KPI SECTION
# # ======================================================
# best = sim_df.loc[sim_df["REVENUE"].idxmax()]

# col1, col2, col3 = st.columns(3)
# col1.metric("📉 Price Elasticity", round(elasticity, 2))
# col2.metric("💰 Optimal Price (₹)", round(best["PRICE"], 2))
# col3.metric("📈 Maximum Revenue (₹)", round(best["REVENUE"], 2))

# st.divider()

# # ======================================================
# # LIVE PRICE SIMULATION
# # ======================================================
# st.subheader("🎯 Live Price Simulation")

# price_input = st.slider(
#     "Select Product Price (₹)",
#     float(df["PRICE"].min()),
#     float(df["PRICE"].max()),
#     float(best["PRICE"])
# )

# base_features = X_test.mean().to_dict()
# base_features["PRICE"] = price_input

# predicted_demand = model.predict(pd.DataFrame([base_features]))[0]
# predicted_revenue = price_input * predicted_demand

# c1, c2 = st.columns(2)
# c1.metric("📦 Predicted Demand", round(predicted_demand, 1))
# c2.metric("💵 Predicted Revenue (₹)", round(predicted_revenue, 2))

# st.divider()

# # ======================================================
# # REVENUE OPTIMIZATION CURVE (CORE INSIGHT)
# # ======================================================
# st.subheader("📊 Revenue Optimization Curve")

# fig_rev = px.line(
#     sim_df,
#     x="PRICE",
#     y="REVENUE",
#     markers=True,
#     title="Price vs Revenue"
# )
# st.plotly_chart(fig_rev, use_container_width=True)

# # ======================================================
# # FEATURE IMPORTANCE (MODEL EXPLAINABILITY)
# # ======================================================
# st.subheader("🔍 What Drives Customer Demand?")

# fi_df = pd.DataFrame({
#     "Feature": FEATURES,
#     "Importance": model.feature_importances_
# }).sort_values(by="Importance", ascending=False)

# fig_fi = px.bar(
#     fi_df,
#     x="Importance",
#     y="Feature",
#     orientation="h",
#     title="Feature Importance (Random Forest)"
# )

# st.plotly_chart(fig_fi, use_container_width=True)

# # ======================================================
# # SHAP ANALYSIS (ADVANCED – COLLAPSED)
# # ======================================================
# with st.expander("🧠 Model Explainability (SHAP Analysis)"):
#     st.image("outputs/plots/shap_summary.png", use_column_width=True)
#     st.caption(
#         "SHAP values explain how each feature contributes to demand predictions. "
#         "Higher impact features influence pricing decisions more strongly."
#     )

# # ======================================================
# # DATA TABLE (OPTIONAL)
# # ======================================================
# with st.expander("📄 View Price Simulation Table"):
#     st.dataframe(sim_df)

# # ======================================================
# # FOOTER
# # ======================================================
# st.divider()
# st.markdown("""
# <div style='text-align:center; color:#c7ecee; font-size:14px;'>
# ⚠️ <strong>Disclaimer:</strong> This project is for educational & analytical purposes only.  
# Not intended as real-world pricing or financial advice.
# </div>
# """, unsafe_allow_html=True)

# st.success("🚀 Dashboard loaded successfully!")


# import sys
# import os
# sys.path.append(os.path.abspath("."))

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.io as pio
# pio.templates.default = "plotly_dark"


# from src.data_loader import load_data
# from src.modeling import train_model, calculate_elasticity
# from src.simulation import simulate_prices


# # =====================================================
# # PAGE CONFIG
# # =====================================================
# st.set_page_config(
#     page_title="AI Price Sensitivity Intelligence",
#     page_icon="💰",
#     layout="wide"
# )

# # =====================================================
# # ADVANCED UI STYLING (PARITY WITH MICRO-TREND)
# # =====================================================
# st.markdown("""
# <style>
# .stApp {
#     background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
#     color: white;
# }

# section[data-testid="stSidebar"] {
#     background: rgba(0,0,0,0.7);
#     backdrop-filter: blur(10px);
# }

# h1, h2, h3 {
#     color: #00f2fe;
#     font-weight: 700;
# }

# div[data-testid="metric-container"] {
#     background: rgba(255,255,255,0.08);
#     border-radius: 16px;
#     padding: 18px;
#     border: 1px solid rgba(255,255,255,0.15);
#     box-shadow: 0 0 25px rgba(0,242,254,0.3);
# }
# </style>
# """, unsafe_allow_html=True)

# # =====================================================
# # HEADER
# # =====================================================
# st.markdown("""
# <h1 style="text-align:center; font-size:46px;">
# 💰 AI Price Sensitivity Intelligence System
# </h1>
# <p style="text-align:center; color:#c7ecee; font-size:18px;">
# From historical sales → optimal pricing → revenue maximization
# </p>
# """, unsafe_allow_html=True)

# st.divider()

# # =====================================================
# # FEATURES
# # =====================================================
# FEATURES = [
#     "PRICE",
#     "IS_WEEKEND",
#     "IS_SCHOOLBREAK",
#     "AVERAGE_TEMPERATURE",
#     "IS_OUTDOOR"
# ]

# # =====================================================
# # LOAD & TRAIN
# # =====================================================
# with st.spinner("Training pricing intelligence model..."):
#     df = load_data()
#     model, X_test = train_model(df, FEATURES)
#     elasticity = calculate_elasticity(df)
#     sim_df = simulate_prices(df, model, X_test)

# best = sim_df.loc[sim_df["REVENUE"].idxmax()]

# # =====================================================
# # EXECUTIVE OVERVIEW (LIKE MICRO-TREND)
# # =====================================================
# st.subheader("📌 Executive Pricing Overview")

# c1, c2, c3, c4 = st.columns(4)
# c1.metric("📉 Price Elasticity", round(elasticity, 2))
# c2.metric("💰 Optimal Price (₹)", round(best["PRICE"], 2))
# c3.metric("📈 Max Revenue (₹)", round(best["REVENUE"], 2))
# c4.metric("📦 Expected Demand", round(best["PREDICTED_DEMAND"], 1))

# st.info(
#     "💡 **Insight:** Demand is sensitive to price changes. "
#     "Small price increases beyond the optimal point reduce revenue significantly."
# )

# st.divider()

# # =====================================================
# # TABS (LIKE MICRO-TREND SEGMENTATION)
# # =====================================================
# tab1, tab2, tab3, tab4 = st.tabs([
#     "📊 Price Sensitivity",
#     "🎯 Revenue Optimization",
#     "🧠 Model Intelligence",
#     "📄 Data Explorer"
# ])

# # -----------------------------------------------------
# # TAB 1: PRICE SENSITIVITY
# # -----------------------------------------------------
# with tab1:
#     st.subheader("Customer Demand Behavior")

#     fig1 = px.scatter(
#         df,
#         x="PRICE",
#         y="QUANTITY",
#         opacity=0.6,
#         title="Price vs Customer Demand"
#     )
#     st.plotly_chart(fig1, use_container_width=True)

#     fig2 = px.histogram(
#         df,
#         x="QUANTITY",
#         nbins=30,
#         title="Distribution of Customer Demand"
#     )
#     st.plotly_chart(fig2, use_container_width=True)

#     st.image("outputs/plots/log_log_elasticity.png", use_column_width=True)
#     st.caption("Log–Log relationship used to estimate price elasticity.")

# # -----------------------------------------------------
# # TAB 2: REVENUE OPTIMIZATION
# # -----------------------------------------------------
# with tab2:
#     st.subheader("Dynamic Price Simulation")

#     price_input = st.slider(
#         "Select Price (₹)",
#         float(df["PRICE"].min()),
#         float(df["PRICE"].max()),
#         float(best["PRICE"])
#     )

#     base = X_test.mean().to_dict()
#     base["PRICE"] = price_input

#     demand = model.predict(pd.DataFrame([base]))[0]
#     revenue = price_input * demand

#     a, b = st.columns(2)
#     a.metric("Predicted Demand", round(demand, 1))
#     b.metric("Predicted Revenue (₹)", round(revenue, 2))

#     fig_rev = px.line(
#         sim_df,
#         x="PRICE",
#         y="REVENUE",
#         markers=True,
#         title="Revenue Optimization Curve"
#     )
#     st.plotly_chart(fig_rev, use_container_width=True)

# # -----------------------------------------------------
# # TAB 3: MODEL INTELLIGENCE
# # -----------------------------------------------------
# with tab3:
#     st.subheader("Model Explainability")

#     fi_df = pd.DataFrame({
#         "Feature": FEATURES,
#         "Importance": model.feature_importances_
#     }).sort_values(by="Importance", ascending=False)

#     fig_fi = px.bar(
#         fi_df,
#         x="Importance",
#         y="Feature",
#         orientation="h",
#         title="Feature Importance (Random Forest)"
#     )
#     st.plotly_chart(fig_fi, use_container_width=True)

#     st.image("outputs/plots/shap_summary.png", use_column_width=True)
#     st.caption("SHAP values explain how each feature affects demand predictions.")

# # -----------------------------------------------------
# # TAB 4: DATA EXPLORER
# # -----------------------------------------------------
# with tab4:
#     st.subheader("Simulation Results")
#     st.dataframe(sim_df)

#     st.download_button(
#         "⬇ Download CSV",
#         sim_df.to_csv(index=False),
#         "price_simulation_results.csv",
#         "text/csv"
#     )

# # =====================================================
# # FOOTER
# # =====================================================
# st.divider()
# st.markdown("""
# <div style="text-align:center; color:#c7ecee;">
# ⚠️ Educational project • Pricing intelligence demo • Not financial advice
# </div>
# """, unsafe_allow_html=True)


# import sys
# import os
# sys.path.append(os.path.abspath("."))

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go

# from sklearn.metrics import r2_score, mean_squared_error

# from src.data_loader import load_data
# from src.modeling import train_model, calculate_elasticity
# from src.simulation import simulate_prices


# # ----------------------------------
# # PAGE CONFIG
# # ----------------------------------
# st.set_page_config(
#     page_title="AI Price Sensitivity Analyzer",
#     page_icon="💰",
#     layout="wide"
# )

# # ----------------------------------
# # GLOBAL DARK THEME
# # ----------------------------------
# st.markdown("""
# <style>
# .stApp {
#     background: radial-gradient(circle at top, #141E30, #000000);
#     color: white;
# }

# section[data-testid="stSidebar"] {
#     background: rgba(0,0,0,0.85);
# }

# h1, h2, h3 {
#     color: #00eaff;
# }

# div[data-testid="metric-container"] {
#     background: rgba(255,255,255,0.08);
#     border-radius: 16px;
#     padding: 15px;
#     box-shadow: 0 0 20px rgba(0,234,255,0.3);
# }

# .stButton>button {
#     background: linear-gradient(135deg,#00eaff,#38ef7d);
#     color: black;
#     font-weight: bold;
#     border-radius: 10px;
# }
# </style>
# """, unsafe_allow_html=True)

# # ----------------------------------
# # TITLE
# # ----------------------------------
# st.markdown("""
# <h1 style="text-align:center;font-size:46px;">💰 AI-Based Price Sensitivity Analyzer</h1>
# <p style="text-align:center;color:#9fdfff;">
# Data-driven pricing intelligence powered by Machine Learning
# </p>
# """, unsafe_allow_html=True)

# # ----------------------------------
# # FEATURES
# # ----------------------------------
# FEATURES = [
#     "PRICE",
#     "IS_WEEKEND",
#     "IS_SCHOOLBREAK",
#     "AVERAGE_TEMPERATURE",
#     "IS_OUTDOOR"
# ]

# # ----------------------------------
# # LOAD DATA & TRAIN MODEL
# # ----------------------------------
# with st.spinner("Training AI pricing model..."):
#     df = load_data()
#     model, X_test = train_model(df, FEATURES)

#     y_true = df.loc[X_test.index, "QUANTITY"]
#     y_pred = model.predict(X_test)

#     r2 = r2_score(y_true, y_pred)
#     rmse = mean_squared_error(y_true, y_pred) ** 0.5

#     elasticity = calculate_elasticity(df)
#     sim_df = simulate_prices(df, model, X_test)

# # ----------------------------------
# # KPIs
# # ----------------------------------
# best = sim_df.loc[sim_df["REVENUE"].idxmax()]

# k1, k2, k3, k4 = st.columns(4)
# k1.metric("📉 Price Elasticity", round(elasticity, 2))
# k2.metric("💰 Optimal Price (₹)", round(best["PRICE"], 2))
# k3.metric("📈 Max Revenue (₹)", round(best["REVENUE"], 2))
# k4.metric("🎯 R² Score", round(r2, 3))

# st.divider()

# # ----------------------------------
# # LIVE SIMULATION
# # ----------------------------------
# st.subheader("🎯 Live Price Simulation")

# price = st.slider(
#     "Select Product Price (₹)",
#     float(df["PRICE"].min()),
#     float(df["PRICE"].max()),
#     float(best["PRICE"])
# )

# base = X_test.mean().to_dict()
# base["PRICE"] = price

# pred_demand = model.predict(pd.DataFrame([base]))[0]
# pred_revenue = price * pred_demand

# c1, c2 = st.columns(2)
# c1.metric("Predicted Demand", round(pred_demand, 1))
# c2.metric("Predicted Revenue (₹)", round(pred_revenue, 2))

# st.divider()

# # ----------------------------------
# # COMMON PLOT LAYOUT (BLACK)
# # ----------------------------------
# plot_layout = dict(
#     paper_bgcolor="#000000",
#     plot_bgcolor="#000000",
#     font=dict(color="white"),
#     xaxis=dict(gridcolor="gray"),
#     yaxis=dict(gridcolor="gray")
# )

# # ----------------------------------
# # 1️⃣ PRICE VS DEMAND
# # ----------------------------------
# st.subheader("📦 Price vs Demand")

# fig1 = px.scatter(
#     df,
#     x="PRICE",
#     y="QUANTITY",
#     opacity=0.6
# )
# fig1.update_layout(**plot_layout)
# st.plotly_chart(fig1, use_container_width=True)

# # ----------------------------------
# # 2️⃣ PRICE VS REVENUE CURVE
# # ----------------------------------
# st.subheader("📊 Revenue Optimization Curve")

# fig2 = px.line(
#     sim_df,
#     x="PRICE",
#     y="REVENUE",
#     markers=True
# )
# fig2.update_layout(**plot_layout)
# st.plotly_chart(fig2, use_container_width=True)

# # ----------------------------------
# # 3️⃣ DEMAND DISTRIBUTION
# # ----------------------------------
# st.subheader("📈 Demand Distribution")

# fig3 = px.histogram(
#     df,
#     x="QUANTITY",
#     nbins=30
# )
# fig3.update_layout(**plot_layout)
# st.plotly_chart(fig3, use_container_width=True)

# # ----------------------------------
# # 4️⃣ LOG-LOG PRICE ELASTICITY
# # ----------------------------------
# st.subheader("📉 Log-Log Price Elasticity")

# df_log = df[df["PRICE"] > 0].copy()
# df_log["LOG_PRICE"] = np.log(df_log["PRICE"])
# df_log["LOG_DEMAND"] = np.log(df_log["QUANTITY"])

# fig4 = px.scatter(
#     df_log,
#     x="LOG_PRICE",
#     y="LOG_DEMAND",
#     opacity=0.6
# )
# fig4.update_layout(**plot_layout)
# st.plotly_chart(fig4, use_container_width=True)

# # ----------------------------------
# # 5️⃣ FEATURE IMPORTANCE
# # ----------------------------------
# st.subheader("🧠 Feature Importance")

# fi = pd.DataFrame({
#     "Feature": FEATURES,
#     "Importance": model.feature_importances_
# }).sort_values(by="Importance")

# fig5 = px.bar(
#     fi,
#     x="Importance",
#     y="Feature",
#     orientation="h"
# )
# fig5.update_layout(**plot_layout)
# st.plotly_chart(fig5, use_container_width=True)

# # ----------------------------------
# # DATA VIEW
# # ----------------------------------
# with st.expander("📄 View Simulation Table"):
#     st.dataframe(sim_df)

# st.success("🚀 Premium AI Pricing Dashboard Loaded Successfully")

# import sys
# import os
# sys.path.append(os.path.abspath("."))

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go

# from src.data_loader import load_data
# from src.modeling import train_model, calculate_elasticity
# from src.simulation import simulate_prices

# # -------------------------------------------------
# # PAGE CONFIG
# # -------------------------------------------------
# st.set_page_config(
#     page_title="AI Price Sensitivity Analyzer",
#     page_icon="💰",
#     layout="wide"
# )

# # -------------------------------------------------
# # GLOBAL AI THEME (CLEAN + PREMIUM)
# # -------------------------------------------------
# st.markdown("""
# <style>
# .stApp {
#     background-color: #05060a;
#     background-image:
#         linear-gradient(rgba(0,255,255,0.05) 1px, transparent 1px),
#         linear-gradient(90deg, rgba(0,255,255,0.05) 1px, transparent 1px);
#     background-size: 70px 70px;
#     animation: gridMove 25s linear infinite;
#     color: white;
# }

# @keyframes gridMove {
#     0% {background-position: 0 0;}
#     100% {background-position: 400px 400px;}
# }

# section[data-testid="stSidebar"] {
#     background: rgba(0,0,0,0.9);
#     backdrop-filter: blur(10px);
#     border-right: 1px solid rgba(0,255,255,0.2);
# }

# h1, h2, h3 {
#     color: #00ffff;
#     text-shadow: 0 0 12px rgba(0,255,255,0.4);
# }

# div[data-testid="metric-container"] {
#     background: rgba(0,0,0,0.7);
#     border-radius: 16px;
#     padding: 18px;
#     border: 1px solid rgba(0,255,255,0.35);
#     box-shadow: 0 0 20px rgba(0,255,255,0.2);
# }

# .js-plotly-plot .plotly {
#     background: #000000 !important;
# }
# </style>
# """, unsafe_allow_html=True)

# # -------------------------------------------------
# # HERO SECTION
# # -------------------------------------------------
# st.markdown("""
# <h1 style="text-align:center;font-size:46px;">💰 AI-Based Price Sensitivity Analyzer</h1>
# <p style="text-align:center;color:#c7ecee;font-size:18px;">
# Understand customer demand • Learn price elasticity • Maximize revenue using AI
# </p>
# """, unsafe_allow_html=True)

# st.divider()

# # -------------------------------------------------
# # FEATURES
# # -------------------------------------------------
# FEATURES = [
#     "PRICE",
#     "IS_WEEKEND",
#     "IS_SCHOOLBREAK",
#     "AVERAGE_TEMPERATURE",
#     "IS_OUTDOOR"
# ]

# # -------------------------------------------------
# # LOAD & TRAIN
# # -------------------------------------------------
# with st.spinner("Training AI model..."):
#     df = load_data()
#     model, X_test = train_model(df, FEATURES)
#     elasticity = calculate_elasticity(df)
#     sim_df = simulate_prices(df, model, X_test)

# best = sim_df.loc[sim_df["REVENUE"].idxmax()]

# # -------------------------------------------------
# # KPI SECTION (BUSINESS FIRST)
# # -------------------------------------------------
# st.subheader("📌 Key Business Insights")

# c1, c2, c3 = st.columns(3)
# c1.metric("📉 Price Elasticity", round(elasticity, 2))
# c2.metric("💰 Optimal Price (₹)", round(best["PRICE"], 2))
# c3.metric("📈 Max Revenue (₹)", round(best["REVENUE"], 2))

# st.divider()

# # =================================================
# # ACT 1 — CUSTOMER BEHAVIOR
# # =================================================
# st.subheader("🧠 How Customers React to Price")

# col1, col2 = st.columns(2)

# with col1:
#     fig_demand = px.histogram(
#         df,
#         x="QUANTITY",
#         nbins=30,
#         title="Distribution of Customer Demand",
#         template="plotly_dark"
#     )
#     st.plotly_chart(fig_demand, use_container_width=True)

# with col2:
#     fig_price_demand = px.scatter(
#         df,
#         x="PRICE",
#         y="QUANTITY",
#         opacity=0.6,
#         title="Price vs Demand",
#         template="plotly_dark"
#     )
#     st.plotly_chart(fig_price_demand, use_container_width=True)

# st.divider()

# # =================================================
# # ACT 2 — AI UNDERSTANDING
# # =================================================
# st.subheader("🤖 What the AI Learns")

# # Feature importance
# fi_df = pd.DataFrame({
#     "Feature": FEATURES,
#     "Importance": model.feature_importances_
# }).sort_values("Importance", ascending=True)

# fig_fi = px.bar(
#     fi_df,
#     x="Importance",
#     y="Feature",
#     orientation="h",
#     title="Feature Importance (Random Forest)",
#     template="plotly_dark"
# )

# st.plotly_chart(fig_fi, use_container_width=True)

# st.divider()

# # =================================================
# # ACT 3 — MONEY MAKING
# # =================================================
# st.subheader("💸 Revenue Optimization")

# fig_rev = px.line(
#     sim_df,
#     x="PRICE",
#     y="REVENUE",
#     markers=True,
#     title="Price vs Revenue Curve",
#     template="plotly_dark"
# )
# st.plotly_chart(fig_rev, use_container_width=True)

# st.subheader("🎯 Live Price Simulation")

# price = st.slider(
#     "Choose Price (₹)",
#     float(df["PRICE"].min()),
#     float(df["PRICE"].max()),
#     float(best["PRICE"])
# )

# base = X_test.mean().to_dict()
# base["PRICE"] = price

# pred_demand = model.predict(pd.DataFrame([base]))[0]
# pred_revenue = price * pred_demand

# c4, c5 = st.columns(2)
# c4.metric("Predicted Demand", round(pred_demand, 1))
# c5.metric("Predicted Revenue (₹)", round(pred_revenue, 2))

# st.divider()

# with st.expander("📄 View Full Price Simulation Table"):
#     st.dataframe(sim_df)

# st.success("🚀 AI Pricing Dashboard Ready")

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

