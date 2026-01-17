# from data_loader import load_data
# from preprocessing import preprocess
# from eda import price_demand_plot, price_revenue_plot
# from modeling import elasticity_model, train_rf
# from simulation import simulate_prices

# features = [
#     'PRICE',
#     'IS_WEEKEND',
#     'IS_SCHOOLBREAK',
#     'AVERAGE_TEMPERATURE',
#     'IS_OUTDOOR'
# ]

# df = load_data()
# df = preprocess(df)

# price_demand_plot(df)
# price_revenue_plot(df)

# elasticity = elasticity_model(df)
# print("Price Elasticity:", round(elasticity, 2))

# rf_model, X_test = train_rf(df, features)

# sim_df = simulate_prices(df, rf_model, X_test)
# best = sim_df.loc[sim_df['REVENUE'].idxmax()]

# print("\nOPTIMAL PRICE STRATEGY")
# print("Price:", round(best['PRICE'],2))
# print("Demand:", round(best['DEMAND'],1))
# print("Revenue:", round(best['REVENUE'],2))

# from data_loader import load_data
# from preprocessing import preprocess
# from eda import price_vs_demand, price_vs_revenue
# from modeling import calculate_elasticity, train_model
# from simulation import simulate_prices
# from graphs import revenue_curve
# from simulation import run_price_simulation

# FEATURES = [
#     "PRICE",
#     "IS_WEEKEND",
#     "IS_SCHOOLBREAK",
#     "AVERAGE_TEMPERATURE",
#     "IS_OUTDOOR"
# ]

# def main():
#     df = load_data()
#     df = preprocess(df)

#     price_vs_demand(df)
#     price_vs_revenue(df)

#     elasticity = calculate_elasticity(df)
#     print("Price Elasticity:", round(elasticity, 2))

#     model, X_test = train_model(df, FEATURES)
#     sim_df = simulate_prices(df, model, X_test)

#     revenue_curve(sim_df)

#     best = sim_df.loc[sim_df["REVENUE"].idxmax()]
#     print("\nOPTIMAL PRICE STRATEGY")
#     print("Optimal Price:", round(best["PRICE"], 2))
#     print("Expected Demand:", round(best["PREDICTED_DEMAND"], 1))
#     print("Expected Revenue:", round(best["REVENUE"], 2))
    

# if __name__ == "__main__":
#     main()


# src/visualization.py

from data_loader import load_data
from preprocessing import preprocess
from eda import price_vs_demand, price_vs_revenue
from modeling import calculate_elasticity, train_model
from simulation import simulate_prices

FEATURES = [
    'PRICE',
    'IS_WEEKEND',
    'IS_SCHOOLBREAK',
    'AVERAGE_TEMPERATURE',
    'IS_OUTDOOR'
]

def main():
    df = load_data()
    df = preprocess(df)

    price_vs_demand(df)
    price_vs_revenue(df)

    elasticity = calculate_elasticity(df)
    print("Price Elasticity:", round(elasticity, 2))

    model, X_test = train_model(df, FEATURES)
    sim_df = simulate_prices(df, model, X_test)

    best = sim_df.loc[sim_df['REVENUE'].idxmax()]

    print("\nOPTIMAL PRICE STRATEGY")
    print("Optimal Price:", round(best['PRICE'], 2))
    print("Expected Demand:", round(best['PREDICTED_DEMAND'], 1))
    print("Expected Revenue:", round(best['REVENUE'], 2))

# if __name__ == "__main__":
#     main()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from data_loader import load_data
from modeling import train_model
from simulation import simulate_prices

# ---------------------------
# CREATE OUTPUT DIRECTORY
# ---------------------------
PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

FEATURES = [
    "PRICE",
    "IS_WEEKEND",
    "IS_SCHOOLBREAK",
    "AVERAGE_TEMPERATURE",
    "IS_OUTDOOR"
]

TARGET = "QUANTITY"

# ---------------------------
# LOAD DATA & TRAIN MODEL
# ---------------------------
df = load_data()
model, X_test = train_model(df, FEATURES)

# ---------------------------
# 1️⃣ DISTRIBUTION OF DEMAND
# ---------------------------
plt.figure(figsize=(8, 5))
plt.hist(df[TARGET], bins=30)
plt.title("Distribution of Customer Demand")
plt.xlabel("Quantity Sold")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/demand_distribution.png")
plt.close()

# ---------------------------
# 2️⃣ LOG–LOG PRICE ELASTICITY
# ---------------------------
df_log = df[df["PRICE"] > 0].copy()
df_log["LOG_PRICE"] = np.log(df_log["PRICE"])
df_log["LOG_DEMAND"] = np.log(df_log["QUANTITY"])

plt.figure(figsize=(8, 5))
plt.scatter(df_log["LOG_PRICE"], df_log["LOG_DEMAND"], alpha=0.5)
plt.title("Log–Log Price Elasticity Relationship")
plt.xlabel("Log(Price)")
plt.ylabel("Log(Demand)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/log_log_elasticity.png")
plt.close()

# ---------------------------
# 3️⃣ FEATURE IMPORTANCE
# ---------------------------
importances = model.feature_importances_
fi_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(fi_df["Feature"], fi_df["Importance"])
plt.title("Feature Importance (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/feature_importance.png")
plt.close()

# ---------------------------
# 4️⃣ SHAP VALUES
# ---------------------------
explainer = shap.Explainer(model, X_test)
# shap_values = explainer(X_test)
shap_values = explainer(X_test, check_additivity=False)


shap.summary_plot(
    shap_values,
    X_test,
    show=False
)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/shap_summary.png")
plt.close()

print("✅ All plots generated successfully in output/plots/")

