import os
import numpy as np
import pandas as pd

REPORT_DIR = "outputs/reports"

def simulate_prices(df, model, X_test):
    os.makedirs(REPORT_DIR, exist_ok=True)

    price_range = np.linspace(df['PRICE'].min(), df['PRICE'].max(), 20)
    base = X_test.mean().to_dict()

    results = []
    for price in price_range:
        base['PRICE'] = price
        demand = model.predict(pd.DataFrame([base]))[0]
        revenue = price * demand
        results.append([price, demand, revenue])

    sim_df = pd.DataFrame(
        results,
        columns=['PRICE', 'PREDICTED_DEMAND', 'REVENUE']
    )

    sim_df.to_csv(
        f"{REPORT_DIR}/price_simulation_results.csv",
        index=False
    )

    return sim_df


