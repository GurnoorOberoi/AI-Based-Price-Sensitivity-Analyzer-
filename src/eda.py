# import matplotlib.pyplot as plt
# import seaborn as sns

# def price_demand_plot(df):
#     plt.figure()
#     sns.scatterplot(x=df['PRICE'], y=df['QUANTITY'], alpha=0.6)
#     plt.title("Price vs Demand")
#     plt.show()

# def price_revenue_plot(df):
#     plt.figure()
#     sns.scatterplot(x=df['PRICE'], y=df['REVENUE'], alpha=0.6)
#     plt.title("Price vs Revenue")
#     plt.show()

import os
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_plot_dir():
    os.makedirs("outputs/plots", exist_ok=True)

def price_vs_demand(df):
    ensure_plot_dir()
    plt.figure()
    sns.scatterplot(x=df['PRICE'], y=df['QUANTITY'], alpha=0.6)
    plt.title("Price vs Demand")
    plt.savefig("outputs/plots/price_vs_demand.png")
    plt.close()

def price_vs_revenue(df):
    ensure_plot_dir()
    plt.figure()
    sns.scatterplot(x=df['PRICE'], y=df['REVENUE'], alpha=0.6)
    plt.title("Price vs Revenue")
    plt.savefig("outputs/plots/price_vs_revenue.png")
    plt.close()

