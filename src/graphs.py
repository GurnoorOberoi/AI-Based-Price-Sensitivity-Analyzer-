import os
import matplotlib.pyplot as plt
import seaborn as sns

PLOT_DIR = "outputs/plots"

def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)

def price_vs_demand(df):
    ensure_plot_dir()
    plt.figure()
    sns.scatterplot(x=df['PRICE'], y=df['QUANTITY'], alpha=0.6)
    plt.title("Price vs Demand")
    plt.savefig(f"{PLOT_DIR}/price_vs_demand.png")
    plt.close()

def price_vs_revenue(df):
    ensure_plot_dir()
    plt.figure()
    sns.scatterplot(x=df['PRICE'], y=df['REVENUE'], alpha=0.6)
    plt.title("Price vs Revenue")
    plt.savefig(f"{PLOT_DIR}/price_vs_revenue.png")
    plt.close()
