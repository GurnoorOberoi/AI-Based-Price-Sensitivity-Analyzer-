# 💰 AI-Based Price Sensitivity Analyzer

> **Data-driven pricing intelligence powered by Machine Learning**  
> Understand how price changes impact demand and revenue — and identify the optimal pricing strategy.

---

📄 **Project Report**: [View Report](https://docs.google.com/document/d/1yvgwIiIGfNUzKarbiACLJSepCb19kliQaFVpRhrvPSk/edit?usp=sharing)  
📊 **Live Dashboard**: [Launch Dashboard](https://gurnooroberoi-ai-based-price-sensitiv-dashboarddashboard-gkhqcy.streamlit.app/)  
<!-- 🎥 **Video Demo & Presentation**: [Watch Demo](https://drive.google.com/file/d/1bIwAPATEmIPfP-m1Sbatw9g6DersHOph/view?usp=drive_link) -->
 
---
## 🚀 Project Overview

The **AI-Based Price Sensitivity Analyzer** is an end-to-end machine learning project that helps businesses understand **how sensitive customer demand is to price changes** and **which price maximizes revenue**.

Unlike simple sales prediction systems, this project focuses on **price elasticity, revenue optimization, and decision support**, making it highly relevant for **retail, e-commerce, and FMCG businesses**.

---

## ✨ Key Features

- 📉 Price Elasticity Analysis  
- 💰 Revenue Optimization using Price Simulation  
- 🤖 Machine Learning (Random Forest Regressor)  
- 📊 Automated EDA & Visual Reports  
- 🧠 Model Explainability (Feature Importance + SHAP)  
- 🎛️ Interactive Streamlit Dashboard  
- 🌌 Premium Animated UI Theme  

---

## 🧠 What Makes This Project Unique?

| Aspect | Traditional Projects | This Project |
|------|---------------------|-------------|
| Goal | Prediction only | Pricing strategy |
| Insight | Black-box | Explainable AI |
| Output | Static charts | Interactive dashboard |
| Business Value | Low | High |

---

## 🏗️ Project Structure
```
AI-Based_Price_Sensitivity_Analyzer/
│
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── eda.py
│ ├── modeling.py
│ ├── simulation.py
│ └── visualization.py
│
├── dashboard/
│ └── dashboard.py
│
├── data/
│ └── raw/
│
├── outputs/
│ ├── plots/
│ └── reports/
│
├── README.md
└── requirements.txt
```
---

## 📊 Dataset Description

Retail café transaction data containing:

- Product price
- Quantity sold
- Weekend indicator
- School break indicator
- Temperature
- Outdoor sales flag
- Calendar metadata

---

## 🤖 Machine Learning Details

### Model
- **Random Forest Regressor**

### Target Variable
- `QUANTITY` (Customer Demand)

### Input Features
- `PRICE`
- `IS_WEEKEND`
- `IS_SCHOOLBREAK`
- `AVERAGE_TEMPERATURE`
- `IS_OUTDOOR`

---

## 📈 Model Performance

- **R² Score:** ~0.92+
- **RMSE:** Low prediction error
- Strong generalization on unseen data

---

## 📉 Price Elasticity

Calculated using a **log–log regression model**:

\[
Elasticity = \frac{\%\ Change\ in\ Demand}{\%\ Change\ in\ Price}
\]

This provides **economic insight**, not just predictions.

---

## 🔍 Explainability

The project includes:

- Feature Importance (Random Forest)
- SHAP Summary Plot
- Log–Log Price Elasticity Visualization

---

## 📊 Generated Outputs

Automatically saved inside `outputs/plots/`:

- Price vs Demand  
- Price vs Revenue  
- Demand Distribution  
- Log–Log Elasticity  
- Feature Importance  
- SHAP Summary Plot  

Reports saved in `outputs/reports/`.

---

## 🎛️ Interactive Dashboard

Built using **Streamlit**, the dashboard provides:

- Business KPIs (Elasticity, Optimal Price, Revenue)
- Live Price Simulation (What-If Analysis)
- Interactive Visualizations
- Animated futuristic background

### Run Dashboard

```bash
streamlit run dashboard/dashboard.py
```

## 🛠️ Tech Stack

### Core
- Python  
- Pandas, NumPy  
- Scikit-learn  
- SHAP  

### Visualization
- Matplotlib  
- Plotly  
- Streamlit  

### UI
- Custom CSS  
- Dark & animated theme  

---

## 🎯 Business Use Cases

- Retail pricing strategy  
- Discount optimization  
- Demand sensitivity analysis  
- AI-powered business decision support  

---

## 📌 Future Enhancements

- Dynamic pricing models  
- Time-series forecasting  
- Multi-product optimization  
- Cloud deployment (AWS / GCP)  

---

## 👤 Author

**Gurnoor Oberoi**  
🎓 Computer Science Engineer  
💡 AI • Data Science • Business Analytics  
