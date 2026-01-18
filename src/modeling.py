import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------------------------------
# PRICE ELASTICITY (ECONOMICS)
# --------------------------------------------------
def calculate_elasticity(df):
    df = df.copy()

    df["LOG_PRICE"] = np.log(df["PRICE"])
    df["LOG_QUANTITY"] = np.log(df["QUANTITY"])

    X = df[["LOG_PRICE"]]
    y = df["LOG_QUANTITY"]

    model = LinearRegression()
    model.fit(X, y)

    return model.coef_[0]


# --------------------------------------------------
# DEMAND PREDICTION MODEL (ML)
# --------------------------------------------------
def train_model(df, features):
    X = df[features]
    y = df["QUANTITY"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("R2 Score:", round(r2_score(y_test, y_pred), 3))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))


    return model, X_test
