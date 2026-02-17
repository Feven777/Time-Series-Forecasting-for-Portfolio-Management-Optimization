"""
ARIMA modeling for TSLA price forecasting.

- Uses log-transformed prices
- Automatic order selection
- Time-aware train/test split
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# =========================
# Paths
# =========================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")


# =========================
# Data loading
# =========================

def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(DATA_DIR, "combined.csv"),
        index_col=0,
        parse_dates=True
    )
    return df


# =========================
# Train-test split
# =========================

def train_test_split(series: pd.Series):
    train = series.loc[: "2024-12-31"]
    test = series.loc["2025-01-01":]
    return train, test


# =========================
# ARIMA pipeline
# =========================

def fit_arima(train_series: pd.Series):
    """Fit ARIMA model using auto_arima."""
    model = auto_arima(
        train_series,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    return model


def fit_arima_fixed(train_series: pd.Series, order: tuple):
    """
    Fit ARIMA using fixed (p,d,q) order.
    Uses default statespace estimator.
    Suppresses convergence warnings only.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        model = ARIMA(train_series, order=order)
        fitted = model.fit()

    return fitted

def forecast_arima(model, n_periods: int):
    """
    Forecast next n_periods.
    Supports both pmdarima and statsmodels models.
    """

    # pmdarima model
    if hasattr(model, "predict"):
        try:
            return model.predict(n_periods=n_periods)
        except TypeError:
            pass

    # statsmodels model
    if hasattr(model, "forecast"):
        return model.forecast(steps=n_periods)

    raise ValueError("Unsupported ARIMA model type.")


# =========================
# Evaluation
# =========================

def evaluate_forecast(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae
def select_arima_order(series: pd.Series):
    """
    Select ARIMA order once using auto_arima.
    """
    model = auto_arima(
        series,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    return model.order

# =========================
# Main execution
# =========================

def main():
    df = load_data()

    # Use log-price
    price = np.log(df["TSLA_adjclose"])

    train, test = train_test_split(price)

    print("Fitting ARIMA...")
    model = fit_arima(train)
    print(model.summary())

    print("Forecasting...")
    forecast_log = forecast_arima(model, len(test))

    # Convert back to price space
    forecast = np.exp(forecast_log)
    actual = np.exp(test)

    rmse, mae = evaluate_forecast(actual, forecast)

    print(f"ARIMA RMSE: {rmse:.4f}")
    print(f"ARIMA MAE: {mae:.4f}")

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(actual.index, actual, label="Actual")
    plt.plot(actual.index, forecast, label="Forecast", linestyle="--")
    plt.title("ARIMA Forecast — TSLA Price")
    plt.legend()
    plt.show()

    return actual, forecast


# =========================
# Script entry point
# =========================

if __name__ == "__main__":
    actual, forecast = main()

    results = pd.DataFrame({
        "actual": actual,
        "forecast": forecast
    })

    results.to_csv(
        os.path.join(DATA_DIR, "arima_forecast_tsla.csv"),
        index=True
    )
