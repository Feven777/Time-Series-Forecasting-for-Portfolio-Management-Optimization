"""
Main pipeline for financial forecasting and portfolio optimization.

This orchestrates:
1. Data fetching
2. Preprocessing
3. Forecasting (ARIMA)
4. Portfolio optimization
"""
import numpy as np
import pandas as pd

from src.config.config import DataConfig, PortfolioConfig

from src.data_fetch import main as fetch_data
from src.preprocess import preprocess_all_assets

from src.models.arima_model import (
    load_data,
    train_test_split,
    fit_arima,
    forecast_arima,
)

from src.portfolio.optimizer import max_sharpe_optimization

from src.portfolio.returns import compute_expected_returns
from src.portfolio.covariance import compute_covariance_matrix


# =========================
# Data pipeline
# =========================

def run_data_pipeline():
    print("\n[1/4] Fetching data...")
    fetch_data()

    print("[2/4] Preprocessing data...")
    df = preprocess_all_assets()

    print("Data pipeline complete.")
    return df


# =========================
# Forecast pipeline
# =========================

def run_forecasting_pipeline():

    print("\n[3/4] Running ARIMA forecast...")

    df = load_data()
    log_price = df["TSLA_adjclose"].apply(lambda x: np.log(x))


    train, test = train_test_split(log_price)

    model = fit_arima(train)

    forecast_log = forecast_arima(model, len(test))

    forecast = pd.Series(
        np.exp(forecast_log),
        index=test.index
    )

    print("Forecasting complete.")
    tsla_forecast_return = forecast.pct_change().mean()
    return tsla_forecast_return



# =========================
# Portfolio pipeline
# =========================

def run_portfolio_pipeline(tsla_forecast_return):

    print("\n[4/4] Running portfolio optimization...")

    df = load_data()

    assets = ["TSLA", "SPY", "BND"]

# Compute return matrix
    from src.portfolio.covariance import compute_return_matrix

    returns = compute_return_matrix(df, assets)

    expected_returns = compute_expected_returns(
    df,
    tsla_forecast_return
)

    covariance_matrix = compute_covariance_matrix(returns)

    portfolio_config = PortfolioConfig()

    weights = max_sharpe_optimization(
        expected_returns,
        covariance_matrix,
        portfolio_config.risk_free_rate
    )

    print("\nOptimal Portfolio Weights:")
    print(weights)

    return weights


# =========================
# Main entrypoint
# =========================

def main():

    print("=" * 50)
    print("Portfolio Forecasting Pipeline")
    print("=" * 50)

    run_data_pipeline()

    tsla_forecast_return = run_forecasting_pipeline()

    weights = run_portfolio_pipeline(tsla_forecast_return)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
