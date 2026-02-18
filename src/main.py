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
from src.backtest.engine import run_rolling_backtest
from src.portfolio.metrics import (
    calculate_cagr,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)

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
    print("\nExpected Returns:")
    print(expected_returns)

    print("\nCovariance Matrix:")
    print(covariance_matrix)

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

    df = run_data_pipeline()

    # Single optimization (kept for reference)
    tsla_expected_return = run_forecasting_pipeline()
    weights = run_portfolio_pipeline(tsla_expected_return)

    # --------------------------
    # Rolling Backtest
    # --------------------------

    print("\n[5/5] Running rolling backtest...")

    portfolio_history = run_rolling_backtest(
        df=df,
        lookback=252,
        rebalance_freq="ME",
        risk_free_rate=PortfolioConfig().risk_free_rate
    )

    print("\nBacktest completed.")
    print(portfolio_history.tail())

    portfolio_history.to_csv("data/processed/backtest_results.csv")

    print("\nPipeline completed successfully.")
    

    print("\n[6/6] Calculating performance metrics...")

    cagr = calculate_cagr(portfolio_history["portfolio_value"])
    volatility = calculate_annualized_volatility(portfolio_history["portfolio_return"])
    sharpe = calculate_sharpe_ratio(
    portfolio_history["portfolio_return"],
    risk_free_rate=PortfolioConfig().risk_free_rate,
)
    max_dd = calculate_max_drawdown(portfolio_history["portfolio_value"])

    print("\nPerformance Metrics:")
    print(f"CAGR: {cagr:.4f}")
    print(f"Annualized Volatility: {volatility:.4f}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Max Drawdown: {max_dd:.4f}")
if __name__ == "__main__":
    main()
