# src/backtest/engine.py

import numpy as np
import pandas as pd

from src.portfolio.covariance import compute_return_matrix, compute_covariance_matrix
from src.portfolio.returns import compute_expected_returns
from src.portfolio.optimizer import max_sharpe_optimization
from src.models.arima_model import fit_arima, forecast_arima
from src.models.arima_model import select_arima_order, fit_arima_fixed

def run_rolling_backtest(
    df: pd.DataFrame,
    lookback: int = 252,
    rebalance_freq: str = "M",
    risk_free_rate: float = 0.02,
):
    """
    Rolling portfolio backtest with monthly rebalancing.
    """

    assets = ["TSLA", "SPY", "BND"]

    # Compute daily returns
    returns = compute_return_matrix(df, assets)

    portfolio_value = 1.0
    portfolio_history = []

    # Resample to monthly rebalance dates
    rebalance_dates = returns.resample(rebalance_freq).last().index
    # Select ARIMA order once using initial lookback window
    initial_log_prices = np.log(df["TSLA_adjclose"].iloc[:lookback])
    arima_order = select_arima_order(initial_log_prices)
    print(f"Selected ARIMA order: {arima_order}")

    for date in rebalance_dates:

        if date not in returns.index:
            continue

        end_loc = returns.index.get_loc(date)

        if end_loc < lookback:
            continue

        window_returns = returns.iloc[end_loc - lookback:end_loc]

        # ---- Forecast TSLA ----
        tsla_prices = df["TSLA_adjclose"].iloc[end_loc - lookback:end_loc]
        log_prices = np.log(tsla_prices)

        model = fit_arima_fixed(log_prices, arima_order)
        forecast_log = forecast_arima(model, 1)

        forecast_price = np.exp(forecast_log.iloc[-1])
        last_price = tsla_prices.iloc[-1]

        tsla_forecast_return = (forecast_price / last_price - 1)

        # ---- Compute expected returns ----
        expected_returns = compute_expected_returns(
            df.iloc[end_loc - lookback:end_loc],
            tsla_forecast_return
        )

        # ---- Covariance ----
        covariance_matrix = compute_covariance_matrix(window_returns)

        # ---- Optimize ----
        weights = max_sharpe_optimization(
            expected_returns,
            covariance_matrix,
            risk_free_rate
        )

        # ---- Apply next month return ----
        next_returns = returns.iloc[end_loc]

        portfolio_return = np.dot(weights.values, next_returns.values)

        portfolio_value *= (1 + portfolio_return)

        portfolio_history.append({
            "date": date,
            "portfolio_value": portfolio_value,
            "portfolio_return": portfolio_return,
        })

    return pd.DataFrame(portfolio_history).set_index("date")