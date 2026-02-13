"""
Main pipeline entrypoint for portfolio forecasting and optimization.

This orchestrates:
- Data fetching
- Preprocessing
- Forecasting
- Portfolio optimization
- Backtesting
"""

from src.config.config import DataConfig, LSTMConfig, ARIMAConfig, PortfolioConfig


def run_data_pipeline():
    print("Running data pipeline...")
    # TODO: connect to src.data_fetch and preprocess modules
    print("Data pipeline complete.")


def run_forecasting_pipeline():
    print("Running forecasting pipeline...")
    # TODO: connect ARIMA and LSTM modules
    print("Forecasting complete.")


def run_portfolio_pipeline():
    print("Running portfolio optimization...")
    # TODO: connect portfolio optimizer
    print("Portfolio optimization complete.")


def run_backtesting_pipeline():
    print("Running backtesting...")
    # TODO: connect backtest module
    print("Backtesting complete.")


def main():

    print("=" * 50)
    print("Portfolio Forecasting and Optimization Pipeline")
    print("=" * 50)

    data_config = DataConfig()
    lstm_config = LSTMConfig()
    arima_config = ARIMAConfig()
    portfolio_config = PortfolioConfig()

    print("Configuration loaded.")

    run_data_pipeline()
    run_forecasting_pipeline()
    run_portfolio_pipeline()
    run_backtesting_pipeline()

    print("=" * 50)
    print("Pipeline complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
