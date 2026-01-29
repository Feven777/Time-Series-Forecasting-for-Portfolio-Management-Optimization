import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Use evaluate_portfolios to get optimized weights
from scripts.evaluation import evaluate_portfolios


def compute_daily_returns(df: pd.DataFrame, assets=("TSLA", "SPY", "BND")) -> pd.DataFrame:
    """
    Compute daily returns for given assets from adjusted close prices.
    """
    return df[[f"{a}_adjclose" for a in assets]].pct_change().dropna()


def backtest_strategy(data_path="data/processed/combined.csv"):
    # Load data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Limit backtest period
    df = df.loc["2025-01-01":"2026-01-15"]

    assets = ["TSLA", "SPY", "BND"]
    returns = compute_daily_returns(df, assets)

    # Get optimized portfolio
    _, opt_weights = evaluate_portfolios(
        data_path=data_path,
        tsla_return_path="data/processed/tsla_expected_return.csv",
    )

    # Benchmark: 60% SPY / 40% BND, 0% TSLA
    benchmark_weights = np.array([0.0, 0.6, 0.4])

    # Portfolio returns
    strategy_returns = returns @ opt_weights.values
    benchmark_returns = returns @ benchmark_weights

    # Cumulative returns
    strategy_cum = (1 + strategy_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(strategy_cum, label="Optimized Strategy")
    plt.plot(benchmark_cum, label="Benchmark (60/40)")
    plt.title("Backtest: Strategy vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    backtest_strategy()
