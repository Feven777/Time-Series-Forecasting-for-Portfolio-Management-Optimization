import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.portfolio.covariance import compute_covariance_matrix
from src.portfolio.returns import compute_expected_returns
from src.portfolio.optimizer import max_sharpe_optimization


ASSETS = ["TSLA", "SPY", "BND"]
RISK_FREE_RATE = 0.02


def evaluate_portfolios(
    data_path="data/processed/combined.csv",
    tsla_return_path="data/processed/tsla_expected_return.csv",
):
    # Load data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Load TSLA forecast return
    tsla_forecast_return = pd.read_csv(tsla_return_path)[
        "expected_annual_return"
    ].iloc[0]

    # Expected returns
    expected_returns = compute_expected_returns(
        df=df,
        tsla_forecast_return=tsla_forecast_return,
        assets=ASSETS,
    )

    # Covariance (from historical daily returns)
    returns_df = df[[f"{a}_return" for a in ASSETS]]
    cov = compute_covariance_matrix(returns_df)

    # Optimized portfolio
    opt_weights = max_sharpe_optimization(
        expected_returns,
        cov,
        risk_free_rate=RISK_FREE_RATE,
    )

    # Equal-weight portfolio
    eq_weights = pd.Series(
        np.repeat(1 / len(ASSETS), len(ASSETS)),
        index=ASSETS,
    )

    # Metrics
    def metrics(weights):
        ret = weights @ expected_returns
        vol = np.sqrt(weights.T @ cov.values @ weights)
        sharpe = (ret - RISK_FREE_RATE) / vol
        return ret, vol, sharpe

    opt_metrics = metrics(opt_weights)
    eq_metrics = metrics(eq_weights)

    summary = pd.DataFrame(
        {
            "Portfolio": ["Optimized", "Equal Weight"],
            "Expected Return": [opt_metrics[0], eq_metrics[0]],
            "Volatility": [opt_metrics[1], eq_metrics[1]],
            "Sharpe Ratio": [opt_metrics[2], eq_metrics[2]],
        }
    )

    print("\n📊 Portfolio Comparison")
    print(summary)

    # Plot weights
    plt.figure(figsize=(8, 5))
    opt_weights.plot(kind="bar")
    plt.title("Optimized Portfolio Allocation")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.show()

    return summary, opt_weights


if __name__ == "__main__":
    evaluate_portfolios()
