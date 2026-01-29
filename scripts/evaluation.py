import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Imports
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.portfolio.returns import compute_return_matrix
from src.portfolio.covariance import compute_covariance_matrix
from src.portfolio.optimizer import max_sharpe_optimization

ASSETS = ["TSLA", "SPY", "BND"]
RISK_FREE_RATE = 0.02


def evaluate_portfolios(
    data_path="data/processed/combined.csv",
    tsla_return_path="data/processed/tsla_expected_return.csv",
):
    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    returns = compute_return_matrix(df, ASSETS)
    cov = compute_covariance_matrix(returns)

    # --------------------------------------------------
    # Expected returns
    # --------------------------------------------------
    hist_expected = returns.mean() * 252  # annualized historical returns

    tsla_expected = pd.read_csv(tsla_return_path).iloc[0]["expected_annual_return"]
    expected_returns = hist_expected.copy()
    expected_returns["TSLA"] = tsla_expected  # override with forecast

    # --------------------------------------------------
    # Optimized portfolio (Max Sharpe)
    # --------------------------------------------------
    opt_weights = max_sharpe_optimization(
        expected_returns.values,
        cov.values,
        risk_free_rate=RISK_FREE_RATE,
    )
    opt_weights = pd.Series(opt_weights, index=ASSETS)

    # --------------------------------------------------
    # Equal-weight portfolio
    # --------------------------------------------------
    eq_weights = pd.Series(
        np.repeat(1 / len(ASSETS), len(ASSETS)),
        index=ASSETS,
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    def portfolio_metrics(weights):
        ret = np.dot(weights, expected_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov.values, weights)))
        sharpe = (ret - RISK_FREE_RATE) / vol
        return ret, vol, sharpe

    opt_metrics = portfolio_metrics(opt_weights.values)
    eq_metrics = portfolio_metrics(eq_weights.values)

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

    # --------------------------------------------------
    # Plot weights
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))
    opt_weights.plot(kind="bar")
    plt.title("Optimized Portfolio Allocation")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.show()

    return summary


if __name__ == "__main__":
    evaluate_portfolios()
