import pandas as pd
from src.portfolio.optimizer import optimize_portfolio

# Expected returns
expected_returns = pd.Series(
    {"TSLA": 0.02, "SPY": 0.01, "BND": 0.003}
)

# Covariance matrix
covariance = pd.DataFrame(
    [
        [0.04, 0.02, 0.01],
        [0.02, 0.02, 0.005],
        [0.01, 0.005, 0.005],
    ],
    index=["TSLA", "SPY", "BND"],
    columns=["TSLA", "SPY", "BND"],
)

# Optimize portfolio
weights = optimize_portfolio(expected_returns, covariance)

print(weights)
print("Sum of weights:", weights.sum())
