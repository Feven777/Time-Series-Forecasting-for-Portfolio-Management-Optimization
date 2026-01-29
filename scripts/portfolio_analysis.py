import pandas as pd
from src.portfolio.covariance import compute_return_matrix, compute_covariance_matrix

# Load data
df = pd.read_csv("data/processed/combined.csv", index_col=0, parse_dates=True)

# Select assets
assets = ["TSLA", "SPY", "BND"]

# Compute returns and covariance
returns = compute_return_matrix(df, assets)
cov = compute_covariance_matrix(returns)

# Print results
print(returns.head())
print(cov)
