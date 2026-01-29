import numpy as np
import pandas as pd
from scipy.optimize import minimize

def max_sharpe_optimization(expected_returns, cov_matrix, risk_free_rate=0.0):
    """
    Compute weights of the maximum Sharpe ratio portfolio.
    """
    n = len(expected_returns)

    def neg_sharpe(weights):
        port_return = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_return - risk_free_rate) / port_vol

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    x0 = np.array([1/n]*n)

    result = minimize(neg_sharpe, x0, bounds=bounds, constraints=constraints)
    return result.x

def min_vol_optimization(cov_matrix):
    """
    Compute weights for minimum volatility portfolio.
    """
    n = cov_matrix.shape[0]

    def port_vol(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Constraint: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1})
    bounds = tuple((0,1) for _ in range(n))
    x0 = np.array([1/n]*n)

    result = minimize(port_vol, x0, bounds=bounds, constraints=constraints)
    return result.x


def optimize_portfolio(expected_returns, cov_matrix, risk_aversion=1.0):
    """
    General mean-variance optimizer (your original function)
    """
    assets = expected_returns.index.tolist()
    n_assets = len(assets)
    mu = expected_returns.values
    Sigma = cov_matrix.values

    def objective(weights):
        return 0.5 * (weights.T @ Sigma @ weights) - risk_aversion * (weights.T @ mu)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0) for _ in range(n_assets)]
    initial_weights = np.array([1.0 / n_assets] * n_assets)

    result = minimize(objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError("Portfolio optimization failed")
    return pd.Series(result.x, index=assets)
