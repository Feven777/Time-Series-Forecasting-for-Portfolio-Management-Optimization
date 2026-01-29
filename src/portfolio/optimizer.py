import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize_portfolio(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_aversion: float = 1.0,
) -> pd.Series:
    """
    Perform mean-variance portfolio optimization (long-only).

    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns for each asset
    covariance_matrix : pd.DataFrame
        Covariance matrix of asset returns
    risk_aversion : float
        Risk aversion parameter (higher = more conservative)

    Returns
    -------
    pd.Series
        Optimal portfolio weights
    """

    assets = expected_returns.index.tolist()
    n_assets = len(assets)

    mu = expected_returns.values
    Sigma = covariance_matrix.values

    def objective(weights):
        portfolio_variance = weights.T @ Sigma @ weights
        portfolio_return = weights.T @ mu
        return 0.5 * portfolio_variance - risk_aversion * portfolio_return

    # Constraints: weights sum to 1
    constraints = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1,
    }

    # Bounds: no short selling
    bounds = [(0.0, 1.0) for _ in range(n_assets)]

    # Initial guess: equal weights
    initial_weights = np.array([1.0 / n_assets] * n_assets)

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError("Portfolio optimization failed")

    return pd.Series(result.x, index=assets)
