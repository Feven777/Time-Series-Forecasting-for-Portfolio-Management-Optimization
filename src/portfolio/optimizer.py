import numpy as np
import pandas as pd
from scipy.optimize import minimize


def max_sharpe_optimization(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> pd.Series:
    """
    Compute Maximum Sharpe Ratio portfolio (long-only).
    """

    assets = expected_returns.index.tolist()
    mu = expected_returns.values
    Sigma = covariance_matrix.values
    n = len(mu)

    def neg_sharpe(weights):
        port_return = weights @ mu
        port_vol = np.sqrt(weights.T @ Sigma @ weights)
        return -(port_return - risk_free_rate) / port_vol

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n)]
    init = np.repeat(1 / n, n)

    result = minimize(
        neg_sharpe,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError("Max Sharpe optimization failed")

    return pd.Series(result.x, index=assets)
