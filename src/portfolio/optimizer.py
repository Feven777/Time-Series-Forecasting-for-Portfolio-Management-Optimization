import numpy as np
import pandas as pd
from scipy.optimize import minimize


def max_sharpe_optimization(expected_returns, covariance_matrix, risk_free_rate):

    epsilon = 1e-6
    covariance_matrix = covariance_matrix + epsilon * np.eye(len(covariance_matrix))

    try:
        # your existing optimization code
        result = minimize(...)

        if not result.success:
            raise RuntimeError("Optimization failed")

        return pd.Series(result.x, index=expected_returns.index)

    except Exception:
        # Fallback to equal weights
        n = len(expected_returns)
        equal_weights = np.ones(n) / n
        return pd.Series(equal_weights, index=expected_returns.index)