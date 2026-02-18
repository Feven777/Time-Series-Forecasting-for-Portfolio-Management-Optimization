import numpy as np
import pandas as pd


def calculate_cagr(portfolio_values: pd.Series, periods_per_year: int = 12) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR)
    """
    total_periods = len(portfolio_values)
    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0]
    return total_return ** (periods_per_year / total_periods) - 1


def calculate_annualized_volatility(returns: pd.Series, periods_per_year: int = 12) -> float:
    """
    Annualized volatility
    """
    return returns.std() * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float,
    periods_per_year: int = 12,
) -> float:
    """
    Annualized Sharpe Ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return (
        excess_returns.mean() * periods_per_year
    ) / (returns.std() * np.sqrt(periods_per_year))


def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Maximum Drawdown
    """
    cumulative_max = portfolio_values.cummax()
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    return drawdown.min()