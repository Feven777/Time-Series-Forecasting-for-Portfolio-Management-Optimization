import numpy as np
import pandas as pd


def compute_expected_returns(
    df: pd.DataFrame,
    forecasts: dict,
) -> pd.Series:
    """
    Compute expected returns from price forecasts.

    Parameters
    ----------
    df : pd.DataFrame
        Historical price data
    forecasts : dict
        Forecasted prices for each asset

    Returns
    -------
    pd.Series
        Expected returns for each asset
    """

    expected_returns = {}

    for asset, forecast_price in forecasts.items():
        latest_price = df[f"{asset}_adjclose"].iloc[-1]
        expected_return = (forecast_price - latest_price) / latest_price
        expected_returns[asset] = expected_return

    return pd.Series(expected_returns)
