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

def compute_return_matrix(df: pd.DataFrame, assets: list = None) -> pd.DataFrame:
    """
    Compute daily returns for selected assets in df.
    Automatically handles '_adjclose' suffix in column names.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price columns (e.g., 'TSLA_adjclose')
    assets : list, optional
        List of tickers to include (e.g., ['TSLA','SPY','BND'])
        If None, use all columns.

    Returns
    -------
    pd.DataFrame
        Daily returns of selected assets
    """
    if assets is not None:
        # Map tickers to '_adjclose' columns
        cols = [f"{asset}_adjclose" for asset in assets]
        df = df[cols]
        df.columns = assets  # rename back to simple tickers
    return df.pct_change().dropna()


