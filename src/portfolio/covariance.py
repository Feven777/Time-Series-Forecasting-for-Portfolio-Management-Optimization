import pandas as pd


def compute_return_matrix(
    df: pd.DataFrame,
    assets: list,
) -> pd.DataFrame:
    """
    Compute daily returns for selected assets.

    Parameters
    ----------
    df : pd.DataFrame
        Historical price data
    assets : list
        Asset tickers (e.g. ["TSLA", "SPY", "BND"])

    Returns
    -------
    pd.DataFrame
        Daily return matrix
    """

    returns = {}

    for asset in assets:
        price_col = f"{asset}_adjclose"
        returns[asset] = df[price_col].pct_change()

    return pd.DataFrame(returns).dropna()


def compute_covariance_matrix(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute covariance matrix of asset returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily return matrix

    Returns
    -------
    pd.DataFrame
        Covariance matrix
    """

    return returns.cov()
