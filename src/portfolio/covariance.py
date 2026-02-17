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
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Compute annualized covariance matrix.
    """
    return returns.cov() * trading_days