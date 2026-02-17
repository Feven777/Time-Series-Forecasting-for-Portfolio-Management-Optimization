import pandas as pd


def compute_expected_returns(
    df: pd.DataFrame,
    tsla_forecast_return: float,
    assets=("TSLA", "SPY", "BND"),
    trading_days: int = 252,
) -> pd.Series:
    """
    Compute expected annual returns.

    TSLA: forecast-based
    SPY & BND: historical mean returns (annualized)
    """

    expected_returns = {}

    for asset in assets:
        if asset == "TSLA":
            expected_returns[asset] = tsla_forecast_return * trading_days
        else:
            daily_mean = df[f"{asset}_return"].mean()
            expected_returns[asset] = daily_mean * trading_days

    return pd.Series(expected_returns)
