import pandas as pd
import numpy as np

from src.portfolio.metrics import (
    calculate_cagr,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)


def test_cagr_positive_growth():
    values = pd.Series([1, 1.1, 1.2])
    cagr = calculate_cagr(values, periods_per_year=1)
    assert cagr > 0


def test_volatility_zero_for_constant_returns():
    returns = pd.Series([0.01, 0.01, 0.01])
    vol = calculate_annualized_volatility(returns, periods_per_year=1)
    assert np.isclose(vol, 0)


def test_sharpe_ratio_positive():
    returns = pd.Series([0.02, 0.01, 0.03])
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=1)
    assert sharpe > 0


def test_max_drawdown_negative():
    values = pd.Series([1.0, 1.2, 0.8, 1.1])
    mdd = calculate_max_drawdown(values)
    assert mdd < 0


def test_metrics_handle_realistic_data():
    returns = pd.Series(np.random.normal(0.01, 0.02, 50))
    values = (1 + returns).cumprod()

    assert calculate_cagr(values) is not None
    assert calculate_max_drawdown(values) <= 0