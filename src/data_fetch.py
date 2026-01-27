"""
Fetch historical financial data for TSLA, SPY, and BND
Source: Yahoo Finance
"""

import os
from datetime import datetime
import yfinance as yf
import pandas as pd

# Root directory
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")

TICKERS = ["TSLA", "SPY", "BND"]
START_DATE = "2015-01-01"
END_DATE = "2026-01-15"


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def fetch_ticker(ticker: str) -> pd.DataFrame:
    print(f"Fetching {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE)

    if df.empty:
        raise ValueError(f"No data fetched for {ticker}")

    df.index = pd.to_datetime(df.index)
    return df


def save_to_csv(df: pd.DataFrame, ticker: str):
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(path)
    print(f"Saved {ticker} data to {path}")


def main():
    ensure_data_dir()

    for ticker in TICKERS:
        df = fetch_ticker(ticker)
        save_to_csv(df, ticker)


if __name__ == "__main__":
    main()
