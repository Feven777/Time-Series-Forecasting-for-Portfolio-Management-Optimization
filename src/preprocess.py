"""
Preprocess financial time-series data into a clean, model-ready dataset.

- Handles yfinance multi-row CSV headers
- Enforces business-day frequency
- Computes returns and rolling volatility
- Outputs a single combined dataset for modeling
"""

import os
import pandas as pd
import numpy as np

# =========================
# Paths & configuration
# =========================

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")

ASSETS = ["TSLA", "SPY", "BND"]


# =========================
# Data loading
# =========================

def load_asset(ticker: str) -> pd.DataFrame:
    """
    Load a yfinance CSV file with multi-row headers and return
    a clean DataFrame indexed by Date.

    Expected raw CSV structure (example):
    Row 0: Price, Close, High, Low, Open, Volume
    Row 1: Ticker, TSLA, TSLA, TSLA, TSLA, TSLA
    Row 2: Date, , , , ,
    Row 3+: actual data
    """

    path = os.path.join(DATA_DIR, f"{ticker}.csv")

    # Skip metadata rows and assign correct column names
    df = pd.read_csv(
        path,
        skiprows=3,
        names=["Date", "Close", "High", "Low", "Open", "Volume"]
    )

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop rows where Date could not be parsed
    df = df.dropna(subset=["Date"])

    # Set Date as index
    df = df.set_index("Date")

    # Ensure chronological order
    df = df.sort_index()

    return df


# =========================
# Preprocessing utilities
# =========================

def enforce_business_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex to business-day frequency to ensure equal time spacing.
    """
    return df.asfreq("B")


def forward_fill_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill price-related columns.
    This is safe for prices but NEVER applied to returns.
    """
    price_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[price_cols] = df[price_cols].ffill()
    return df


def compute_returns(price_series: pd.Series) -> pd.Series:
    """
    Compute daily percentage returns.
    """
    return price_series.pct_change()


def compute_rolling_volatility(
    returns: pd.Series,
    window: int = 21
) -> pd.Series:
    """
    Compute annualized rolling volatility.
    """
    return returns.rolling(window).std() * np.sqrt(252)


# =========================
# Main preprocessing pipeline
# =========================

def preprocess_all_assets() -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    - Load raw CSVs
    - Align to business days
    - Forward-fill prices
    - Compute returns and volatility
    - Combine all assets into one DataFrame
    """

    processed_frames = []

    for asset in ASSETS:
        df = load_asset(asset)
        df = enforce_business_days(df)
        df = forward_fill_prices(df)

        returns = compute_returns(df["Close"])
        volatility = compute_rolling_volatility(returns)

        processed = pd.DataFrame(
            {
                f"{asset}_adjclose": df["Close"],
                f"{asset}_return": returns,
                f"{asset}_vol": volatility,
            }
        )

        processed_frames.append(processed)

    # Combine all assets on the date index
    combined = pd.concat(processed_frames, axis=1)

    # Drop rows where TSLA price is missing (core asset)
    combined = combined.dropna(subset=["TSLA_adjclose"])

    # Save output
    output_path = os.path.join(DATA_DIR, "combined.csv")
    combined.to_csv(output_path)

    return combined


# =========================
# Script entry point
# =========================

if __name__ == "__main__":
    df = preprocess_all_assets()
    print("Preprocessing complete.")
    print(df.tail())
