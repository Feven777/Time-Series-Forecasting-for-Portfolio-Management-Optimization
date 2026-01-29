import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Make project root importable
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.lstm.dataset import LSTMDataset


def forecast_tsla_expected_return(
    data_path: str = "data/processed/combined.csv",
    model_path: str = "models/lstm_tsla.keras",
    lookback: int = 30,
    forecast_horizon: int = 252,  # 12 months
    output_path: str = "data/processed/tsla_expected_return.csv",
):
    """
    Forecast TSLA prices using trained LSTM and compute
    annualized expected return (Task 3 → Task 4).
    """

    # --------------------------------------------------
    # Load dataset and model
    # --------------------------------------------------
    dataset = LSTMDataset(
        data_path=data_path,
        lookback=lookback,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = dataset.prepare_data()
    model = load_model(model_path)

    # --------------------------------------------------
    # Start from last known window
    # --------------------------------------------------
    last_window = X_test[-1].copy()
    future_prices = []

    for _ in range(forecast_horizon):
        pred_scaled = model.predict(
            last_window.reshape(1, lookback, last_window.shape[1]),
            verbose=0,
        )

        pred_price = dataset.scaler_y.inverse_transform(pred_scaled)[0, 0]
        future_prices.append(pred_price)

        # Update window (roll forward)
        last_window = np.roll(last_window, shift=-1, axis=0)
        last_window[-1, 0] = pred_scaled[0, 0]  # TSLA target

    # --------------------------------------------------
    # Convert prices → returns
    # --------------------------------------------------
    future_prices = np.array(future_prices)
    daily_returns = np.diff(future_prices) / future_prices[:-1]

    expected_daily_return = daily_returns.mean()
    expected_annual_return = expected_daily_return * 252

    # --------------------------------------------------
    # Save result
    # --------------------------------------------------
    result = pd.DataFrame(
        {
            "asset": ["TSLA"],
            "expected_annual_return": [expected_annual_return],
        }
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("TSLA expected annual return saved:")
    print(result)

    return expected_annual_return


if __name__ == "__main__":
    forecast_tsla_expected_return()
