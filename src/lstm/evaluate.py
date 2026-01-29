import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.lstm.dataset import LSTMDataset


def evaluate_lstm(
    data_path: str = "data/processed/combined.csv",
    model_path: str = "models/lstm_tsla.keras",
    lookback: int = 30,
):
    # Load dataset
    dataset = LSTMDataset(
        data_path=data_path,
        lookback=lookback,
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = dataset.prepare_data()

    # Load trained model
    model = load_model(model_path)

    # Predict
    y_pred_scaled = model.predict(X_test)

    # Inverse scaling
    y_pred = dataset.scaler_y.inverse_transform(y_pred_scaled)
    y_true = dataset.scaler_y.inverse_transform(y_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"LSTM RMSE: {rmse:.4f}")
    print(f"LSTM MAE:  {mae:.4f}")

    # Create time index for plotting
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    test_index = df.index[-len(y_true):]

    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(test_index, y_true, label="Actual TSLA Price")
    plt.plot(test_index, y_pred, label="Predicted TSLA Price")
    plt.title("LSTM Forecast vs Actual TSLA Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return rmse, mae


if __name__ == "__main__":
    evaluate_lstm()
