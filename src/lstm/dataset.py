import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


class LSTMDataset:
    """
    Prepares time series data for LSTM models using sliding windows.
    """

    def __init__(
        self,
        data_path: str,
        lookback: int = 30,
        test_size: float = 0.1,
        val_size: float = 0.1,
    ):
        self.data_path = data_path
        self.lookback = lookback
        self.test_size = test_size
        self.val_size = val_size

        self.feature_cols = [
            "TSLA_adjclose",
            "SPY_adjclose",
            "BND_adjclose",
        ]
        self.target_col = "TSLA_adjclose"

        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        return df[self.feature_cols]

    def create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_seq, y_seq = [], []

        for i in range(self.lookback, len(X)):
            X_seq.append(X[i - self.lookback : i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def split_data(self, X: np.ndarray, y: np.ndarray):
        total_len = len(X)
        test_len = int(total_len * self.test_size)
        val_len = int(total_len * self.val_size)

        X_train = X[: -(test_len + val_len)]
        y_train = y[: -(test_len + val_len)]

        X_val = X[-(test_len + val_len) : -test_len]
        y_val = y[-(test_len + val_len) : -test_len]

        X_test = X[-test_len:]
        y_test = y[-test_len:]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare_data(self):
        df = self.load_data()

        X = df.values
        y = df[[self.target_col]].values

        # Split BEFORE scaling (avoid leakage)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        # Fit scalers only on training data
        self.scaler_X.fit(X_train)
        self.scaler_y.fit(y_train)

        X_train_scaled = self.scaler_X.transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)

        y_train_scaled = self.scaler_y.transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        y_test_scaled = self.scaler_y.transform(y_test)

        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(
            X_train_scaled, y_train_scaled
        )
        X_val_seq, y_val_seq = self.create_sequences(
            X_val_scaled, y_val_scaled
        )
        X_test_seq, y_test_seq = self.create_sequences(
            X_test_scaled, y_test_scaled
        )

        return (
            X_train_seq,
            X_val_seq,
            X_test_seq,
            y_train_seq,
            y_val_seq,
            y_test_seq,
        )
