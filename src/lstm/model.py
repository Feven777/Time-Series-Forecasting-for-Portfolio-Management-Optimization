from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm_model(
    input_shape: tuple,
) -> Sequential:
    """
    Builds and compiles an LSTM model for time series forecasting.

    Parameters
    ----------
    input_shape : tuple
        Shape of input data (lookback, num_features)

    Returns
    -------
    model : Sequential
        Compiled LSTM model
    """

    model = Sequential()

    # First LSTM layer
    model.add(
        LSTM(
            units=64,
            return_sequences=True,
            input_shape=input_shape,
        )
    )
    model.add(Dropout(0.2))

    # Second LSTM layer
    model.add(
        LSTM(
            units=32,
            return_sequences=False,
        )
    )
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse",
    )

    return model
