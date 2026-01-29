import os
import sys
from pathlib import Path

# --------------------------------------------------
# Make project root importable (fixes `src` imports)
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# --------------------------------------------------
# Imports
# --------------------------------------------------
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.lstm.dataset import LSTMDataset
from src.lstm.model import build_lstm_model


def train_lstm(
    data_path: str = "data/processed/combined.csv",
    lookback: int = 30,
    batch_size: int = 32,
    epochs: int = 50,
    model_dir: str = "models",
):
    """
    Train an LSTM model for time-series forecasting.
    """

    # --------------------------------------------------
    # Prepare dataset
    # --------------------------------------------------
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

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )

    # ✅ MUST compile before training
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    model.summary()

    # --------------------------------------------------
    # Model saving
    # --------------------------------------------------
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "lstm_tsla.keras")

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint],
        verbose=1,
    )

    # --------------------------------------------------
    # Evaluate on test set
    # --------------------------------------------------
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest MAE: {test_mae:.4f}")

    return model, history, dataset


if __name__ == "__main__":
    train_lstm()
