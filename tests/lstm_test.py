import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.lstm.dataset import LSTMDataset

def main():
    dataset = LSTMDataset(
        data_path="data/processed/combined.csv",
        lookback=30,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = dataset.prepare_data()

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

if __name__ == "__main__":
    main()
