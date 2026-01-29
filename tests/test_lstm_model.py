import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.lstm.model import build_lstm_model


def main():
    model = build_lstm_model(input_shape=(30, 3))
    model.summary()


if __name__ == "__main__":
    main()
