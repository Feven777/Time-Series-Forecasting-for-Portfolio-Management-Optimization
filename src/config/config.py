from dataclasses import dataclass


@dataclass
class DataConfig:
    start_date: str = "2015-01-01"
    end_date: str = "2026-01-15"
    assets: list[str] = ("TSLA", "BND", "SPY")


@dataclass
class LSTMConfig:
    window_size: int = 60
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001


@dataclass
class ARIMAConfig:
    p: int = 5
    d: int = 1
    q: int = 0


@dataclass
class PortfolioConfig:
    risk_free_rate: float = 0.02
    benchmark_weights: dict = None

    def __post_init__(self):
        if self.benchmark_weights is None:
            self.benchmark_weights = {
                "SPY": 0.6,
                "BND": 0.4
            }
