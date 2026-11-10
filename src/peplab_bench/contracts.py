"""Dependency-free integration contracts. No assay files are given to a model."""
from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Protocol, Sequence

@dataclass(frozen=True)
class Prediction:
    ids: tuple[str, ...]
    means: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...] | None = None
    observation_noise: float = 0.0
    uncertainty_label: str = "unavailable"

    def validate(self, expected: Sequence[str]) -> None:
        if self.ids != tuple(expected) or len(set(self.ids)) != len(self.ids):
            raise ValueError("Prediction IDs must exactly match query order")
        if len(self.means) != len(self.ids) or not all(isfinite(x) for x in self.means):
            raise ValueError("Nonfinite or missing predictive means")
        if not isfinite(self.observation_noise) or self.observation_noise < 0:
            raise ValueError("Invalid observation noise")
        if self.covariance is not None:
            n = len(self.ids)
            if len(self.covariance) != n or any(len(r) != n for r in self.covariance):
                raise ValueError("Wrong covariance shape")
            if not all(isfinite(v) for row in self.covariance for v in row):
                raise ValueError("Nonfinite covariance")
            if any(self.covariance[i][i] < -1e-8 for i in range(n)):
                raise ValueError("Negative marginal variance")
            if any(abs(self.covariance[i][j] - self.covariance[j][i]) > 1e-8 for i in range(n) for j in range(n)):
                raise ValueError("Asymmetric covariance")

class Predictor(Protocol):
    def predict(self, features: Mapping[str, object], visible: Mapping[str, float],
                query_ids: Sequence[str], sources: Sequence[str], seed: int) -> Prediction: ...

class Selector(Protocol):
    def select(self, prediction: Prediction, visible_values: Sequence[float],
               batch_size: int, seed: int) -> Sequence[str]: ...

def greedy(prediction: Prediction, batch_size: int) -> list[str]:
    prediction.validate(prediction.ids)
    if not 0 < batch_size <= len(prediction.ids):
        raise ValueError("Invalid batch size")
    return [prediction.ids[i] for i in sorted(range(len(prediction.ids)), key=lambda i: (-prediction.means[i], i))[:batch_size]]

def supported_selectors(prediction: Prediction) -> tuple[str, ...]:
    return ("greedy",) if prediction.covariance is None else ("greedy", "EI", "UCB", "TS", "top8utility")
