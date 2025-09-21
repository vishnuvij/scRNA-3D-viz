"""Variational autoencoder approximation using linear projections."""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import tanh
from pathlib import Path
from typing import Iterable, List


def _dot(a: List[float], b: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class SimpleVAE:
    """Linear encoder that mimics a variational autoencoder."""

    weights: List[List[float]]
    bias: List[float]

    @property
    def input_dim(self) -> int:
        return len(self.weights[0]) if self.weights else 0

    @property
    def latent_dim(self) -> int:
        return len(self.weights)

    def encode_latent(self, rows: Iterable[Iterable[float]]) -> List[List[float]]:
        encoded: List[List[float]] = []
        for row in rows:
            latent = []
            for component, bias in zip(self.weights, self.bias):
                activation = _dot(component, row) + bias
                latent.append(tanh(activation))
            encoded.append(latent)
        return encoded


def load_trained_vae(path: Path) -> SimpleVAE:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SimpleVAE(weights=payload["weights"], bias=payload.get("bias", [0.0] * len(payload["weights"])))


def save_vae(model: SimpleVAE, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"weights": model.weights, "bias": model.bias}, handle)


__all__ = ["SimpleVAE", "load_trained_vae", "save_vae"]
