"""Preprocessing assets used by ML models."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class StandardScalerModel:
    """Minimal standardization transform."""

    means: List[float]
    scales: List[float]
    epsilon: float = 1e-8

    def transform(self, rows: Iterable[Iterable[float]]) -> List[List[float]]:
        transformed = []
        for row in rows:
            scaled_row = []
            for value, mean, scale in zip(row, self.means, self.scales):
                denominator = scale if abs(scale) > self.epsilon else 1.0
                scaled_row.append((float(value) - mean) / denominator)
            transformed.append(scaled_row)
        return transformed


def load_scaler(path: Path) -> StandardScalerModel:
    """Load scaler parameters stored as JSON."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return StandardScalerModel(
        means=payload["means"],
        scales=payload["scales"],
        epsilon=payload.get("epsilon", 1e-8),
    )


def save_scaler(model: StandardScalerModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"means": model.means, "scales": model.scales, "epsilon": model.epsilon}, handle)


__all__ = ["StandardScalerModel", "load_scaler", "save_scaler"]
