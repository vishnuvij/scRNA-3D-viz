"""Cell type classification utilities without external dependencies."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def _squared_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((ax - bx) ** 2 for ax, bx in zip(a, b))


@dataclass
class CellTypeClassifier:
    """Nearest-prototype classifier used for inference."""

    prototypes: Dict[str, List[float]]

    def predict(self, data: Iterable[Sequence[float]]) -> List[str]:
        labels = []
        for vector in data:
            label = min(
                self.prototypes.items(),
                key=lambda item: _squared_distance(vector, item[1]),
            )[0]
            labels.append(label)
        return labels

    @property
    def classes_(self) -> List[str]:
        return list(self.prototypes.keys())


def load_classifier(path: Path) -> CellTypeClassifier:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return CellTypeClassifier(prototypes=payload["prototypes"])


def save_classifier(classifier: CellTypeClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"prototypes": classifier.prototypes}, handle)


__all__ = ["CellTypeClassifier", "load_classifier", "save_classifier"]
