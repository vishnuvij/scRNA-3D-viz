"""Utilities for loading and preprocessing single-cell RNA sequencing datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from ml.data.datasets import load_expression_table
from ml.models import load_scaler, load_trained_vae
from ml.utils.math_ops import kmeans


class DatasetNotFoundError(ValueError):
    """Raised when a requested dataset is not available."""


class DatasetManager:
    """Load and preprocess available datasets for downstream services."""

    def __init__(self, data_dir: Path, artifact_dir: Path | None = None) -> None:
        self._data_dir = data_dir
        self._datasets: Dict[str, Dict[str, object]] = {}
        self._annotations: Dict[str, Dict[str, str]] = {}
        default_artifact_dir = Path(__file__).resolve().parents[2] / "ml" / "models" / "artifacts"
        self._artifact_dir = artifact_dir or default_artifact_dir
        self._scaler = load_scaler(self._artifact_dir / "scaler.json")
        self._vae = load_trained_vae(self._artifact_dir / "simple_vae.json")
        self._load_default_datasets()

    # ------------------------------------------------------------------
    def _load_default_datasets(self) -> None:
        sample_dataset = self._data_dir / "sample_dataset.csv"
        if sample_dataset.exists():
            self.register_dataset(dataset_id="sample", path=sample_dataset)

    # ------------------------------------------------------------------
    def register_dataset(self, dataset_id: str, path: Path) -> None:
        cell_ids, gene_columns, expressions, clusters = load_expression_table(path)
        normalized = self._scaler.transform(expressions)
        latent = self._vae.encode_latent(normalized)
        inferred_clusters = kmeans(latent, k=min(4, max(1, len(set(clusters)))))

        dataset_payload = {
            "cell_ids": cell_ids,
            "gene_columns": gene_columns,
            "expressions": expressions,
            "normalized": normalized,
            "latent": latent,
            "clusters": inferred_clusters,
            "original_clusters": clusters,
        }

        self._datasets[dataset_id] = dataset_payload
        self._annotations.setdefault(dataset_id, {})

    # ------------------------------------------------------------------
    def available_datasets(self) -> Iterable[str]:
        return self._datasets.keys()

    # ------------------------------------------------------------------
    def get_summary(self, dataset_id: str) -> Dict[str, object]:
        dataset = self._get_dataset(dataset_id)
        return {
            "dataset_id": dataset_id,
            "cell_count": len(dataset["cell_ids"]),
            "gene_count": len(dataset["gene_columns"]),
        }

    # ------------------------------------------------------------------
    def get_embedding(self, dataset_id: str) -> List[Dict[str, object]]:
        dataset = self._get_dataset(dataset_id)
        annotations = self._annotations.setdefault(dataset_id, {})
        latent = dataset["latent"]
        cluster_labels = dataset["clusters"]
        cell_ids = dataset["cell_ids"]

        payload = []
        for idx, cell_id in enumerate(cell_ids):
            vector = latent[idx]
            point = {
                "cell_id": cell_id,
                "x": float(vector[0] if len(vector) > 0 else 0.0),
                "y": float(vector[1] if len(vector) > 1 else 0.0),
                "z": float(vector[2] if len(vector) > 2 else 0.0),
                "cluster": int(cluster_labels[idx]),
                "annotation": annotations.get(cell_id),
            }
            payload.append(point)
        return payload

    # ------------------------------------------------------------------
    def get_metadata(self, dataset_id: str) -> Dict[str, object]:
        dataset = self._get_dataset(dataset_id)
        return {
            "gene_columns": dataset["gene_columns"],
            "original_clusters": dataset["original_clusters"],
        }

    # ------------------------------------------------------------------
    def get_expression(self, dataset_id: str, cell_id: str) -> List[float]:
        dataset = self._get_dataset(dataset_id)
        try:
            index = dataset["cell_ids"].index(cell_id)
        except ValueError as exc:  # pragma: no cover - defensive
            raise DatasetNotFoundError(f"Cell '{cell_id}' not found in dataset '{dataset_id}'") from exc
        return list(map(float, dataset["expressions"][index]))

    # ------------------------------------------------------------------
    def transform_expression(self, dataset_id: str, expression: List[float]) -> List[float]:
        normalized = self._scaler.transform([expression])
        latent = self._vae.encode_latent(normalized)
        return latent[0]

    # ------------------------------------------------------------------
    def update_annotation(self, dataset_id: str, cell_id: str, label: str) -> None:
        dataset = self._get_dataset(dataset_id)
        if cell_id not in dataset["cell_ids"]:
            raise DatasetNotFoundError(f"Cell '{cell_id}' not found in dataset '{dataset_id}'")
        self._annotations.setdefault(dataset_id, {})[cell_id] = label

    # ------------------------------------------------------------------
    def annotations(self, dataset_id: str) -> Dict[str, str]:
        return dict(self._annotations.setdefault(dataset_id, {}))

    # ------------------------------------------------------------------
    def _get_dataset(self, dataset_id: str) -> Dict[str, object]:
        try:
            return self._datasets[dataset_id]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise DatasetNotFoundError(f"Dataset '{dataset_id}' is not available") from exc
