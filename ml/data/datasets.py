"""Utility helpers for loading training data."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

CELL_TYPE_MAP: Dict[int, str] = {
    0: "Type-A",
    1: "Type-B",
    2: "Type-C",
    3: "Type-D",
}


def dataset_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "backend" / "data" / "sample_dataset.csv"


def load_expression_table(path: Path | None = None) -> Tuple[List[str], List[str], List[List[float]], List[int]]:
    file_path = dataset_path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        gene_columns = [col for col in reader.fieldnames or [] if col not in {"cell_id", "cluster_label"}]
        cell_ids: List[str] = []
        expressions: List[List[float]] = []
        clusters: List[int] = []
        for row in reader:
            cell_ids.append(row["cell_id"])
            clusters.append(int(row.get("cluster_label", 0)))
            expressions.append([float(row[col]) for col in gene_columns])
    return cell_ids, gene_columns, expressions, clusters


def map_clusters_to_types(clusters: Sequence[int]) -> List[str]:
    return [CELL_TYPE_MAP.get(cluster, f"Type-{cluster}") for cluster in clusters]


__all__ = ["CELL_TYPE_MAP", "dataset_path", "load_expression_table", "map_clusters_to_types"]
