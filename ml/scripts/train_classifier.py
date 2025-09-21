"""Recompute classifier prototypes from embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path

from ml.data.datasets import load_expression_table, map_clusters_to_types
from ml.models import CellTypeClassifier, load_scaler, load_trained_vae, save_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-train cell type classifier from embeddings")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "artifacts",
    )
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args()

    cell_ids, gene_columns, features, clusters = load_expression_table(args.dataset)
    scaler = load_scaler(args.artifact_dir / "scaler.json")
    vae = load_trained_vae(args.artifact_dir / "simple_vae.json")

    normalized = scaler.transform(features)
    latent = vae.encode_latent(normalized)
    labels = map_clusters_to_types(clusters)

    prototypes = {}
    counts = {}
    for vector, label in zip(latent, labels):
        if label not in prototypes:
            prototypes[label] = [0.0 for _ in vector]
            counts[label] = 0
        counts[label] += 1
        for idx, value in enumerate(vector):
            prototypes[label][idx] += value
    for label, count in counts.items():
        prototypes[label] = [value / count for value in prototypes[label]]

    classifier = CellTypeClassifier(prototypes=prototypes)
    save_classifier(classifier, args.artifact_dir / "cell_type_classifier.json")


if __name__ == "__main__":
    main()
