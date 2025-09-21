"""Train the linear VAE approximation and export artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from ml.data.datasets import load_expression_table, map_clusters_to_types
from ml.models import (
    CellTypeClassifier,
    SimpleVAE,
    StandardScalerModel,
    save_classifier,
    save_scaler,
    save_vae,
)
from ml.utils.math_ops import (
    covariance_matrix,
    mean,
    project_rows,
    std,
    subtract_rank_one,
    transpose,
    power_iteration,
)


def fit_scaler(features: List[List[float]]) -> StandardScalerModel:
    columns = transpose(features)
    means = [mean(column) for column in columns]
    scales = [std(column, ddof=1) or 1.0 for column in columns]
    return StandardScalerModel(means=means, scales=scales)


def scale_features(features: List[List[float]], scaler: StandardScalerModel) -> List[List[float]]:
    return scaler.transform(features)


def compute_components(normalized: List[List[float]], latent_dim: int) -> List[List[float]]:
    cov = covariance_matrix(normalized)
    components: List[List[float]] = []
    for _ in range(latent_dim):
        eigenvalue, eigenvector = power_iteration(cov)
        components.append(eigenvector)
        subtract_rank_one(cov, eigenvalue, eigenvector)
    return components


def compute_prototypes(latent: List[List[float]], labels: List[str]) -> Dict[str, List[float]]:
    accumulator: Dict[str, List[float]] = {}
    counts: Dict[str, int] = {}
    for vector, label in zip(latent, labels):
        if label not in accumulator:
            accumulator[label] = [0.0 for _ in vector]
            counts[label] = 0
        counts[label] += 1
        for idx, value in enumerate(vector):
            accumulator[label][idx] += value
    for label, count in counts.items():
        accumulator[label] = [value / count for value in accumulator[label]]
    return accumulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Train simplified VAE and classifier")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "artifacts",
    )
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args()

    cell_ids, gene_columns, features, clusters = load_expression_table(args.dataset)
    scaler = fit_scaler(features)
    normalized = scale_features(features, scaler)
    components = compute_components(normalized, latent_dim=args.latent_dim)

    vae = SimpleVAE(weights=components, bias=[0.0] * args.latent_dim)
    latent = project_rows(normalized, components)

    labels = map_clusters_to_types(clusters)
    prototypes = compute_prototypes(latent, labels)
    classifier = CellTypeClassifier(prototypes=prototypes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_scaler(scaler, args.output_dir / "scaler.json")
    save_vae(vae, args.output_dir / "simple_vae.json")
    save_classifier(classifier, args.output_dir / "cell_type_classifier.json")


if __name__ == "__main__":
    main()
