"""Lightweight numerical helpers implemented with pure Python."""
from __future__ import annotations

import random
from math import sqrt
from typing import Iterable, List, Sequence, Tuple


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: Sequence[float], ddof: int = 0) -> float:
    if len(values) <= ddof:
        return 0.0
    m = mean(values)
    variance = sum((value - m) ** 2 for value in values) / (len(values) - ddof)
    return sqrt(variance)


def transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(column) for column in zip(*matrix)]


def covariance_matrix(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    n_samples = len(matrix)
    n_features = len(matrix[0]) if matrix else 0
    cov = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    for row in matrix:
        for i in range(n_features):
            for j in range(i, n_features):
                cov[i][j] += row[i] * row[j]
    for i in range(n_features):
        for j in range(i, n_features):
            value = cov[i][j] / max(1, n_samples - 1)
            cov[i][j] = value
            cov[j][i] = value
    return cov


def matrix_vector_product(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    return [sum(m * v for m, v in zip(row, vector)) for row in matrix]


def normalize(vector: Sequence[float]) -> List[float]:
    norm = sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def power_iteration(matrix: Sequence[Sequence[float]], num_iterations: int = 100) -> Tuple[float, List[float]]:
    n = len(matrix)
    vector = [random.random() for _ in range(n)]
    vector = normalize(vector)
    for _ in range(num_iterations):
        vector = matrix_vector_product(matrix, vector)
        vector = normalize(vector)
    eigenvalue = sum(vector[i] * sum(matrix[i][j] * vector[j] for j in range(n)) for i in range(n))
    return eigenvalue, vector


def subtract_rank_one(matrix: List[List[float]], eigenvalue: float, eigenvector: Sequence[float]) -> None:
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j]


def project_rows(rows: Sequence[Sequence[float]], components: Sequence[Sequence[float]]) -> List[List[float]]:
    projected = []
    for row in rows:
        latent = [sum(component[i] * row[i] for i in range(len(row))) for component in components]
        projected.append(latent)
    return projected


def kmeans(points: Sequence[Sequence[float]], k: int, iterations: int = 25) -> List[int]:
    if not points:
        return []
    centroids = [list(points[i]) for i in range(min(k, len(points)))]
    assignments = [0] * len(points)
    for _ in range(iterations):
        # Assignment step
        for idx, point in enumerate(points):
            assignments[idx] = min(
                range(len(centroids)),
                key=lambda c: sum((point[d] - centroids[c][d]) ** 2 for d in range(len(point))),
            )
        # Update step
        for c in range(len(centroids)):
            cluster_points = [points[i] for i, assignment in enumerate(assignments) if assignment == c]
            if cluster_points:
                centroids[c] = [mean(dim) for dim in zip(*cluster_points)]  # type: ignore[arg-type]
    return assignments


__all__ = [
    "covariance_matrix",
    "kmeans",
    "matrix_vector_product",
    "mean",
    "normalize",
    "power_iteration",
    "project_rows",
    "std",
    "subtract_rank_one",
    "transpose",
]
