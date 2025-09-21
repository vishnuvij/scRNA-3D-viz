"""Pydantic models for the backend API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class DatasetSummary(BaseModel):
    """Metadata summary for an available dataset."""

    dataset_id: str
    cell_count: int
    gene_count: int


class EmbeddingPoint(BaseModel):
    """A single 3D embedding point with optional annotations."""

    cell_id: str
    x: float
    y: float
    z: float
    cluster: int
    annotation: Optional[str] = None


class EmbeddingResponse(BaseModel):
    """Response payload for embedding requests."""

    dataset_id: str
    embedding: List[EmbeddingPoint]


class AnnotationRequest(BaseModel):
    """Payload used to annotate a single cell."""

    dataset_id: str
    cell_id: str
    label: str


class AnnotationUpdate(BaseModel):
    """Annotation broadcast message for WebSocket clients."""

    dataset_id: str
    cell_id: str
    label: str


class InferenceRequest(BaseModel):
    """Inference request using either a cell identifier or expression vector."""

    dataset_id: str
    cell_id: Optional[str] = None
    expression: Optional[List[float]] = None

    def ensure_payload(self) -> None:
        """Validate that the request contains enough information for inference."""

        if self.cell_id is None and self.expression is None:
            raise ValueError("Either cell_id or expression must be provided")


class InferenceResponse(BaseModel):
    """Inference response containing predicted cell type and latent representation."""

    dataset_id: str
    cell_id: Optional[str]
    predicted_type: str
    latent_vector: List[float]
