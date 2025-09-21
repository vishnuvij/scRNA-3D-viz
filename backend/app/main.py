"""FastAPI application exposing data, annotation, and ML inference endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .data_manager import DatasetManager, DatasetNotFoundError
from .ml_integration import MLInferenceService
from .schemas import (
    AnnotationRequest,
    AnnotationUpdate,
    DatasetSummary,
    EmbeddingPoint,
    EmbeddingResponse,
    InferenceRequest,
    InferenceResponse,
)


class AnnotationConnectionManager:
    """Track websocket connections that listen for annotation updates."""

    def __init__(self) -> None:
        self.active: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast(self, message: Dict[str, object]) -> None:
        for connection in list(self.active):
            try:
                await connection.send_json(message)
            except RuntimeError:
                self.disconnect(connection)


# Initialise services -----------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
REPO_ROOT = BACKEND_DIR.parent

ARTIFACT_DIR = REPO_ROOT / "ml" / "models" / "artifacts"
_data_manager = DatasetManager(data_dir=BACKEND_DIR / "data", artifact_dir=ARTIFACT_DIR)
_ml_service = MLInferenceService(artifact_dir=ARTIFACT_DIR)
_connections = AnnotationConnectionManager()

app = FastAPI(title="scRNA-3D Visualization Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="Service health check")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/datasets", response_model=List[DatasetSummary], summary="List datasets")
def list_datasets() -> List[DatasetSummary]:
    return [DatasetSummary(**_data_manager.get_summary(ds)) for ds in _data_manager.available_datasets()]


@app.get(
    "/datasets/{dataset_id}/embedding",
    response_model=EmbeddingResponse,
    summary="Retrieve 3D embedding for a dataset",
)
def get_embedding(dataset_id: str) -> EmbeddingResponse:
    try:
        embedding = _data_manager.get_embedding(dataset_id)
    except DatasetNotFoundError as exc:  # pragma: no cover - validated via tests
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return EmbeddingResponse(dataset_id=dataset_id, embedding=[EmbeddingPoint(**item) for item in embedding])


@app.get("/datasets/{dataset_id}/metadata", summary="Dataset metadata")
def get_metadata(dataset_id: str) -> Dict[str, object]:
    try:
        return _data_manager.get_metadata(dataset_id)
    except DatasetNotFoundError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/annotations", summary="Annotate a single cell")
async def annotate_cell(request: AnnotationRequest) -> Dict[str, object]:
    try:
        _data_manager.update_annotation(request.dataset_id, request.cell_id, request.label)
    except DatasetNotFoundError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    update = AnnotationUpdate(dataset_id=request.dataset_id, cell_id=request.cell_id, label=request.label)
    await _connections.broadcast(update.model_dump())
    return {"status": "ok", "annotation": update.model_dump()}


@app.websocket("/ws/annotations")
async def websocket_annotations(websocket: WebSocket) -> None:
    await _connections.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _connections.disconnect(websocket)


@app.post("/inference", response_model=InferenceResponse, summary="Run ML inference")
async def run_inference(request: InferenceRequest) -> InferenceResponse:
    try:
        request.ensure_payload()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.expression is not None:
        expression = request.expression
    else:
        try:
            expression = _data_manager.get_expression(request.dataset_id, request.cell_id or "")
        except DatasetNotFoundError as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    predicted_type, latent_vector = _ml_service.infer(expression)
    return InferenceResponse(
        dataset_id=request.dataset_id,
        cell_id=request.cell_id,
        predicted_type=predicted_type,
        latent_vector=latent_vector,
    )


@app.get("/ml/metadata", summary="Describe loaded ML assets")
def get_ml_metadata() -> Dict[str, object]:
    return _ml_service.metadata


__all__ = ["app"]
