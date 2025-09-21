from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_dataset_listing() -> None:
    response = client.get("/datasets")
    payload = response.json()
    assert response.status_code == 200
    assert any(dataset["dataset_id"] == "sample" for dataset in payload)


def test_embedding_and_annotation_flow() -> None:
    response = client.get("/datasets/sample/embedding")
    assert response.status_code == 200
    embedding = response.json()["embedding"]
    assert len(embedding) > 0
    cell_id = embedding[0]["cell_id"]

    annotate = client.post(
        "/annotations",
        json={"dataset_id": "sample", "cell_id": cell_id, "label": "TestType"},
    )
    assert annotate.status_code == 200
    assert annotate.json()["annotation"]["label"] == "TestType"


def test_inference_endpoint() -> None:
    response = client.post(
        "/inference",
        json={"dataset_id": "sample", "cell_id": "cell_000"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_type"] in {"Type-A", "Type-B", "Type-C", "Type-D"}
    assert len(payload["latent_vector"]) == 3
