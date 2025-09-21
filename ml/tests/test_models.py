"""Ensure packaged ML artifacts load and run basic inference."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.datasets import load_expression_table
from ml.models import load_classifier, load_scaler, load_trained_vae
from ml.utils.math_ops import project_rows

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "models" / "artifacts"


def test_artifact_files_exist() -> None:
    assert (ARTIFACT_DIR / "simple_vae.json").exists()
    assert (ARTIFACT_DIR / "cell_type_classifier.json").exists()
    assert (ARTIFACT_DIR / "scaler.json").exists()


def test_inference_pipeline() -> None:
    _, _, features, _ = load_expression_table()

    scaler = load_scaler(ARTIFACT_DIR / "scaler.json")
    normalized = scaler.transform(features)

    vae = load_trained_vae(ARTIFACT_DIR / "simple_vae.json")
    latent = project_rows(normalized, vae.weights)

    classifier = load_classifier(ARTIFACT_DIR / "cell_type_classifier.json")
    predictions = classifier.predict(latent[:5])

    assert len(predictions) == 5
    assert set(predictions).issubset(set(classifier.classes_))
