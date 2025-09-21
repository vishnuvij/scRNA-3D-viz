"""Integration helpers that bridge the backend API and the ML package."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from ml.models.classifier import load_classifier
from ml.models.preprocessing import load_scaler
from ml.models.vae import load_trained_vae


class MLInferenceService:
    """Load trained models and expose convenience inference helpers."""

    def __init__(self, artifact_dir: Path) -> None:
        self._artifact_dir = artifact_dir
        self._scaler = load_scaler(artifact_dir / "scaler.json")
        self._vae = load_trained_vae(artifact_dir / "simple_vae.json")
        self._classifier = load_classifier(artifact_dir / "cell_type_classifier.json")

    # ------------------------------------------------------------------
    def infer(self, expression: List[float]) -> Tuple[str, List[float]]:
        """Run the classifier and return the predicted cell type and latent vector."""

        normalized = self._scaler.transform([expression])
        latent_vector = self._vae.encode_latent(normalized)[0]
        prediction = self._classifier.predict([latent_vector])[0]
        return prediction, latent_vector

    # ------------------------------------------------------------------
    @property
    def metadata(self) -> Dict[str, object]:
        """Return information about the loaded ML assets."""

        return {
            "latent_dim": int(self._vae.latent_dim),
            "input_dim": int(self._vae.input_dim),
            "classes": list(self._classifier.classes_),
        }
