# scRNA-3D-viz

An end-to-end environment for exploring single-cell RNA sequencing (scRNA-seq) experiments.
The repository is organised into dedicated **backend**, **frontend**, and **machine-learning**
packages so that data engineering, visualisation, and modelling can evolve independently
while sharing the same synthetic example dataset.

## Repository layout

```
.
├── backend/              # FastAPI service exposing datasets, annotations, and inference endpoints
│   ├── app/
│   │   ├── data_manager.py
│   │   ├── ml_integration.py
│   │   ├── main.py
│   │   └── schemas.py
│   ├── data/             # Example scRNA-seq matrix packaged with the repo
│   ├── requirements.txt
│   └── tests/
├── frontend/             # React + Three.js interface for interactive visualisation
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── ml/                   # Lightweight ML utilities, training scripts, and saved models
│   ├── data/
│   ├── models/
│   │   ├── artifacts/    # JSON artefacts for scaler, encoder, classifier
│   │   └── …
│   ├── scripts/
│   └── tests/
└── README.md
```

## Data and preprocessing

`backend/data/sample_dataset.csv` ships with 120 mock cells and 20 synthetic gene
measurements. The machine-learning module provides a minimal preprocessing pipeline:

* **Standardisation** – feature-wise mean and scale values are computed during training and
  stored in `ml/models/artifacts/scaler.json`.
* **Dimensionality reduction** – a linear encoder approximating a variational autoencoder
  projects each cell into a 3-dimensional latent space (`simple_vae.json`).
* **Clustering & prototypes** – latent embeddings are clustered for visualisation while
  class prototypes are saved in `cell_type_classifier.json` to support inference.

The backend consumes the exported artefacts to normalise raw counts, generate the 3D
embedding, and serve ML predictions without recomputing heavy statistics at runtime.

## Backend service

1. (Optional but recommended) create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Launch the API:
   ```bash
   uvicorn backend.app.main:app --reload
   ```
4. Available endpoints include:
   * `GET /datasets` – list datasets with summary metadata.
   * `GET /datasets/{id}/embedding` – retrieve 3D coordinates, cluster labels, and annotations.
   * `POST /annotations` – submit manual labels which are broadcast over a WebSocket channel.
   * `POST /inference` – request cell-type predictions and latent vectors via the ML module.

## Frontend application

The React interface uses Three.js to render embeddings and orchestrates API calls for
annotation and inference. To run locally (requires access to the npm registry):

```bash
cd frontend
npm install
npm run dev
```

Set the backend base URL with `VITE_BACKEND_URL` if the API is not reachable at
`http://localhost:8000`.

## Machine-learning module

Training utilities live under `ml/scripts/` and operate purely with Python’s standard
library so they run in restricted environments:

* `python -m ml.scripts.train_vae` – recomputes the scaler, encoder, and classifier artefacts.
* `python -m ml.scripts.train_classifier` – refreshes classifier prototypes while reusing
  the existing scaler/encoder.

Artefacts are stored as JSON to remain transparent and easy to version.

## Automated checks

Run the following commands from the repository root after installing the necessary toolchains:

| Component  | Command                             | Purpose                              |
|------------|--------------------------------------|--------------------------------------|
| Backend    | `pytest backend/tests/test_api.py`   | Exercises REST endpoints and flows   |
| ML module  | `pytest ml/tests/test_models.py`     | Verifies artefact loading & inference|
| Frontend   | `npm run lint` (inside `frontend/`)  | ESLint for TypeScript/React code     |

> **Note:** The execution environment used to author this repository does not provide internet
> access, therefore Python and Node packages could not be installed and the backend/frontend
> checks currently fail with missing dependency errors. The ML tests succeed because they only
> rely on the standard library. When running locally with network access, install the listed
> dependencies before executing the commands above.

## Development roadmap

* Support loading external `.csv` or `.h5ad` datasets via configurable data directories.
* Extend the ML module with alternative encoders (UMAP, graph autoencoders) and richer
  classifiers.
* Add screenshot/recording utilities on the frontend to export annotated scenes.
* Harden the WebSocket workflow with authentication and persistence.
