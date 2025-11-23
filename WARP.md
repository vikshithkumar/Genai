# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

This repo contains an MVP for AI-based financial transaction categorization built around a FastAPI backend, a simple static UI, and an embedding-based k-NN classifier trained on synthetic data.

Key pieces:
- **Backend & ML logic** in `mvp/src` (preprocessing, embeddings, training, inference, explainability, API).
- **Configuration** in `mvp/config` (taxonomy, thresholds, random seed, etc.).
- **Data & artifacts** in `mvp/data` and `mvp/models` (generated CSVs, corrections, potential model outputs).
- **Evaluation pipeline** in `mvp/evaluation` (offline metrics and plots).
- **Static UI** in `mvp/ui` (browser-based interface talking to the FastAPI API).

All commands below assume the working directory is the repo root (`ai-financial-tx-categorization-mvp`).

## Environment & Dependencies

Python project with dependencies managed via `requirements.txt` at the repo root. Core libraries include FastAPI/uvicorn, `sentence-transformers`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, and `seaborn`.

Install dependencies (ideally in a virtual environment):

```bash
pip install -r requirements.txt
```

## Core Workflows & Commands

### 1. Generate synthetic dataset

Generates synthetic financial transaction data for training/validation/testing.

- Command:

  ```bash
  python -m mvp.src.generate_synthetic_dataset
  ```

- Outputs:
  - CSVs under `mvp/data/generated/`: `train.csv`, `val.csv`, `test.csv`.
  - Each row includes `description`, `amount`, `merchant`, `category_id`.

This script should be run before training/inference if the generated data is not already present.

### 2. Run the API server

Starts the FastAPI app that exposes REST endpoints for prediction, taxonomy management, corrections, and evaluation artifact access.

- Command (from repo root):

  ```bash
  uvicorn mvp.src.api:app --reload
  ```

- By default, uvicorn serves on `http://127.0.0.1:8000`.
- First request will incur extra latency due to loading the sentence-transformers model and training the k-NN classifier on the synthetic data.

### 3. Use the UI

The UI is served as static files by FastAPI.

- After starting the API server as above, open:
  - `http://localhost:8000/ui/index.html`

The UI supports single predictions, batch CSV upload, taxonomy administration, and submitting corrections.

### 4. Run evaluation

Runs an offline evaluation on the synthetic test split and writes metrics and a confusion matrix plot.

- Command:

  ```bash
  python -m mvp.evaluation.evaluate
  ```

- Outputs (under `mvp/evaluation/`):
  - `confusion_matrix.png`
  - `per_class_metrics.json`
  - `macro_f1_score.json`
  - `performance_notes.md` (created on first run with a brief summary of macro F1 and pointers to the confusion matrix).

### 5. Testing & linting

- There is **no automated test suite or linting configuration** (e.g., `pytest`, `tox`, `flake8`, `ruff`) defined in this repo as of this snapshot.
- If you introduce tests or linters, prefer to add explicit commands (e.g., via `Makefile` or dedicated scripts) and update this section accordingly.

## High-Level Architecture

### Configuration & Paths (`mvp/src/config_loader.py`)

- `BASE_DIR` is the `mvp` package root; key directories are derived from it:
  - `data`: `mvp/data`
  - `models`: `mvp/models`
  - `config`: `mvp/config`
  - `ui`: `mvp/ui`
  - `evaluation`: `mvp/evaluation`
- `project_paths()` centralizes these paths and is used throughout the codebase (API, model training, feedback, evaluation) to avoid hardcoding locations.
- `Taxonomy` (Pydantic model):
  - Holds a list of `Category` objects (`id`, `name`, `description`, `aliases`).
  - Provides helper methods:
    - `id_to_name()` for mapping category IDs to human-readable names.
    - `alias_to_id()` for resolving lowercase aliases to canonical IDs (used by other components like dataset generation or potential UI logic).
- `Settings` (Pydantic model):
  - `low_confidence_threshold`: probability threshold below which predictions are mapped to an `UNKNOWN` category.
  - `random_seed`: used to control reproducibility (e.g., in dataset generation and potentially model training).
  - `k_neighbors`: number of neighbors in the k-NN classifier.
- `load_taxonomy()` and `load_settings()` load JSON from `mvp/config/taxonomy.json` and `mvp/config/settings.json`, with reasonable defaults if `settings.json` is missing.

This module is the single source of truth for configuration and paths; other modules import from here instead of re-resolving directories.

### Preprocessing & Embeddings (`mvp/src/preprocessing.py`, `mvp/src/embeddings.py`)

**Preprocessing**:
- `normalize_text` performs deterministic normalization of transaction descriptions:
  - Lowercasing and trimming.
  - Removing trailing digits (e.g., POS terminal IDs).
  - Replacing non-alphanumeric characters with spaces.
  - Collapsing repeated whitespace.
- `normalize_batch` applies `normalize_text` over iterables of strings.

**Embeddings**:
- Uses `sentence-transformers/all-MiniLM-L6-v2` via `SentenceTransformer`.
- `get_model()` loads and caches the transformer model with `@lru_cache(maxsize=1)` so the heavy model load only occurs once per process.
- `generate_embeddings`:
  - Normalizes input texts (via `normalize_batch`).
  - Encodes to dense vectors using the SBERT model.
  - Returns `float32` numpy arrays, suitable for scikit-learn and FAISS usage.

These modules define a consistent feature pipeline that both training (`model.py`, `evaluation/evaluate.py`) and inference (`inference.py`) rely on.

### Model Training & Prediction (`mvp/src/model.py`)

- `load_training_data()` reads `train.csv` from `mvp/data/generated/` using `project_paths()`.
- `ModelArtifacts` bundles together:
  - The fitted `KNeighborsClassifier` instance.
  - Label-index mappings (`label_to_index`, `index_to_label`).
  - Training embeddings, texts, and labels (used for explainability/nearest-neighbor queries).
- `train_classifier()`:
  - Loads settings (for `k_neighbors`).
  - Loads and normalizes training texts.
  - Generates embeddings via `generate_embeddings`.
  - Trains a cosine-distance k-NN classifier and returns `ModelArtifacts`.
- `predict_proba` / `batch_predict_proba`:
  - Generate embeddings for incoming text(s).
  - Use the classifier’s `predict_proba` to compute class probabilities.
  - Return predicted labels, confidences, and full probability distributions for downstream logic.

Training is intentionally lightweight and re-runnable; the classifier is (re)trained in memory when needed rather than persisted to disk.

### Explainability (`mvp/src/explainability.py`)

- `top_k_neighbors`:
  - Uses cosine similarity between a query embedding and stored `train_embeddings` from `ModelArtifacts`.
  - Returns the top-k neighbors including original training text, category ID, and similarity.
- `keyword_matches`:
  - Normalizes the description and computes token frequencies.
  - Returns simple textual indicators like `"coffee (x2)"` for interpretability.
- `build_rationale`:
  - Produces a short, human-readable rationale referencing the predicted category, confidence, and neighbor categories.
- `explain_prediction`:
  - Ties the above together to produce an `Explanation` dataclass (neighbors, keyword matches, rationale).

These explanations are surfaced to API consumers via `PredictionResult` and ultimately the UI.

### Inference Orchestration (`mvp/src/inference.py`)

- `PredictionResult` dataclass is the canonical internal representation of a prediction, including description, category ID and name, confidence, flags for low-confidence/unknown, and an `Explanation` instance.
- Module-level caches:
  - `_model_artifacts`: trained `ModelArtifacts` from `train_classifier()`.
  - `_settings`, `_taxonomy`: loaded once via `load_settings()` and `load_taxonomy()`.
- `_ensure_loaded()` lazily populates these caches on first use; subsequent predictions reuse them.
- `_apply_threshold` applies the `low_confidence_threshold` from `Settings` and maps low-confidence predictions to `taxonomy.unknown_category_id` while flagging `is_low_confidence`.
- `_map_to_taxonomy` maps category IDs to human-readable names via `Taxonomy.id_to_name()`.
- `predict_with_confidence`:
  - Normalizes the input, generates embeddings, calls `batch_predict_proba`, applies thresholding, maps to category name, and attaches an explanation.
- `batch_inference` performs the same pipeline over a list of descriptions, including per-row explanations.

This module encapsulates the full online inference behavior the API relies on and should be the primary extension point for changes to prediction behavior.

### API Layer (`mvp/src/api.py`)

- FastAPI app (`app`) configured with permissive CORS to allow local UI usage.
- Static UI mounting:
  - Resolves `ui_dir` via `project_paths()` and mounts it at `/ui` if present.
- Pydantic request/response models:
  - `PredictRequest` / `PredictResponse` for single prediction.
  - `CorrectionRequest` for user feedback.
- Key endpoints:
  - `GET /` – returns a simple HTML link to `/ui/index.html`.
  - `POST /predict` – single prediction using `predict_with_confidence`; returns flattened explanation (neighbors, keyword matches, rationale).
  - `POST /predict_batch` – accepts a CSV upload (must contain a `description` column), runs `batch_inference`, and returns a list of `PredictResponse` objects.
  - `GET /taxonomy` – returns the current `Taxonomy` (JSON from `config/taxonomy.json`).
  - `POST /upload_taxonomy` – replaces `config/taxonomy.json` with an uploaded JSON file.
  - `POST /corrections` – records a correction via `feedback.record_correction`.
  - `GET /config` – returns `Settings` as JSON.
  - `GET /evaluation/confusion-matrix` – serves `confusion_matrix.png` from the evaluation directory (404 if not yet computed).

The API is designed as a thin layer over `inference`, `config_loader`, and `feedback`, with minimal business logic beyond validation and wiring.

### Feedback & Corrections (`mvp/src/feedback.py`)

- `CorrectionRecord` dataclass captures timestamped corrections with arbitrary metadata.
- Corrections are stored in `mvp/data/corrections.json`:
  - `_corrections_path()` derives the location via `project_paths()`.
  - `_load_corrections()` and `_save_corrections()` handle JSON I/O and directory creation.
- `record_correction`:
  - Adds a new record with an ISO8601 UTC timestamp.
  - Appends to the existing list and writes back to disk.

This module provides a simple feedback loop mechanism that can be consumed later for retraining, manual review, or analytics.

### Evaluation Pipeline (`mvp/evaluation/evaluate.py`)

- `load_split(name)` reads a named split (e.g., `"test"`) from `mvp/data/generated/{name}.csv`.
- `evaluate_split(split, out_dir)`:
  - Trains a fresh classifier via `train_classifier()`.
  - Normalizes descriptions and generates embeddings.
  - Predicts labels and computes a confusion matrix and classification report.
  - Saves:
    - `confusion_matrix.png` using `ConfusionMatrixDisplay`.
    - `per_class_metrics.json` with per-class metrics.
    - `macro_f1_score.json` with overall macro F1.
- `main()` orchestrates evaluation for the `test` split and populates `performance_notes.md` with a short summary (including macro F1).

This pipeline is separate from the online API but shares the same preprocessing and embedding stack.

### UI (`mvp/ui`)

- Static HTML/JS/CSS assets served at `/ui` by the FastAPI app.
- Interacts with the API endpoints for:
  - Single and batch predictions.
  - Viewing/updating taxonomy.
  - Sending correction feedback.

The UI is intentionally minimal and can be iterated on independently of the backend as long as the API contracts remain stable.
