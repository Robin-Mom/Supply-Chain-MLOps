"""Train KMeans and Spherical KMeans on normalized review embeddings.

The pretrained text encoder is used as-is, without fine-tuning.
Run `python train.py --help` for command-line options.
Keep train.py, evaluate.py, predict.py and tracking.py in the same directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted

from tracking import configure_tracking

METHODS = ("kmeans", "spherical_kmeans")
DEFAULT_ENCODER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def validate_embeddings(X):
    """Validate a dense embedding matrix and normalize each row."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not len(X) or not X.shape[1]:
        raise ValueError("Expected a non-empty two-dimensional embedding matrix.")
    if not np.isfinite(X).all() or np.any(np.linalg.norm(X, axis=1) == 0):
        raise ValueError("Embeddings must be finite and have nonzero row norms.")
    return normalize(X)


def load_reviews(path, text_column="commentaire"):
    """Read a CSV without silently dropping or reordering observations."""
    df = pd.read_csv(path)
    if text_column not in df:
        raise ValueError(f"Missing text column: {text_column}")
    texts = df[text_column].fillna("").astype(str).str.strip()
    invalid = texts.eq("")
    if df.empty or invalid.any():
        raise ValueError(f"Empty dataset or blank reviews at row positions {np.flatnonzero(invalid)[:10].tolist()}")
    return df, texts.to_numpy()


def write_json(path, value):
    """Write standards-compliant JSON, rejecting NaN and infinity."""
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


class SphericalKMeans(ClusterMixin, BaseEstimator):
    """Cluster normalized embeddings using cosine similarity.

    Adapted from notebook cell 5; centroids are normalized after each update.
    """
    def __init__(self, n_clusters=10, max_iter=100,
                 n_init=5, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X, y=None):
        X = validate_embeddings(X)
        if not 1 <= self.n_clusters <= len(X):
            raise ValueError("n_clusters must be between 1 and the sample count.")
        if self.n_init < 1 or self.max_iter < 1:
            raise ValueError("n_init and max_iter must be positive.")
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        best_inertia = np.inf

        for _ in range(self.n_init):
            seed = int(rng.integers(0, 2**31 - 1))
            centers, _ = kmeans_plusplus(
                X, n_clusters=self.n_clusters,
                random_state=seed
            )
            centers = normalize(centers)

            for iteration in range(self.max_iter):
                labels = np.argmax(X @ centers.T, axis=1)

                new_centers = np.zeros_like(centers)
                np.add.at(new_centers, labels, X)

                # Reinitialize empty clusters from existing observations.
                empty = np.flatnonzero(
                    np.linalg.norm(new_centers, axis=1) == 0
                )
                for k in empty:
                    idx = rng.integers(len(X))
                    new_centers[k] = X[idx]

                new_centers = normalize(new_centers)

                if np.allclose(centers, new_centers, atol=1e-6):
                    centers = new_centers
                    break

                centers = new_centers

            labels = np.argmax(X @ centers.T, axis=1)
            similarities = np.sum(X * centers[labels], axis=1)
            inertia = np.sum(1 - similarities)

            if inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centers.copy()
                self.labels_ = labels.copy()
                self.inertia_ = float(inertia)
                self.n_iter_ = iteration + 1

        return self

    def predict(self, X):
        check_is_fitted(self, "cluster_centers_")
        X = validate_embeddings(X)
        return np.argmax(X @ self.cluster_centers_.T, axis=1)

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


def default_config(n_clusters=10, seed=42):
    """Use the notebook's training defaults consistently for stability runs."""
    return {
        "kmeans": {"n_clusters": n_clusters, "n_init": 10, "random_state": seed},
        "spherical_kmeans": {"n_clusters": n_clusters, "n_init": 5, "max_iter": 100, "random_state": seed},
    }


def fit_clustering(X, method, config, seed=None):
    """Fit one strategy and return estimator, None, labels and embeddings."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    params = dict(config[method])
    if seed is not None:
        params["random_state"] = seed
    cls = KMeans if method == "kmeans" else SphericalKMeans
    model = cls(**params)
    labels = model.fit_predict(X)
    # Preserve the fit-result interface used by stability evaluation.
    return model, None, labels, X


def main(argv=None):
    """Encode reviews once, train selected methods and save reusable bundles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-column", default="commentaire")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--revision", help="Optional Hugging Face commit hash for reproducibility.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, help="JSON overrides grouped by supported method.")
    tracking = parser.add_mutually_exclusive_group()
    tracking.add_argument("--tracking-uri", help="MLflow server URI; defaults to MLFLOW_TRACKING_URI.")
    tracking.add_argument("--dagshub-repo", help="Track on DagsHub: OWNER/REPOSITORY.")
    parser.add_argument("--experiment", default="Supply-chain_model-selection")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new directory.")
    if args.batch_size < 1:
        parser.error("batch-size must be positive.")
    try:
        tracking_uri = configure_tracking(args.tracking_uri, args.dagshub_repo)
    except ValueError as error:
        parser.error(str(error))
    if tracking_uri:
        import mlflow
        mlflow.set_experiment(args.experiment)
    _, texts = load_reviews(args.input, args.text_column)
    config = default_config(args.n_clusters, args.seed)
    if args.config:
        for key, values in json.loads(args.config.read_text()).items():
            if key not in config or not isinstance(values, dict):
                parser.error(f"Invalid configuration section: {key}")
            config[key].update(values)
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(args.encoder, revision=args.revision, device=args.device,
                                  trust_remote_code=False, model_kwargs={"use_safetensors": True})
    X = validate_embeddings(encoder.encode(texts.tolist(), batch_size=args.batch_size,
                                          normalize_embeddings=True, show_progress_bar=True))
    args.output_dir.mkdir(parents=True)
    encoder.save_pretrained(str(args.output_dir / "encoder"))
    np.save(args.output_dir / "embeddings.npy", X)
    pd.DataFrame({"text": texts}).to_csv(args.output_dir / "texts.csv", index=False)
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "sentence-transformers", "joblib", "mlflow", "bertopic", "spacy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    metadata = {"encoder": args.encoder, "revision": args.revision, "text_column": args.text_column,
                "normalize_embeddings": True, "methods": list(dict.fromkeys(args.methods)),
                "config": config, "n_samples": len(X), "n_features": X.shape[1],
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(), "versions": packages}
    write_json(args.output_dir / "metadata.json", metadata)
    for method in metadata["methods"]:
        print(f"Training {method} on {X.shape}...", flush=True)
        start = perf_counter()
        model, _, labels, _ = fit_clustering(X, method, config)
        directory = args.output_dir / method
        directory.mkdir()
        joblib.dump({"method": method, "model": model,
                     "n_features": X.shape[1]}, directory / "model.joblib")
        np.save(directory / "labels.npy", labels)
        training_metrics = {"fit_seconds": perf_counter() - start}
        if hasattr(model, "inertia_"):
            training_metrics["inertia"] = float(model.inertia_)
        write_json(directory / "training_metrics.json", training_metrics)
        if tracking_uri:
            import mlflow
            with mlflow.start_run(run_name=f"mpnet-{method}") as run:
                write_json(directory / "mlflow_run.json", {"run_id": run.info.run_id, "tracking_uri": tracking_uri})
                mlflow.log_params({"method": method, "encoder": args.encoder, "normalize_embeddings": True,
                                   **config[method]})
                mlflow.log_metrics(training_metrics)
                # Store portable artifacts; these are not MLflow flavor models.
                mlflow.log_artifacts(str(directory), artifact_path="clustering")
                mlflow.log_artifacts(str(args.output_dir / "encoder"), artifact_path="encoder")
                mlflow.log_artifact(str(args.output_dir / "metadata.json"))
                mlflow.log_artifact(__file__, artifact_path="source")
                mlflow.log_artifact(str(Path(__file__).with_name("tracking.py")), artifact_path="source")
        print(f"Saved {directory}", flush=True)


if __name__ == "__main__":
    # Use a stable module name so joblib can reload SphericalKMeans later.
    from train import main as entry_point
    entry_point()
