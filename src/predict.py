"""Assign new reviews to an existing KMeans or Spherical KMeans model.

No estimator is fitted in this module. scikit-learn HDBSCAN has no predict
method and is deliberately rejected rather than replaced with a heuristic.
Source: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from train import load_reviews, validate_embeddings


def load_bundle(artifact_dir, method):
    """Load a trusted local bundle; never load untrusted joblib files."""
    if method not in {"kmeans", "spherical_kmeans"}:
        raise ValueError("New-data prediction is supported only for kmeans and spherical_kmeans. "
                         "The notebook uses sklearn.cluster.HDBSCAN, which has no predict method.")
    bundle = joblib.load(Path(artifact_dir) / method / "model.joblib")
    if bundle["method"] != method:
        raise ValueError("The model bundle does not match the requested method.")
    return bundle


def predict_embeddings(bundle, embeddings):
    """Predict labels in the same normalized space used during training."""
    X = validate_embeddings(embeddings)
    if X.shape[1] != bundle["n_features"]:
        raise ValueError("Embedding dimensions do not match the trained model.")
    return np.asarray(bundle["model"].predict(X))


def predict_reviews(artifact_dir, method, texts, device=None, batch_size=32):
    """Reload the saved encoder and model, then return labels and embeddings."""
    bundle = load_bundle(artifact_dir, method)
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(str(Path(artifact_dir) / "encoder"), device=device,
                                  local_files_only=True, trust_remote_code=False,
                                  model_kwargs={"use_safetensors": True})
    X = validate_embeddings(encoder.encode(list(texts), batch_size=batch_size,
                                          normalize_embeddings=True, show_progress_bar=True))
    return predict_embeddings(bundle, X), X


def main(argv=None):
    """Write the input columns plus cluster_id, preserving input row order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("kmeans", "spherical_kmeans"), required=True)
    parser.add_argument("--text-column", default="commentaire")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output file already exists; choose a new path.")
    if args.batch_size < 1:
        parser.error("batch-size must be positive.")
    df, texts = load_reviews(args.input, args.text_column)
    if "cluster_id" in df:
        parser.error("Input already contains cluster_id; rename that column first.")
    labels, _ = predict_reviews(args.artifact_dir, args.method, texts, args.device, args.batch_size)
    df["cluster_id"] = labels
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} predictions to {args.output}")


if __name__ == "__main__":
    main()
