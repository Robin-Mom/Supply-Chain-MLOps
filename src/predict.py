"""Assign new reviews to an existing KMeans or Spherical KMeans model.

No estimator is fitted in this module.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from train import METHODS, load_reviews, validate_embeddings


def load_bundle(artifact_dir, method):
    """Load a trusted local bundle; never load untrusted joblib files."""
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}. Choose from {METHODS}.")
    bundle = joblib.load(Path(artifact_dir) / method / "model.joblib")
    if bundle["method"] != method:
        raise ValueError("The model bundle does not match the requested method.")
    return bundle


def predict_embeddings(bundle, embeddings):
    """Predict labels in the same normalized space used during training."""
    if bundle["method"] not in METHODS:
        raise ValueError(f"Unsupported method: {bundle['method']}")
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
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--params" in arguments:
        from experiment import load_params
        parser = argparse.ArgumentParser()
        parser.add_argument("--params", required=True)
        return predict_experiment(load_params(parser.parse_args(arguments).params))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
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


class ExperimentPredictor:
    """Reusable frozen inference service for trusted experiment bundles."""
    def __init__(self, manifest):
        from experiment import file_hash
        self.manifest = manifest
        if file_hash(manifest["model_path"]) != manifest["model_sha256"]:
            raise ValueError("Model checksum mismatch.")
        if manifest["transformations"]:
            raise ValueError("Unsupported transformations; no fitting is permitted during inference.")
        self.bundle = joblib.load(manifest["model_path"])
        config = manifest["configuration"]
        if (self.bundle["logical_id"] != manifest["logical_id"] or self.bundle["encoder"] != config["encoder"]
                or self.bundle["preprocessing"] != config["preprocessing"]
                or self.bundle["method"] != config["method"]):
            raise ValueError("Bundle and manifest configurations differ.")
        self.encoder = None

    def predict_cached(self, embeddings):
        return predict_embeddings(self.bundle, embeddings)

    @classmethod
    def from_mlflow(cls, manifest_uri, destination):
        """Materialize only the selected clustering bundle for future API use.

        Configure MLflow tracking first; the transformer remains in the HF cache.
        """
        import mlflow
        from experiment import read_json
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        local_manifest = mlflow.artifacts.download_artifacts(artifact_uri=manifest_uri, dst_path=str(destination))
        manifest = read_json(local_manifest)
        manifest["model_path"] = mlflow.artifacts.download_artifacts(
            artifact_uri=manifest["model_uri"], dst_path=str(destination))
        return cls(manifest)

    def predict_texts(self, texts, batch_size=32, device=None):
        """Load the pinned encoder lazily and reuse it across API requests."""
        from sentence_transformers import SentenceTransformer
        from experiment import PREPROCESSING
        if self.manifest["configuration"]["preprocessing"] != PREPROCESSING:
            raise ValueError("Unsupported preprocessing version.")
        cleaned = [str(text).strip() if text is not None else "" for text in texts]
        if not cleaned or any(not text for text in cleaned) or batch_size < 1:
            raise ValueError("Provide nonblank texts and a positive batch size.")
        if self.encoder is None:
            spec = self.manifest["configuration"]["encoder"]
            self.encoder = SentenceTransformer(spec["id"], revision=spec["revision"], device=device,
                                              trust_remote_code=False, model_kwargs={"use_safetensors": True})
            self.encoder.max_seq_length = spec["max_seq_length"]
        labels = []
        for start in range(0, len(cleaned), batch_size):
            block = self.encoder.encode(cleaned[start:start + batch_size], batch_size=batch_size,
                                        normalize_embeddings=True, convert_to_numpy=True)
            labels.append(self.predict_cached(block))
        return np.concatenate(labels)


def predict_experiment(params):
    import pandas as pd
    from experiment import (checked_manifest, digest, file_hash, load_embeddings, prediction_dir, write_json)
    manifest = checked_manifest(params)
    X, cache = load_embeddings(params, params["experiment"]["encoder"], "eval")
    if X.shape[1] != manifest["n_features"]:
        raise ValueError("Train/eval dimensions differ.")
    labels = ExperimentPredictor(manifest).predict_cached(X)
    directory = prediction_dir(params)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "assignments.csv"
    pd.DataFrame({"observation_id": cache["observation_ids"], "logical_id": manifest["logical_id"],
                  "cluster_id": labels}).to_csv(path, index=False)
    info = {"logical_id": manifest["logical_id"], "train_mlflow_run_id": manifest["mlflow_run_id"],
            "eval_cache_id": cache["cache_id"], "eval_embedding_sha256": cache["sha256"],
            "predictions_path": str(path), "predictions_sha256": file_hash(path),
            "observation_ids_sha256": digest(cache["observation_ids"])}
    write_json(directory / "manifest.json", info)
    return info


if __name__ == "__main__":
    main()
