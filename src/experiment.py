"""Shared contracts for the held-out clustering experiment pipeline."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from importlib.metadata import version

import numpy as np
import pandas as pd
import yaml

PREPROCESSING = {"version": 1, "text": "fillna-empty/str/strip/reject-empty", "normalize": "l2", "dtype": "float32"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_params(path="params.yaml"):
    params = yaml.safe_load(Path(path).read_text())
    data = params["data"]
    if data["train"] != "data/train_eval/train_20k.csv" or data["eval"] != "data/train_eval/eval_4k.csv":
        raise ValueError("Only the dedicated train_20k.csv and eval_4k.csv are supported.")
    seeds = params["exploration"]["seeds"]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Exactly five distinct seeds are required.")
    if params["selection"]["final_seed"] not in seeds:
        raise ValueError("The predetermined final seed must belong to the exploration.")
    if (set(params["exploration"]["methods"]) != {"kmeans", "spherical_kmeans"}
            or len(params["exploration"]["methods"]) != 2 or len(params["encoders"]) != 2):
        raise ValueError("Expected two encoders and both supported clustering methods.")
    for alias, encoder in params["encoders"].items():
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", alias):
            raise ValueError("Encoder aliases must be safe path components.")
        if not re.fullmatch(r"[0-9a-f]{40}", encoder.get("revision") or ""):
            raise ValueError(f"Pin {alias} to an immutable Hugging Face commit: run run_experiments.py --pin-revisions.")
    if params["evaluation"]["sample_size"] < 0 or params["evaluation"]["working_memory_mb"] <= 0:
        raise ValueError("Invalid silhouette sampling or memory budget.")
    return params


def reviews(params, split):
    frame = pd.read_csv(params["data"][split], dtype={params["data"]["id_column"]: str})
    text_column, id_column = params["data"]["text_column"], params["data"]["id_column"]
    texts = frame[text_column].fillna("").astype(str).str.strip()
    if len(frame) != params["data"][f"{split}_rows"] or texts.eq("").any():
        raise ValueError(f"Unexpected row count or empty reviews in {split}.")
    if frame[id_column].isna().any() or frame[id_column].duplicated().any():
        raise ValueError(f"Missing or duplicate observation IDs in {split}.")
    return texts.tolist(), frame[id_column].tolist()


def validate_splits(params):
    _, train_ids = reviews(params, "train")
    _, eval_ids = reviews(params, "eval")
    if set(train_ids) & set(eval_ids):
        raise ValueError("Train/evaluation observation IDs overlap.")


def embedding_spec(params, alias, split):
    return {"encoder": params["encoders"][alias], "preprocessing": PREPROCESSING,
            "input_sha256": file_hash(params["data"][split]), "split": split,
            "text_column": params["data"]["text_column"], "id_column": params["data"]["id_column"]}


def cache_dir(alias):
    return Path("models/embeddings") / alias


def load_embeddings(params, alias, split):
    directory = cache_dir(alias)
    info = read_json(directory / f"{split}.json")
    expected = embedding_spec(params, alias, split)
    if info["spec"] != expected or info["cache_id"] != digest(expected):
        raise ValueError("Embedding cache does not match dataset/encoder/preprocessing.")
    path = directory / f"{split}.npy"
    if file_hash(path) != info["sha256"]:
        raise ValueError("Embedding cache checksum mismatch.")
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.dtype != np.float32 or list(matrix.shape) != info["shape"]:
        raise ValueError("Embedding cache has invalid shape/dtype.")
    for start in range(0, len(matrix), 512):
        block = matrix[start:start + 512]
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=1e-5):
            raise ValueError("Embedding cache must contain finite L2-normalized rows.")
    _, ids = reviews(params, split)
    if info["observation_ids"] != ids or len(matrix) != len(ids):
        raise ValueError("Embedding observation order mismatch.")
    return matrix, info


def configuration(params):
    active = params["experiment"]
    alias, method, seed = active["encoder"], active["method"], active["seed"]
    if alias not in params["encoders"] or method not in params["exploration"]["methods"] or seed not in params["exploration"]["seeds"]:
        raise ValueError("Active experiment is outside the declared matrix.")
    return {"encoder_alias": alias, "encoder": params["encoders"][alias],
            "method": method, "hyperparameters": {**params["clustering"], "random_state": seed},
            "preprocessing": PREPROCESSING, "train_cache_id": digest(embedding_spec(params, alias, "train"))}


def sweep_id(params):
    return digest({"encoders": params["encoders"], "clustering": params["clustering"],
                   "exploration": params["exploration"], "evaluation": params["evaluation"],
                   "selection": params["selection"], "preprocessing": PREPROCESSING,
                   "data": params["data"], "train_hash": file_hash(params["data"]["train"]),
                   "eval_hash": file_hash(params["data"]["eval"]),
                   "sources": {name: file_hash(Path(__file__).with_name(name)) for name in
                               ("embed.py", "experiment.py", "train.py", "predict.py", "evaluate.py", "tracking.py")},
                   "environment_lock": file_hash(Path(__file__).resolve().parents[1] / "uv.lock")})[:20]


def run_dir(params):
    active = params["experiment"]
    return Path("models/runs") / active["encoder"] / active["method"] / str(active["seed"])


def prediction_dir(params):
    return Path("models/predictions") / run_dir(params).relative_to("models/runs")


def evaluation_dir(params):
    return Path("models/evaluations") / run_dir(params).relative_to("models/runs")


def provenance():
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"git_revision": git("rev-parse", "HEAD"),
            "dvc_experiment_name": os.environ.get("SUPPLY_CHAIN_DVC_NAME") or os.environ.get("DVC_EXP_NAME"),
            "versions": {p: version(p) for p in ("numpy", "scikit-learn", "sentence-transformers", "mlflow", "dvc")}}


def tracking(params):
    """Use an explicit stable store across DVC temporary workspaces."""
    from tracking import configure_tracking, load_tracking_environment
    env_file = params["tracking"].get("env_file")
    load_tracking_environment(env_file)
    uri = (os.environ.get("SUPPLY_CHAIN_TRACKING_URI") or params["tracking"]["uri"]
           or os.environ.get("MLFLOW_TRACKING_URI"))
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        raise ValueError("Set SUPPLY_CHAIN_TRACKING_URI to an absolute sqlite URI before running DVC experiments.")
    if not uri:
        raise ValueError("Set MLFLOW_TRACKING_URI in .env.dagshub.local or export a tracking URI.")
    return configure_tracking(uri, env_file=env_file)


def start_run(params, kind, logical_id, extra_tags=None):
    import mlflow
    uri = tracking(params)
    name = params["tracking"][kind + "_experiment"]
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is None and uri.startswith("sqlite:////"):
        # Never place MLflow artifacts inside a temporary DVC workspace.
        database = Path(uri[len("sqlite:///"):])
        artifact_root = database.parent / "mlartifacts" / name
        client.create_experiment(name, artifact_location=artifact_root.as_uri())
        experiment = client.get_experiment_by_name(name)
    if experiment is None:
        experiment = mlflow.set_experiment(name)
    return mlflow.start_run(experiment_id=experiment.experiment_id, run_name=logical_id,
                            tags={"logical_id": logical_id, "sweep_id": sweep_id(params),
                                  "pipeline_role": kind, **(extra_tags or {})})


def checked_manifest(params):
    manifest = read_json(run_dir(params) / "manifest.json")
    config = configuration(params)
    if manifest["configuration"] != config or manifest["logical_id"] != digest(config):
        raise ValueError("Training manifest does not match the active configuration.")
    if manifest["model_sha256"] != file_hash(manifest["model_path"]):
        raise ValueError("Model checksum mismatch.")
    return manifest
