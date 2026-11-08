"""Select a clustering model using a declared geometric metric.

Selection is a local experiment decision, not deployment or promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from train import METHODS, write_json


def rank_models(metrics, metric="silhouette_cosine", direction="maximize"):
    """Rank finite scores, breaking ties by method name for repeatability."""
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be maximize or minimize.")
    candidates = []
    excluded = []
    for method, scores in sorted(metrics.items()):
        if method not in METHODS:
            raise ValueError(f"Unsupported method: {method}")
        score = scores.get(metric)
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score):
            excluded.append(method)
            continue
        candidates.append({"method": method, "score": float(score)})
    if not candidates:
        raise ValueError(f"No finite scores available for {metric}.")
    candidates.sort(key=lambda item: (
        -item["score"] if direction == "maximize" else item["score"], item["method"]
    ))
    return candidates, excluded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=("silhouette_cosine", "davies_bouldin"), default="silhouette_cosine")
    parser.add_argument("--direction", choices=("maximize", "minimize"), default="maximize")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output file already exists; choose a new path.")
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    try:
        ranking, excluded = rank_models(metrics, args.metric, args.direction)
    except ValueError as error:
        parser.error(str(error))
    metadata_path = args.artifact_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if any(item["method"] not in metadata["methods"] for item in ranking):
        parser.error("Metrics contain methods absent from training metadata.")
    selected = ranking[0]["method"]
    model_path = args.artifact_dir / selected / "model.joblib"
    result = {
        "method": selected, "metric": args.metric, "direction": args.direction,
        "score": ranking[0]["score"], "ranking": ranking, "excluded_methods": excluded,
        "model_path": str(model_path), "encoder_path": str(args.artifact_dir / "encoder"),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "metrics_sha256": hashlib.sha256(args.metrics.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "input_sha256": metadata["input_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(f"Selected {selected}: {args.metric}={result['score']}")


if __name__ == "__main__":
    main()
