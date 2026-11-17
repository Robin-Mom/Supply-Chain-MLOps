"""Sealed final evaluation: audit first, execute only after checkpoint approval."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd

from evaluate import held_out_silhouette
from experiment import digest, file_hash, read_json, write_json
from monitor_drift import dataset, require, selection_identity, validate_labels


COHORTS = {"training": "data/train_eval/train_20k.csv", "selection": "data/train_eval/eval_4k.csv",
           "reference": "data/processed/avis_4k_reference.csv", "production": "data/processed/avis_8k_production.csv"}
PACKAGES = ("numpy", "pandas", "scikit-learn", "sentence-transformers", "transformers", "torch", "mlflow", "matplotlib")
SOURCES = ("src/evaluate_final.py", "src/predict.py", "src/evaluate.py", "src/train.py",
           "src/api/main.py", "src/experiment.py", "src/monitor_drift.py", "uv.lock")


def matrix_hash(matrix):
    return hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()


def audit(args):
    identity = selection_identity(args.selection)
    require(identity["configuration"]["method"] == "spherical_kmeans", "requires frozen spherical model")
    rows, info = dataset(args.input, args.text_column, args.id_column)
    require(len(rows) == args.expected_rows, "unexpected final dataset row count")
    ids = {r["source_row_id"] for r in rows}
    text_hashes = [digest(r["text"].strip()) for r in rows]
    cohorts = {}
    for name, path in COHORTS.items():
        other, metadata = dataset(Path(path), args.text_column, args.id_column)
        overlap = len(ids & {r["source_row_id"] for r in other})
        require(overlap == 0, "final IDs overlap a development/monitoring cohort")
        other_texts = {digest(r["text"].strip()) for r in other}
        cohorts[name] = {"path": path, **metadata, "overlapping_ids": overlap,
                         "final_rows_with_matching_stripped_text": sum(h in other_texts for h in text_hashes)}
    prior = read_json(args.selection_record)
    require(all(prior[k] == identity[k] for k in ("logical_id", "configuration", "model_sha256")),
            "historical selection differs from frozen model")
    previous = None
    if args.previous_result:
        previous = {"path": str(args.previous_result.resolve()), "sha256": file_hash(args.previous_result)}
    protocol = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "input_path": str(args.input.resolve()), "dataset": info,
                "selection_path": str(args.selection.resolve()), "selection_sha256": file_hash(args.selection),
                "model": identity, "expected_rows": args.expected_rows,
                "implementation_sha256": {p: file_hash(p) for p in SOURCES},
                "versions": {name: importlib.metadata.version(name) for name in PACKAGES},
                "cohorts": cohorts, "selection_record_sha256": file_hash(args.selection_record),
                "selection_record_path": str(args.selection_record.resolve()),
                "silhouette": {"sample_size": 0, "sample_seed": 42, "working_memory_mb": 32},
                "geometry": "cosine on L2-normalized embeddings; original frozen unit centroids",
                "uncertainty": {"status": "not_estimated", "reason": "No justified dependence-aware resampling protocol; no refitting."},
                "previous_result": previous,
                "historical_context": {"selection_rule": prior.get("selection_rule"), "ranking": prior.get("ranking")},
                "independence": {"status": "current_ID_disjointness_verified",
                    "limitation": "Current IDs and pipeline inputs exclude final rows. Unrecorded historical use and shared text/author dependence cannot be ruled out; no complete immutable usage ledger exists. Drift thresholds were fixed before this audit."}}
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "seal.json", {"protocol_sha256": file_hash(args.output / "protocol.json")})
    return {"status": "awaiting_checkpoint_7A_1", "rows": len(rows), "protocol": str(args.output / "protocol.json"),
            "protocol_sha256": file_hash(args.output / "protocol.json"), "independence": protocol["independence"]}


def validate_geometry(X, labels, centers):
    X, labels, centers = np.asarray(X), np.asarray(labels), np.asarray(centers)
    require(X.ndim == 2 and len(X) > 0 and X.shape[1] > 0, "invalid embeddings")
    require(centers.ndim == 2 and centers.shape[1] == X.shape[1] and len(centers) >= 2, "invalid frozen centroids")
    require(labels.shape == (len(X),) and np.issubdtype(labels.dtype, np.integer), "invalid label shape/type")
    require(np.isfinite(X).all() and np.isfinite(centers).all(), "nonfinite geometry")
    require(np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-5) and
            np.allclose(np.linalg.norm(centers, axis=1), 1, atol=1e-5), "geometry must be L2 normalized")
    require(((labels >= 0) & (labels < len(centers))).all(), "cluster ID outside frozen range")
    similarities = X @ centers.T
    require(np.array_equal(labels, similarities.argmax(axis=1)), "labels differ from frozen centroid assignments")
    return X, labels, centers, np.clip(similarities, -1., 1.)


def describe(values):
    values = np.asarray(values)
    if not len(values):
        return {"status": "unavailable", "reason": "empty cluster", "count": 0}
    return {"status": "available", "count": len(values), "mean": float(np.mean(values)),
            "std_population": float(np.std(values)),
            **{name: float(np.quantile(values, q)) for name, q in
               [("min", 0), ("p05", .05), ("median", .5), ("p95", .95), ("max", 1)]}}


def geometry_metrics(X, labels, centers, silhouette_config):
    X, labels, centers, similarities = validate_geometry(X, labels, centers)
    assigned = similarities[np.arange(len(X)), labels]
    other = similarities.copy()
    other[np.arange(len(X)), labels] = -np.inf
    margins = assigned - other.max(axis=1)
    distances = 1 - assigned
    counts = np.bincount(labels, minlength=len(centers))
    proportions = counts / len(X)
    separation = np.clip(1 - centers @ centers.T, 0., 2.)
    pairwise = separation[np.triu_indices(len(centers), 1)]
    silhouette = held_out_silhouette(X, labels, silhouette_config)
    result = {"silhouette": silhouette, "centroid_origin": "original training centroids; never recomputed",
              "represented_clusters": int(np.count_nonzero(counts)),
              "empty_cluster_ids": np.flatnonzero(counts == 0).tolist(),
              "largest_cluster_proportion": float(proportions.max()),
              "largest_to_smallest_nonempty_count_ratio": float(counts.max() / counts[counts > 0].min()),
              "assigned_centroid_cosine_similarity": describe(assigned),
              "within_dispersion_one_minus_cosine": describe(distances),
              "assignment_margin_cosine_similarity": describe(margins),
              "original_centroid_pair_cosine_distance": describe(pairwise),
              "original_centroid_cosine_distance_matrix": separation.tolist(),
              "clusters": [{"cluster_id": k, "count": int(counts[k]), "proportion": float(proportions[k]),
                            "assigned_similarity": describe(assigned[labels == k]),
                            "dispersion": describe(distances[labels == k]),
                            "margin": describe(margins[labels == k])} for k in range(len(centers))]}
    return result, assigned, distances, margins


def comparison(previous, protocol, final_score):
    if previous is None:
        return {"status": "unavailable", "reason": "No verified individual prior result supplied; five-seed means are context only."}
    require(file_hash(previous["path"]) == previous["sha256"], "previous result changed")
    old = read_json(previous["path"])
    compatible = (old.get("logical_id") == protocol["model"]["logical_id"] and
                  old.get("configuration") == protocol["model"]["configuration"] and
                  old.get("evaluation_config") == protocol["silhouette"] and
                  old.get("train_manifest", {}).get("model_sha256") == protocol["model"]["model_sha256"] and
                  old.get("eval_input_sha256") == protocol["cohorts"]["selection"]["sha256"] and
                  old.get("metrics", {}).get("silhouette_n_samples") == protocol["cohorts"]["selection"]["rows"])
    if not compatible:
        return {"status": "incompatible", "reason": "Model/checksum, geometry, source fingerprint or sampling protocol not verified identical."}
    value = old["metrics"].get("silhouette_cosine")
    return {"status": "available" if value is not None and final_score is not None else "unavailable",
            "previous_silhouette_cosine": value, "final_silhouette_cosine": final_score,
            "difference_final_minus_previous": float(final_score - value) if value is not None and final_score is not None else None,
            "interpretation": "Different cohorts; descriptive geometric difference, not proof of performance change."}


def figures(directory, labels, assigned, margins, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir()
    groups = [k for k in range(len(metrics["clusters"])) if np.any(labels == k)]
    specs = [("cluster_population", "Cluster population", "Reviews"),
             ("centroid_similarity", "Similarity to frozen assigned centroid", "Cosine similarity"),
             ("assignment_margin", "Margin to next-best frozen centroid", "Cosine similarity margin")]
    for name, title, ylabel in specs:
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        if name == "cluster_population":
            ax.bar([c["cluster_id"] for c in metrics["clusters"]], [c["count"] for c in metrics["clusters"]])
            ax.set_xticks(range(len(metrics["clusters"])));ax.set_xlabel("Frozen cluster ID")
        elif name == "centroid_similarity":
            ax.boxplot([assigned[labels == k] for k in groups], tick_labels=[str(k) for k in groups], showfliers=False)
            ax.set_xlabel("Frozen cluster ID (empty clusters omitted)")
        else:
            ax.hist(margins, bins=np.linspace(0, 2, 41));ax.set_xlabel("Cosine similarity margin");ylabel="Reviews"
        ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis="y", alpha=.2)
        fig.savefig(directory / f"{name}.png", dpi=300)
        fig.savefig(directory / f"{name}.svg")
        plt.close(fig)


def execute(args, loader=None):
    start = perf_counter()
    require(args.execute and args.protocol is not None, "explicit --execute and sealed --protocol required")
    seal = read_json(args.protocol.with_name("seal.json"))
    require(file_hash(args.protocol) == seal["protocol_sha256"], "protocol seal mismatch")
    protocol = read_json(args.protocol)
    require(protocol["schema_version"] == 1, "unsupported protocol")
    require(all(file_hash(p) == h for p,h in protocol["implementation_sha256"].items()), "evaluation implementation changed")
    require(all(importlib.metadata.version(p) == v for p,v in protocol["versions"].items()), "evaluation software versions changed")
    require(args.batch_size > 0, "invalid batch size")
    require(file_hash(protocol["selection_path"]) == protocol["selection_sha256"], "selection changed")
    identity = selection_identity(protocol["selection_path"])
    require(identity == protocol["model"], "model identity differs from protocol")
    rows, info = dataset(Path(protocol["input_path"]), protocol["dataset"]["text_column"], protocol["dataset"]["id_column"])
    require(info == protocol["dataset"] and len(rows) == protocol["expected_rows"], "final dataset changed")
    require(file_hash(protocol["selection_record_path"]) == protocol["selection_record_sha256"], "selection history changed")
    for cohort in protocol["cohorts"].values():
        require(file_hash(cohort["path"]) == cohort["sha256"], "audit cohort changed")
    # Reserve before loading/inference: an interrupted attempt cannot silently rerun.
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    with args.protocol.with_name("execution_claim.json").open("x", encoding="utf-8") as stream:
        json.dump({"output": str(args.output.resolve()), "protocol_sha256": seal["protocol_sha256"]}, stream)
    write_json(args.output / "execution_state.json", {"status": "started", "protocol_sha256": seal["protocol_sha256"]})
    if loader is None:
        from api.main import load_selected_model
        loader = load_selected_model
    with tempfile.TemporaryDirectory(prefix="final-evaluation-") as destination:
        predictor = loader(Path(protocol["selection_path"]), destination)
        manifest = predictor.manifest
        require(all(manifest[k] == identity[k] for k in ("logical_id", "configuration", "model_sha256")), "loaded model differs")
        require(file_hash(manifest["model_path"]) == identity["model_sha256"], "loaded binary checksum differs")
        model = predictor.bundle["model"]
        from train import SphericalKMeans
        require(isinstance(model, SphericalKMeans), "unexpected clustering estimator")
        require(all(model.get_params().get(k) == v for k,v in identity["configuration"]["hyperparameters"].items()), "frozen hyperparameters differ")
        centers = model.cluster_centers_.copy()
        center_hash = matrix_hash(centers)
        require(centers.ndim == 2 and len(centers) == identity["n_clusters"] and
                centers.shape[1] == predictor.bundle.get("n_features", centers.shape[1]) and
                np.isfinite(centers).all() and
                np.allclose(np.linalg.norm(centers, axis=1), 1, atol=1e-5),
                "invalid original centroid geometry")
        labels, X = predictor.predict_texts([r["text"] for r in rows], batch_size=args.batch_size,
                                           device="cpu", return_embeddings=True)
        require(matrix_hash(model.cluster_centers_) == center_hash, "centroids mutated during inference")
        require(file_hash(manifest["model_path"]) == identity["model_sha256"], "binary changed during inference")
    records = [{"source_row_id": r["source_row_id"], "source_row_index": r["source_row_index"],
                "model_id": identity["logical_id"], "success": True, "cluster_id": int(label)} for r,label in zip(rows,labels)]
    require(len(labels) == len(rows) and np.issubdtype(np.asarray(labels).dtype, np.integer), "prediction coverage/type mismatch")
    _, coverage = validate_labels(records, rows, identity, False)
    metrics, assigned, distances, margins = geometry_metrics(X, labels, centers, protocol["silhouette"])
    metrics["comparison"] = comparison(protocol["previous_result"], protocol, metrics["silhouette"]["silhouette_cosine"])
    metrics["uncertainty"] = protocol["uncertainty"]
    require(file_hash(protocol["input_path"]) == info["sha256"], "input changed during inference")
    table = pd.DataFrame(records).assign(assigned_centroid_cosine_similarity=assigned,
                                         assigned_centroid_cosine_distance=distances, assignment_margin=margins)
    table.to_csv(args.output / "predictions.csv", index=False)
    np.save(args.output / "embeddings.npy", X, allow_pickle=False)
    np.save(args.output / "original_centroids.npy", centers, allow_pickle=False)
    figures(args.output / "figures", labels, assigned, margins, metrics)
    write_json(args.output / "metrics.json", metrics)
    write_json(args.output / "protocol.json", protocol)
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    metadata = {"schema_version": 1, "model": identity, "dataset": info, "coverage": coverage,
                "protocol_sha256": seal["protocol_sha256"], "original_centroid_bytes_sha256": center_hash,
                "versions": versions, "runtime_seconds": perf_counter() - start,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "artifacts": {str(p.relative_to(args.output)): file_hash(p) for p in sorted(args.output.rglob("*"))
                              if p.is_file() and p.name != "execution_state.json"}}
    write_json(args.output / "metadata.json", metadata)
    write_json(args.output / "execution_state.json", {"status": "complete"})
    return {"status": "complete", "rows": len(rows), "output": str(args.output)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    p.add_argument("--input", type=Path, default=Path("data/processed/avis_3k_test.csv"))
    p.add_argument("--expected-rows", type=int, default=3000)
    p.add_argument("--selection", type=Path, default=Path("config/selection.docker.json"))
    p.add_argument("--selection-record", type=Path, default=Path("reports/experiment_selection/selection.json"))
    p.add_argument("--text-column", default="commentaire");p.add_argument("--id-column", default="")
    p.add_argument("--previous-result", type=Path)
    p.add_argument("--protocol", type=Path)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        print(json.dumps(audit(args) if args.audit_only else execute(args), allow_nan=False, indent=2))
    except Exception:
        # Remote exceptions may contain secrets. Direct functions retain fixed diagnostics for tests.
        print(json.dumps({"status": "failed", "reason": "Final evaluation rejected; inspect input/protocol/model consistency. No completed result claimed."}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
