"""Offline text-feature and frozen-model cluster monitoring; never fits a model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import permutation_test, wasserstein_distance

from simulate_production import atomic_json, fingerprint, load_rows, read_json, run_lock, utc_now


FEATURES = ("character_length", "word_count")


class MonitoringError(ValueError):
    """Fixed diagnostics containing no review content or remote credentials."""


def require(condition, message):
    if not condition:
        raise MonitoringError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def selection_identity(path):
    selection = read_json(path)
    config = selection["configuration"]
    # Use the training pipeline's canonicalization, including its JSON defaults.
    from experiment import digest, PREPROCESSING
    require(selection["logical_id"] == digest(config), "invalid selected model identity")
    require(config["preprocessing"] == PREPROCESSING, "unsupported preprocessing")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", config["encoder"]["revision"])), "unpinned encoder revision")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", selection["model_sha256"])), "invalid selected model checksum")
    require(selection["final_seed"] == config["hyperparameters"]["random_state"], "seed mismatch")
    clusters = config["hyperparameters"]["n_clusters"]
    require(type(clusters) is int and clusters > 1, "invalid cluster count")
    return {"logical_id": selection["logical_id"], "model_sha256": selection["model_sha256"],
            "configuration": config, "n_clusters": clusters}


def dataset(path, text_column, id_column):
    before = fingerprint(path)
    rows = load_rows(path, text_column, id_column)
    require(before == fingerprint(path), "dataset changed while reading")
    return rows, {"sha256": before, "rows": len(rows), "text_column": text_column,
                  "id_column": id_column}


def features(rows):
    # Exactly the frozen inference cleanup; no authors, ratings or raw text in reports.
    return pd.DataFrame({"character_length": [len(r["text"].strip()) for r in rows],
                         "word_count": [len(r["text"].strip().split()) for r in rows]})


def ks_distance(a, b):
    a, b = np.sort(a), np.sort(b)
    support = np.union1d(a, b)
    return float(np.max(np.abs(np.searchsorted(a, support, side="right") / len(a)
                               - np.searchsorted(b, support, side="right") / len(b))))


def permutation(a, b, statistic, resamples, seed):
    result = permutation_test((np.asarray(a), np.asarray(b)), statistic, vectorized=False,
                              alternative="greater", n_resamples=resamples, batch=32, rng=seed)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def descriptive(values):
    return {"count": len(values), "mean": float(np.mean(values)),
            "std_population": float(np.std(values)),
            **{name: float(np.quantile(values, q)) for name, q in
               [("min", 0), ("p25", .25), ("median", .5), ("p75", .75), ("p95", .95), ("max", 1)]}}


def input_analysis(reference, production, resamples, seed, threshold):
    result = {}
    for name in FEATURES:
        a, b = reference[name].to_numpy(), production[name].to_numpy()
        result[name] = {"test": "permutation_two_sample_KS", **permutation(a, b, ks_distance, resamples, seed),
                        "reference": descriptive(a), "production": descriptive(b),
                        "wasserstein_raw_units": float(wasserstein_distance(a, b)),
                        "effect_name": "KS_D", "effect_size": ks_distance(a, b),
                        "practical_threshold": threshold}
    return result


def cluster_analysis(a, b, n_clusters, resamples, seed, threshold):
    a, b = np.asarray(a), np.asarray(b)
    def counts(values):
        return np.bincount(values.astype(int), minlength=n_clusters)
    def pearson(x, y):
        observed = np.array([counts(x), counts(y)], dtype=float)
        observed = observed[:, observed.sum(axis=0) > 0]
        expected = observed.sum(axis=1)[:, None] * observed.sum(axis=0)[None, :] / observed.sum()
        return float(np.sum((observed - expected) ** 2 / expected))
    ca, cb = counts(a), counts(b)
    pa, pb = ca / len(a), cb / len(b)
    return {"status": "available", "test": "permutation_Pearson_homogeneity",
            **permutation(a, b, pearson, resamples, seed), "reference_rows": len(a),
            "production_rows": len(b), "effect_name": "total_variation",
            "effect_size": float(np.sum(np.abs(pb - pa)) / 2), "practical_threshold": threshold,
            "clusters": [{"cluster_id": i, "reference_count": int(ca[i]),
                          "production_count": int(cb[i]), "reference_proportion": float(pa[i]),
                          "production_proportion": float(pb[i]), "proportion_delta": float(pb[i] - pa[i])}
                         for i in range(n_clusters)]}


def holm(tests, alpha):
    ordered = sorted(tests, key=lambda t: t["p_value"])
    previous = 0.
    for i, test in enumerate(ordered):
        previous = min(1., max(previous, (len(ordered) - i) * test["p_value"]))
        test.update(adjusted_p_value=previous, statistically_significant=previous < alpha,
                    practically_important=test["effect_size"] >= test["practical_threshold"])


def validate_labels(records, rows, identity, allow_subset):
    lookup = {r["source_row_id"]: r for r in rows}
    seen, labels = set(), []
    for record in records:
        source_id = record["source_row_id"]
        require(source_id in lookup and source_id not in seen, "unknown or duplicate prediction ID")
        seen.add(source_id)
        require(type(record["source_row_index"]) is int and
                record["source_row_index"] == lookup[source_id]["source_row_index"], "row alignment mismatch")
        require(record["model_id"] == identity["logical_id"], "prediction model mismatch")
        require(type(record["success"]) is bool, "invalid prediction success flag")
        if record["success"]:
            label = record["cluster_id"]
            require(type(label) is int and 0 <= label < identity["n_clusters"], "invalid cluster ID")
            labels.append(label)
        else:
            require(record["cluster_id"] is None, "failed prediction has a cluster")
    require(labels, "empty successful prediction set")
    require(allow_subset or len(labels) == len(rows), "incomplete predictions; explicit --allow-subset required")
    return labels, {"dataset_rows": len(rows), "recorded_rows": len(seen), "successful_rows": len(labels),
                    "failed_rows": len(seen) - len(labels), "missing_rows": len(rows) - len(seen),
                    "subset": len(labels) != len(rows),
                    "successful_ids_sha256": canonical_hash(sorted(r["source_row_id"] for r in records if r["success"]))}


def reference_labels(path, rows, info, identity, allow_subset):
    before = fingerprint(path)
    artifact = read_json(path)
    require(artifact["schema_version"] == 1 and artifact["kind"] == "reference_predictions", "invalid reference schema")
    require(artifact["model"] == identity, "reference model/configuration/checksum mismatch")
    require(artifact["dataset"] == info, "reference dataset mismatch")
    require(artifact["records_sha256"] == canonical_hash(artifact["records"]), "reference records checksum mismatch")
    labels, coverage = validate_labels(artifact["records"], rows, identity, allow_subset)
    require(fingerprint(path) == before, "reference artifact changed while reading")
    coverage["artifact_sha256"] = before
    return labels, coverage


def production_labels(directory, rows, info, identity, allow_subset):
    """Read the simulator's checkpoint source of truth, never its summary counts."""
    manifest = read_json(directory / "run.json")
    config = manifest["configuration"]
    require(config["schema_version"] == 1, "unsupported simulator schema")
    require(config["input_sha256"] == info["sha256"] and config["text_column"] == info["text_column"]
            and config["id_column"] == info["id_column"], "production dataset mismatch")
    require(config["model_id"] == identity["logical_id"] and config["n_clusters"] == identity["n_clusters"],
            "production model mismatch")
    # Reconstruct the exact replay plan, including shuffled limited simulations.
    import random
    planned = list(rows)
    if config["shuffle_seed"] is not None:
        random.Random(config["shuffle_seed"]).shuffle(planned)
    if config["limit"] is not None:
        require(type(config["limit"]) is int and config["limit"] > 0, "invalid simulation limit")
        planned = planned[:config["limit"]]
    order = hashlib.sha256(json.dumps([r["source_row_id"] for r in planned]).encode()).hexdigest()
    require(manifest["selected_reviews"] == len(planned) and manifest["order_sha256"] == order, "simulation plan mismatch")
    size = config["batch_size"]
    require(type(size) is int and 1 <= size <= 64, "invalid batch size")
    batches = [planned[i:i + size] for i in range(0, len(planned), size)]
    records, sources = [], {"run.json": fingerprint(directory / "run.json")}
    for path in sorted((directory / "requests").glob("*.json")):
        checkpoint = read_json(path)
        if checkpoint["kind"] == "health" and path.stem.startswith("health-"):
            continue
        require(checkpoint["kind"] == "prediction" and path.stem.startswith("batch-"), "unknown checkpoint")
        index = int(path.stem.removeprefix("batch-"))
        require(0 <= index < len(batches) and path.stem == f"batch-{index:06d}", "invalid batch index")
        batch = batches[index]
        require(checkpoint["run_id"] == manifest["run_id"] and checkpoint["request_id"] == path.stem
                and checkpoint["source_row_ids"] == [r["source_row_id"] for r in batch], "checkpoint plan mismatch")
        require(checkpoint["state"] in {"pending", "complete"}, "invalid checkpoint state")
        if checkpoint["state"] == "pending":
            require(not checkpoint["records"], "pending checkpoint contains records")
        else:
            require(len(checkpoint["records"]) == len(batch), "incomplete batch")
            for record, row in zip(checkpoint["records"], batch):
                require(record["source_row_id"] == row["source_row_id"] and
                        record["run_id"] == manifest["run_id"] and record["request_id"] == path.stem,
                        "record checkpoint mismatch")
                require(not record["success"] or record["http_status"] == 200 and record["error_category"] is None,
                        "inconsistent successful response")
            records.extend(checkpoint["records"])
        sources[str(path.relative_to(directory))] = fingerprint(path)
    labels, coverage = validate_labels(records, rows, identity, allow_subset)
    coverage.update(run_id=manifest["run_id"], selected_reviews=len(planned), checkpoint_sha256=sources,
                    identity_limit="Simulator records logical model ID, not observed artifact checksum; relies on verified API loader.")
    return labels, coverage


def html_report(reference, production, path, categorical=False):
    from evidently import Report, Dataset, DataDefinition
    from evidently.metrics import ValueDrift
    columns = list(reference.columns)
    definition = DataDefinition(numerical_columns=[] if categorical else columns,
                                categorical_columns=columns if categorical else [])
    # HTML shows distances/distributions; inferential conclusions live in results.json.
    method = "jensenshannon" if categorical else "wasserstein"
    snapshot = Report([ValueDrift(column=c, method=method, threshold=.1) for c in columns]).run(
        current_data=Dataset.from_pandas(production, data_definition=definition),
        reference_data=Dataset.from_pandas(reference, data_definition=definition))
    snapshot.save_html(str(path))
    snapshot.save_json(str(path.with_suffix(".json")))


def generate_reference(args, loader=None):
    require(args.batch_size > 0, "batch size must be positive")
    identity = selection_identity(args.selection)
    rows, info = dataset(args.reference, args.text_column, args.id_column)
    require(not args.output.exists(), "reference output already exists")
    if loader is None:
        from api.main import load_selected_model
        loader = load_selected_model
    with tempfile.TemporaryDirectory(prefix="monitor-reference-") as destination:
        predictor = loader(args.selection, destination)
        require(predictor.manifest["logical_id"] == identity["logical_id"] and
                predictor.manifest["model_sha256"] == identity["model_sha256"] and
                predictor.manifest["configuration"] == identity["configuration"], "loaded model mismatch")
        labels = predictor.predict_texts([r["text"] for r in rows], batch_size=args.batch_size, device="cpu")
    require(len(labels) == len(rows), "reference prediction count mismatch")
    require(all(isinstance(label, (int, np.integer)) and not isinstance(label, (bool, np.bool_))
                for label in labels), "noninteger reference cluster")
    records = [{"source_row_id": r["source_row_id"], "source_row_index": r["source_row_index"],
                "model_id": identity["logical_id"], "success": True, "cluster_id": int(label)}
               for r, label in zip(rows, labels)]
    validate_labels(records, rows, identity, False)
    require(fingerprint(args.reference) == info["sha256"], "reference changed during inference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids replacing a previously validated reference artifact.
    artifact = {"schema_version": 1, "kind": "reference_predictions", "created_at": utc_now(),
                "dataset": info, "model": identity, "records": records,
                "records_sha256": canonical_hash(records), "strategy": "verified frozen MLflow bundle; CPU inference"}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(artifact, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return {"reference_predictions": str(args.output), "rows": len(rows), "model_id": identity["logical_id"]}


def monitor(args):
    require(0 < args.alpha < 1 and 0 < args.practical_threshold <= 1 and args.resamples >= 99,
            "invalid statistical configuration")
    require((args.reference_predictions is None) == (args.production_run is None),
            "supply both reference predictions and production run, or neither")
    identity = selection_identity(args.selection)
    reference, ri = dataset(args.reference, args.text_column, args.id_column)
    production, pi = dataset(args.production, args.text_column, args.id_column)
    require(not ({r["source_row_id"] for r in reference} & {r["source_row_id"] for r in production}),
            "reference and production IDs overlap; cohorts must be disjoint")
    rf, pf = features(reference), features(production)
    results = {"schema_version": 1, "created_at": utc_now(), "datasets": {"reference": ri, "production": pi},
               "configuration": {"alpha": args.alpha, "multiple_testing": "Holm across all available tests",
                                 "resamples": args.resamples, "seed": args.seed,
                                 "practical_threshold": args.practical_threshold, "allow_subset": args.allow_subset},
               "versions": {name: importlib.metadata.version(name) for name in ("evidently", "numpy", "scipy", "pandas")},
               "input": input_analysis(rf, pf, args.resamples, args.seed, args.practical_threshold),
               "predictions": {"status": "unavailable", "reason": "No recorded production/reference prediction pair supplied"},
               "embeddings": {"status": "deferred", "reason": "Requires separate multivariate design and frozen encoder validation"},
               "interpretation": "Distribution shift is not a measure of clustering quality or performance degradation."}
    tests = list(results["input"].values())
    a = b = None
    if args.production_run:
        # Refuse a writer racing this snapshot; retain hashes of the exact checkpoint inputs.
        with run_lock(args.production_run):
            b, bc = production_labels(args.production_run, production, pi, identity, args.allow_subset)
        a, ac = reference_labels(args.reference_predictions, reference, ri, identity, args.allow_subset)
        results["predictions"] = cluster_analysis(a, b, identity["n_clusters"], args.resamples, args.seed, args.practical_threshold)
        results["predictions"].update(model=identity, coverage={"reference": ac, "production": bc},
                                      reference_artifact_sha256=ac["artifact_sha256"])
        tests.append(results["predictions"])
    holm(tests, args.alpha)
    require(fingerprint(args.reference) == ri["sha256"] and fingerprint(args.production) == pi["sha256"],
            "dataset changed during analysis")
    args.output.mkdir(parents=True, exist_ok=False)
    html_report(rf, pf, args.output / "input.html")
    if a is not None:
        html_report(pd.DataFrame({"cluster_id": [str(x) for x in a]}),
                    pd.DataFrame({"cluster_id": [str(x) for x in b]}), args.output / "predictions.html", True)
    atomic_json(args.output / "results.json", results)
    return {"output_directory": str(args.output), "prediction_status": results["predictions"]["status"],
            "input": {key: {k: v[k] for k in ("adjusted_p_value", "effect_size", "statistically_significant", "practically_important")}
                      for key, v in results["input"].items()}}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("report", "predict-reference"):
        command = sub.add_parser(name)
        command.add_argument("--reference", type=Path, default=Path("data/processed/avis_4k_reference.csv"))
        command.add_argument("--selection", type=Path, default=Path("config/selection.docker.json"))
        command.add_argument("--text-column", default="commentaire")
        command.add_argument("--id-column", default="")
        command.add_argument("--output", type=Path, required=True)
        if name == "predict-reference":
            command.add_argument("--batch-size", type=int, default=32)
        else:
            command.add_argument("--production", type=Path, default=Path("data/processed/avis_8k_production.csv"))
            command.add_argument("--reference-predictions", type=Path)
            command.add_argument("--production-run", type=Path)
            command.add_argument("--allow-subset", action="store_true")
            command.add_argument("--alpha", type=float, default=.05)
            command.add_argument("--practical-threshold", type=float, default=.1)
            command.add_argument("--resamples", type=int, default=1999)
            command.add_argument("--seed", type=int, default=42)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        summary = generate_reference(args) if args.command == "predict-reference" else monitor(args)
    except MonitoringError as error:
        print(json.dumps({"status": "failed", "reason": str(error)}))
        return 1
    except Exception:
        # Remote loader failures can contain credentials; keep CLI diagnostics safe.
        print(json.dumps({"status": "failed", "reason": "Invalid inputs or monitoring/inference failure; no successful report claimed"}))
        return 1
    print(json.dumps(summary, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
