"""Compare a complete MLflow sweep and select a predetermined-seed model."""
from __future__ import annotations

import argparse
from itertools import combinations, product
from pathlib import Path
import tempfile

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from experiment import (configuration, digest, file_hash, load_params, read_json,
                        reviews, sweep_id, tracking, write_json)


def collect_results(params, client, download_dir):
    current_sweep = sweep_id(params)
    eval_hash = file_hash(params["data"]["eval"])
    experiment = client.get_experiment_by_name(params["tracking"]["eval_experiment"])
    if experiment is None:
        raise ValueError("Evaluation experiment does not exist.")
    # MLflow search uses pagination; never silently select from a truncated page.
    runs, token = [], None
    while True:
        page = client.search_runs([experiment.experiment_id],
            filter_string=f"tags.sweep_id = '{current_sweep}' and tags.pipeline_role = 'eval'",
            order_by=["attributes.start_time DESC"], max_results=1000, page_token=token)
        runs.extend(page)
        token = page.token
        if not token:
            break
    expected = {}
    for alias, method, seed in product(params["encoders"], params["exploration"]["methods"], params["exploration"]["seeds"]):
        active = {**params, "experiment": {"encoder": alias, "method": method, "seed": seed}}
        config = configuration(active)
        expected[digest(config)] = config
    found = {}
    _, expected_ids = reviews(params, "eval")
    train_experiment = client.get_experiment_by_name(params["tracking"]["train_experiment"])
    for run in runs:
        logical_id = run.data.tags.get("logical_id")
        if run.info.status != "FINISHED" or logical_id in found:
            continue
        if logical_id not in expected:
            raise ValueError("Unexpected configuration in the sweep.")
        path = client.download_artifacts(run.info.run_id, "evaluation/result.json", str(download_dir))
        result = read_json(path)
        if (result["configuration"] != expected[logical_id] or result["logical_id"] != logical_id
                or result["eval_mlflow_run_id"] != run.info.run_id or result["sweep_id"] != current_sweep
                or result["evaluation_config"] != params["evaluation"]
                or result["eval_input_sha256"] != eval_hash):
            raise ValueError("Inconsistent evaluation metadata.")
        train_run = client.get_run(result["train_mlflow_run_id"])
        if (train_experiment is None or train_run.info.experiment_id != train_experiment.experiment_id
                or train_run.info.status != "FINISHED" or train_run.data.tags.get("logical_id") != logical_id
                or run.data.tags.get("train_mlflow_run_id") != train_run.info.run_id):
            raise ValueError("Broken training/evaluation MLflow link.")
        prediction_path = client.download_artifacts(run.info.run_id, "evaluation/assignments.csv", str(download_dir))
        if file_hash(prediction_path) != result["prediction"]["predictions_sha256"]:
            raise ValueError("Prediction checksum mismatch.")
        predictions = pd.read_csv(prediction_path, dtype={"observation_id": str})
        ids = predictions.observation_id.tolist()
        if (ids != expected_ids or len(set(ids)) != len(ids)
                or digest(ids) != result["observation_ids_sha256"]
                or not predictions.logical_id.eq(logical_id).all()):
            raise ValueError("Invalid prediction identifiers.")
        result["labels"] = predictions.cluster_id.to_numpy()
        result["observation_ids"] = ids
        result["dvc_revision"] = run.data.tags.get("dvc_revision")
        result["dvc_experiment_name"] = run.data.tags.get("dvc_experiment_name")
        found[logical_id] = result
    if set(found) != set(expected):
        raise ValueError(f"Incomplete sweep: {len(found)}/{len(expected)} completed configurations; selection refused.")
    return list(found.values())


def aggregate_results(results):
    """Compute sample standard deviation (ddof=1) and all ten ARI pairs."""
    reference = results[0]
    for result in results:
        if (result["observation_ids"] != reference["observation_ids"]
                or result["metrics"]["sample_indices_sha256"] != reference["metrics"]["sample_indices_sha256"]):
            raise ValueError("All configurations must use identical observations and silhouette samples.")
    groups = {}
    for result in results:
        config = result["configuration"]
        groups.setdefault((config["encoder_alias"], config["method"]), []).append(result)
    rows, pairs = [], []
    for (alias, method), group in sorted(groups.items()):
        group.sort(key=lambda item: item["configuration"]["hyperparameters"]["random_state"])
        if len(group) != 5:
            raise ValueError("Each configuration group must contain five seeds.")
        aris = []
        for a, b in combinations(group, 2):
            ari = float(adjusted_rand_score(a["labels"], b["labels"]))
            aris.append(ari)
            pairs.append({"encoder": alias, "method": method,
                          "seed_a": a["configuration"]["hyperparameters"]["random_state"],
                          "seed_b": b["configuration"]["hyperparameters"]["random_state"], "ari": ari})
        scores = [item["metrics"]["silhouette_cosine"] for item in group]
        valid = [score for score in scores if score is not None and np.isfinite(score)]
        eligible = len(valid) == 5
        rows.append({"encoder": alias, "method": method, "n_seeds": 5, "valid_silhouettes": len(valid),
                     "eligible": eligible, "silhouette_mean": float(np.mean(valid)) if eligible else None,
                     "silhouette_std": float(np.std(valid, ddof=1)) if eligible else None,
                     "ari_mean": float(np.mean(aris)), "ari_std": float(np.std(aris, ddof=1)), "ari_pairs": len(aris)})
    return rows, pairs


def select(params, output):
    tracking(params)
    client = mlflow.MlflowClient()
    with tempfile.TemporaryDirectory() as directory:
        results = collect_results(params, client, Path(directory))
    rows, pairs = aggregate_results(results)
    ranking = sorted([row for row in rows if row["eligible"]],
                     key=lambda row: (-row["silhouette_mean"], -row["ari_mean"], row["encoder"], row["method"]))
    if not ranking:
        raise ValueError("No group has five defined silhouettes.")
    winner = ranking[0]
    seed = params["selection"]["final_seed"]
    selected = next(result for result in results if result["configuration"]["encoder_alias"] == winner["encoder"]
                    and result["configuration"]["method"] == winner["method"]
                    and result["configuration"]["hyperparameters"]["random_state"] == seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(pairs).to_csv(output / "ari_pairs.csv", index=False)
    write_json(output / "comparison.json", rows)
    manifest = selected["train_manifest"]
    result = {"sweep_id": sweep_id(params), "logical_id": selected["logical_id"], "configuration": selected["configuration"],
              "selection_rule": "mean held-out cosine silhouette; ARI breaks exact ties; std uses ddof=1",
              "ranking": ranking, "final_seed": seed, "train_mlflow_run_id": selected["train_mlflow_run_id"],
              "eval_mlflow_run_id": selected["eval_mlflow_run_id"], "dvc_revision": selected["dvc_revision"],
              "dvc_experiment_name": selected["dvc_experiment_name"],
              "model_uri": manifest["model_uri"], "model_sha256": manifest["model_sha256"],
              "manifest_uri": f"runs:/{selected['train_mlflow_run_id']}/clustering/manifest.json",
              "tracking_uri": mlflow.get_tracking_uri(),
              "future_refit": {"executed": False, "train_rows": 24000,
                               "inputs": [params["data"]["train"], params["data"]["eval"]],
                               "seed": seed, "configuration": selected["configuration"]}}
    experiment = mlflow.set_experiment(params["tracking"]["eval_experiment"])
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"selection-{sweep_id(params)}",
                          tags={"sweep_id": sweep_id(params), "pipeline_role": "selection"}) as run:
        result["selection_mlflow_run_id"] = run.info.run_id
        write_json(output / "selection.json", result)
        mlflow.log_metrics({"silhouette_mean": winner["silhouette_mean"], "ari_mean": winner["ari_mean"]})
        mlflow.log_artifacts(str(output), artifact_path="selection")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--output", default="reports/experiment_selection")
    args = parser.parse_args(argv)
    select(load_params(args.params), args.output)


if __name__ == "__main__":
    main()
