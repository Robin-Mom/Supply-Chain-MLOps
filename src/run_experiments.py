"""Sequential DVC experiment sweep; global selection only after all 30 runs."""
from __future__ import annotations

import argparse
from itertools import product
import json
import os
from pathlib import Path
import subprocess

import yaml

from experiment import (configuration, digest, evaluation_dir, load_params,
                        sweep_id, tracking, validate_splits, write_json)


def pin_revisions(path):
    """Resolve explicitly requested model revisions, without downloading weights."""
    from huggingface_hub import HfApi
    params = yaml.safe_load(Path(path).read_text())
    for spec in params["encoders"].values():
        if not spec.get("revision"):
            spec["revision"] = HfApi().model_info(spec["id"], revision="main", timeout=30).sha
    Path(path).write_text(yaml.safe_dump(params, sort_keys=False, allow_unicode=True))


def plan(params):
    prefix = f"sc-{sweep_id(params)}"
    for alias, method, seed in product(params["encoders"], params["exploration"]["methods"], params["exploration"]["seeds"]):
        name = f"{prefix}-{alias}-{method}-{seed}"
        yield name, ["dvc", "exp", "run", "evaluate", "--name", name,
                     "-S", f"experiment.encoder={alias}", "-S", f"experiment.method={method}",
                     "-S", f"experiment.seed={seed}"], {"encoder": alias, "method": method, "seed": seed}


def record_revision(name, params, directory):
    """Experiment commits only exist after DVC has finished the run."""
    import mlflow
    refs = subprocess.check_output(["git", "for-each-ref", "--format=%(objectname) %(refname)", "refs/exps"], text=True)
    matches = [line.split()[0] for line in refs.splitlines() if line.split()[1].endswith("/" + name)]
    if len(set(matches)) != 1:
        raise ValueError(f"Cannot resolve unique DVC revision for {name}.")
    revision = matches[0]
    client = mlflow.MlflowClient()
    logical_id = digest(configuration(params))
    # Read the actual cached result at this commit, even if the stages were skipped.
    # Selecting the latest matching MLflow run could link a different execution.
    import dvc.api
    result = json.loads(dvc.api.read(str(evaluation_dir(params) / "result.json"), repo=".", rev=revision))
    if result["logical_id"] != logical_id or result["sweep_id"] != sweep_id(params):
        raise ValueError("DVC result does not match the requested configuration.")
    evaluation = client.get_run(result["eval_mlflow_run_id"])
    training = client.get_run(result["train_mlflow_run_id"])
    if (training.info.status != "FINISHED" or training.data.tags.get("logical_id") != logical_id
            or evaluation.info.status != "FINISHED"
            or evaluation.data.tags.get("train_mlflow_run_id") != training.info.run_id):
        raise ValueError("The cached evaluation has no valid matching training run.")
    ids = []
    for run in (training, evaluation):
        names = json.loads(run.data.tags.get("dvc_experiment_names", "[]"))
        original = run.data.tags.get("dvc_experiment_name")
        names = sorted(set(names + [name] + ([original] if original else [])))
        client.set_tag(run.info.run_id, "dvc_revision", revision)
        client.set_tag(run.info.run_id, "dvc_experiment_name", name)
        client.set_tag(run.info.run_id, "dvc_experiment_names", json.dumps(names))
        ids.append(run.info.run_id)
    write_json(Path(directory) / f"{name}.json", {"dvc_experiment_name": name, "dvc_revision": revision, "mlflow_run_ids": ids})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--pin-revisions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--temp", action="store_true", help="Run each experiment in an isolated DVC workspace.")
    parser.add_argument("--resume", action="store_true", help="Reuse experiments already recorded under this sweep's DVC names.")
    parser.add_argument("--output", default="reports/experiment_selection")
    args = parser.parse_args(argv)
    if args.pin_revisions:
        pin_revisions(args.params)
        print("Pinned revisions. Review and commit params.yaml before starting the sweep.")
        return
    if Path(args.params) != Path("params.yaml"):
        parser.error("DVC uses the repository params.yaml; alternate files are unsupported for the sweep.")
    params = load_params(args.params)
    validate_splits(params)
    commands = list(plan(params))
    if args.dry_run:
        import shlex
        for name, command, _ in commands:
            print(f"SUPPLY_CHAIN_DVC_NAME={name} " + shlex.join(command + (["--temp"] if args.temp else [])))
        return
    tracking(params)
    for name, command, active in commands:
        active_params = {**params, "experiment": active}
        if args.resume:
            refs = subprocess.check_output(["git", "for-each-ref", "--format=%(refname)", "refs/exps"], text=True)
            if any(ref.endswith("/" + name) for ref in refs.splitlines()):
                record_revision(name, active_params, Path(args.output).with_name("dvc_run_index"))
                continue
        environment = {**os.environ, "SUPPLY_CHAIN_DVC_NAME": name}
        subprocess.run(command + (["--temp"] if args.temp else []), env=environment, check=True)
        record_revision(name, active_params, Path(args.output).with_name("dvc_run_index"))
    from select_best import select
    select(params, args.output)


if __name__ == "__main__":
    main()
