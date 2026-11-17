"""Offline held-out pipeline checks, including DVC temporary workspaces."""
from __future__ import annotations

from contextlib import contextmanager
import copy
from itertools import product
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlflow
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import embed
import evaluate
import experiment
import predict
import run_experiments
import select_best
import train

FAKE_ENCODER = '''
import os
from pathlib import Path
import numpy as np
class SentenceTransformer:
    def __init__(self, *args, **kwargs):
        if os.environ.get("ENCODER_COUNT_FILE"):
            with Path(os.environ["ENCODER_COUNT_FILE"]).open("a") as stream:
                stream.write("loaded\\n")
    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            number = int(text.split()[-1])
            row = np.zeros(6, dtype=np.float32)
            row[number % 3] = 1
            row[3:] = np.random.default_rng(number).uniform(0.01, 0.08, 3)
            rows.append(row)
        return np.asarray(rows)
'''
namespace = {}
exec(FAKE_ENCODER, namespace)
OfflineEncoder = namespace["SentenceTransformer"]


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def fixture(root):
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    params["data"]["train_rows"], params["data"]["eval_rows"] = 30, 12
    params["clustering"].update(n_clusters=3, n_init=1, max_iter=15)
    params["embedding"]["batch_size"] = 7
    params["evaluation"]["working_memory_mb"] = 1
    params["tracking"]["uri"] = "sqlite:///" + str(root.parent / "tracking" / "mlflow.db")
    (root.parent / "tracking").mkdir(exist_ok=True)
    data = root / "data/train_eval"
    data.mkdir(parents=True)
    for split, count, offset in (("train", 30, 0), ("eval", 12, 100)):
        pd.DataFrame({"Unnamed: 0": range(offset, offset + count),
                      "commentaire": [f"review {i}" for i in range(count)]}).to_csv(data / f"{split}_{'20k' if split == 'train' else '4k'}.csv", index=False)
    (root / "params.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
    return params


class HeldOutPipelineTests(unittest.TestCase):
    def test_complete_matrix_cache_integrity_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            params = fixture(root)
            with working_directory(root), patch.dict(os.environ, {"MLFLOW_DISABLE_AGENT_HINT": "1", "SUPPLY_CHAIN_TRACKING_URI": ""}):
                self.assertEqual(len(list(run_experiments.plan(params))), 20)
                experiment.validate_splits(params)
                modules = {"sentence_transformers": SimpleNamespace(SentenceTransformer=OfflineEncoder)}
                with patch.dict(sys.modules, modules), patch.object(embed, "encode_to_file", wraps=embed.encode_to_file) as encoding:
                    for alias in params["encoders"]:
                        embed.build(params, alias)
                    self.assertEqual(encoding.call_count, 4)
                X, _ = experiment.load_embeddings(params, "mpnet", "train")
                self.assertIsInstance(X, np.memmap)
                self.assertEqual(X.dtype, np.float32)
                # Changed revisions and corrupted cached arrays must be rejected.
                changed = copy.deepcopy(params)
                changed["encoders"]["mpnet"]["revision"] = "0" * 40
                with self.assertRaisesRegex(ValueError, "cache"):
                    experiment.load_embeddings(changed, "mpnet", "train")
                manifests, evaluations = [], []
                for alias, method, seed in product(params["encoders"], params["exploration"]["methods"], params["exploration"]["seeds"]):
                    params["experiment"] = {"encoder": alias, "method": method, "seed": seed}
                    with patch.object(train, "fit_clustering", wraps=train.fit_clustering) as fit:
                        manifest = train.train_experiment(params)
                        self.assertEqual(fit.call_args.args[0].shape[0], 30)
                    manifests.append(manifest)
                    with patch.object(train, "fit_clustering", side_effect=AssertionError("Inference must not fit")), \
                         patch.object(train.KMeans, "fit", side_effect=AssertionError("Inference must not fit")), \
                         patch.object(train.SphericalKMeans, "fit", side_effect=AssertionError("Inference must not fit")):
                        prediction = predict.predict_experiment(params)
                        evaluations.append(evaluate.evaluate_experiment(params))
                    self.assertEqual(prediction["train_mlflow_run_id"], manifest["mlflow_run_id"])
                    if len(evaluations) == 1:
                        with self.assertRaisesRegex(ValueError, "Incomplete sweep"):
                            select_best.select(params, root / "incomplete")
                selected = select_best.select(params, root / "selection")
                table = pd.read_csv(root / "selection/comparison.csv")
                self.assertEqual(len(table), 4)
                self.assertTrue(table.ari_pairs.eq(10).all())
                self.assertEqual(len(pd.read_csv(root / "selection/ari_pairs.csv")), 40)
                self.assertEqual(selected["final_seed"], 42)
                self.assertFalse(selected["future_refit"]["executed"])
                self.assertEqual(len({item["logical_id"] for item in manifests}), 20)
                client = mlflow.MlflowClient()
                for kind in ("train", "eval"):
                    exp = client.get_experiment_by_name(params["tracking"][kind + "_experiment"])
                    runs = client.search_runs([exp.experiment_id], filter_string=f"tags.pipeline_role = '{kind}'")
                    self.assertEqual(len(runs), 20)
                    for run in runs:
                        self.assertEqual(run.info.status, "FINISHED")
                        for metric in ("silhouette_cosine", "inertia", "ari_mean", "ari_std", "ari_pairs"):
                            self.assertIn(metric, run.data.metrics)
                        self.assertEqual(run.data.metrics["ari_pairs"], 4)
                        item = next(value for value in evaluations if value[kind + "_mlflow_run_id"] == run.info.run_id)
                        self.assertEqual(run.data.metrics["silhouette_cosine"], item["metrics"]["silhouette_cosine"])
                        self.assertEqual(run.data.metrics["inertia"], item["metrics"]["inertia"])
                        peers = [value for value in evaluations
                                 if value["configuration"]["encoder_alias"] == item["configuration"]["encoder_alias"]
                                 and value["configuration"]["method"] == item["configuration"]["method"]
                                 and value["logical_id"] != item["logical_id"]]
                        own_labels = pd.read_csv(item["prediction"]["predictions_path"]).cluster_id
                        expected_ari = np.mean([evaluate.adjusted_rand_score(
                            own_labels, pd.read_csv(peer["prediction"]["predictions_path"]).cluster_id)
                            for peer in peers])
                        self.assertAlmostEqual(run.data.metrics["ari_mean"], expected_ari)
                    self.assertTrue(exp.artifact_location.startswith((root.parent / "tracking").as_uri()))
                artifacts = client.list_artifacts(manifests[0]["mlflow_run_id"])
                self.assertEqual([item.path for item in artifacts], ["clustering"])
                service = predict.ExperimentPredictor.from_mlflow(selected["manifest_uri"], root / "inference")
                with patch.dict(sys.modules, modules):
                    self.assertEqual(service.predict_texts(["review 0", "review 1"]).shape, (2,))
                # An incomplete or mismatched partition is rejected before ARI.
                ids = evaluations[-1]["observation_ids_sha256"]
                self.assertEqual(len(ids), 64)
                X, _ = experiment.load_embeddings(params, "mpnet", "eval")
                undefined = evaluate.held_out_silhouette(X, np.zeros(len(X)), params["evaluation"])
                self.assertIsNone(undefined["silhouette_cosine"])
                config = {**params["evaluation"], "sample_size": 8}
                a = evaluate.held_out_silhouette(X, np.arange(len(X)) % 3, config)
                b = evaluate.held_out_silhouette(X, np.arange(len(X)) % 2, config)
                self.assertEqual(a["sample_indices_sha256"], b["sample_indices_sha256"])
                path = experiment.cache_dir("mpnet") / "eval.npy"
                with path.open("ab") as stream:
                    stream.write(b"corruption")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    experiment.load_embeddings(params, "mpnet", "eval")

    def test_isolated_dvc_reuses_embeddings_between_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "repo"
            root.mkdir()
            params = fixture(root)
            shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
            for name in ("dvc.yaml", "pyproject.toml", "uv.lock"):
                shutil.copy(ROOT / name, root / name)
            (root / ".gitignore").write_text("models/\n*.db\n__pycache__/\n")
            fakes = parent / "fakes"
            fakes.mkdir()
            (fakes / "sentence_transformers.py").write_text(FAKE_ENCODER)
            (fakes / "torch.py").write_text("from types import SimpleNamespace\ncuda = SimpleNamespace(is_available=lambda: False)\n")
            env = {**os.environ, "PATH": str(ROOT / ".venv/bin") + os.pathsep + os.environ["PATH"],
                   "PYTHONPATH": str(fakes), "XDG_CACHE_HOME": str(parent / "cache"),
                   "XDG_CONFIG_HOME": str(parent / "config"), "DVC_NO_ANALYTICS": "1",
                   "MLFLOW_DISABLE_AGENT_HINT": "1", "SUPPLY_CHAIN_TRACKING_URI": params["tracking"]["uri"],
                   "ENCODER_COUNT_FILE": str(parent / "encoder_count.txt")}
            def command(*args):
                completed = subprocess.run(args, cwd=root, env=env, text=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.assertEqual(completed.returncode, 0, completed.stdout[-8000:])
                return completed.stdout
            command("git", "init", "-q")
            command("git", "config", "user.email", "offline-test@example.invalid")
            command("git", "config", "user.name", "Offline test")
            command("dvc", "init", "-v")
            command("git", "add", ".")
            command("git", "commit", "-qm", "offline fixture")
            for seed in (42, 43):
                env["SUPPLY_CHAIN_DVC_NAME"] = f"offline-{seed}"
                command("dvc", "exp", "run", "evaluate", "--temp", "--name", f"offline-{seed}",
                        "-S", f"experiment.seed={seed}")
            self.assertEqual((parent / "encoder_count.txt").read_text().splitlines(), ["loaded"])
            self.assertFalse((root / "models").exists())
            # DVC --temp overrides do not update the in-memory fixture params.
            active_params = {**params, "experiment": {**params["experiment"], "seed": 42}}
            with working_directory(root), patch.dict(os.environ, env):
                experiment.tracking(params)
                run_experiments.record_revision("offline-42", active_params, parent / "index")
                info = experiment.read_json(parent / "index/offline-42.json")
                self.assertEqual(len(info["dvc_revision"]), 40)
                self.assertEqual(len(info["mlflow_run_ids"]), 2)
            # A new DVC name may restore the same artifacts and original MLflow IDs.
            env["SUPPLY_CHAIN_DVC_NAME"] = "offline-cached"
            command("dvc", "exp", "run", "evaluate", "--temp", "--name", "offline-cached", "-S", "experiment.seed=42")
            with working_directory(root), patch.dict(os.environ, env):
                run_experiments.record_revision("offline-cached", active_params, parent / "index")
                reused = experiment.read_json(parent / "index/offline-cached.json")
                self.assertEqual(reused["mlflow_run_ids"], info["mlflow_run_ids"])
            self.assertEqual((parent / "encoder_count.txt").read_text().splitlines(), ["loaded"])


if __name__ == "__main__":
    unittest.main()
