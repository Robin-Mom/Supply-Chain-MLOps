"""Offline integration checks: actual clustering and MLflow, synthetic encoder."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import evaluate
import select_model
import train
from predict import load_bundle, predict_embeddings


class OfflineEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        return np.random.default_rng(42).normal(size=(len(texts), 6))

    def save_pretrained(self, directory):
        Path(directory).mkdir()
        (Path(directory) / 'config.json').write_text('{}')


class PipelineTests(unittest.TestCase):
    def test_selection_handles_invalid_scores_and_ties(self):
        metrics = {'spherical_kmeans': {'silhouette_cosine': 0.5},
                   'kmeans': {'silhouette_cosine': 0.5}}
        ranking, _ = select_model.rank_models(metrics)
        self.assertEqual(ranking[0]['method'], 'kmeans')
        metrics['kmeans']['silhouette_cosine'] = None
        ranking, excluded = select_model.rank_models(metrics)
        self.assertEqual(ranking[0]['method'], 'spherical_kmeans')
        self.assertEqual(excluded, ['kmeans'])
        with self.assertRaises(ValueError):
            select_model.rank_models({'kmeans': {'silhouette_cosine': float('nan')}})
        ranking, _ = select_model.rank_models(
            {'kmeans': {'davies_bouldin': 2}, 'spherical_kmeans': {'davies_bouldin': 1}},
            'davies_bouldin', 'minimize')
        self.assertEqual(ranking[0]['method'], 'spherical_kmeans')

    def test_local_train_evaluate_select(self):
        import mlflow
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            uri = f'sqlite:///{root / "mlflow.db"}'
            mlflow.set_tracking_uri(uri)
            experiment = 'offline-pipeline'
            experiment_id = mlflow.create_experiment(experiment, artifact_location=(root / 'mlartifacts').as_uri())
            pd.DataFrame({'commentaire': [f'avis {i}' for i in range(30)]}).to_csv(root / 'input.csv', index=False)
            with patch.dict(sys.modules, {'sentence_transformers': SimpleNamespace(SentenceTransformer=OfflineEncoder)}):
                train.main(['--input', str(root / 'input.csv'), '--output-dir', str(root / 'models'),
                            '--n-clusters', '3', '--n-init', '2', '--max-iter', '20',
                            '--tracking-uri', uri, '--experiment', experiment])
            client = mlflow.MlflowClient(tracking_uri=uri)
            training_runs = client.search_runs([experiment_id])
            self.assertEqual(len(training_runs), 2)
            original_ids = {run.info.run_id for run in training_runs}
            for run in training_runs:
                self.assertEqual(run.data.tags['mlflow.runType'], 'training')
                self.assertEqual(run.data.tags['mlflow.runSourceType'], 'training')
            # Evaluation also repairs the classification of older pipeline runs.
            for run in training_runs:
                client.set_tag(run.info.run_id, 'mlflow.runType', 'genai_evaluate')
                client.set_tag(run.info.run_id, 'mlflow.runSourceType', 'evaluation')
            # Evaluation must resume the saved runs even with another active experiment.
            mlflow.set_experiment('unrelated-experiment')
            evaluate.main(['--artifact-dir', str(root / 'models'), '--output-dir', str(root / 'reports'),
                           '--metrics-output', str(root / 'metrics.json'), '--skip-keywords', '--log-mlflow'])
            select_model.main(['--artifact-dir', str(root / 'models'), '--metrics', str(root / 'metrics.json'),
                               '--output', str(root / 'selection.json')])
            selection = json.loads((root / 'selection.json').read_text())
            self.assertIn(selection['method'], train.METHODS)
            self.assertEqual(len(selection['model_sha256']), 64)
            X = np.load(root / 'models/embeddings.npy')
            client = mlflow.MlflowClient(tracking_uri=uri)
            for method in train.METHODS:
                bundle = load_bundle(root / 'models', method)
                labels = np.load(root / 'models' / method / 'labels.npy')
                np.testing.assert_array_equal(predict_embeddings(bundle, X), labels)
                repeated = train.fit_clustering(X, method, train.default_config(3))[2]
                np.testing.assert_array_equal(repeated, train.fit_clustering(X, method, train.default_config(3))[2])
                run_info = json.loads((root / 'models' / method / 'mlflow_run.json').read_text())
                run = client.get_run(run_info['run_id'])
                self.assertEqual(run.info.experiment_id, experiment_id)
                self.assertEqual(run.data.tags['mlflow.runType'], 'training')
                self.assertEqual(run.data.tags['mlflow.runSourceType'], 'training')
                self.assertEqual(run_info['experiment_id'], experiment_id)
                self.assertEqual(run_info['experiment_name'], experiment)
                self.assertIn('training_silhouette_cosine', run.data.metrics)
                self.assertEqual(run.data.params['n_init'], '2')
                self.assertEqual(run.data.params['max_iter'], '20')
            evaluated_runs = client.search_runs([experiment_id])
            self.assertEqual({run.info.run_id for run in evaluated_runs}, original_ids)
            self.assertTrue(all(run.info.status == 'FINISHED' for run in evaluated_runs))


if __name__ == '__main__':
    unittest.main()
