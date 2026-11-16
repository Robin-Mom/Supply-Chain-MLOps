"""Offline HTTP and selected-artifact integration tests."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from api.main import create_app, load_selected_model
from experiment import PREPROCESSING, digest, file_hash
from train import SphericalKMeans
from sklearn.cluster import KMeans


@pytest.fixture
def config():
    return {"encoder": {"id": "offline-encoder", "revision": "a" * 40, "max_seq_length": 128},
            "method": "kmeans", "preprocessing": copy.deepcopy(PREPROCESSING),
            "hyperparameters": {"random_state": 42, "n_clusters": 2, "n_init": 10, "max_iter": 300}}


@pytest.fixture
def service(config):
    predictor = SimpleNamespace(
        manifest={"logical_id": digest(config), "configuration": config},
        predict_texts=Mock(side_effect=lambda texts: [len(text.strip()) for text in texts]))
    loader = Mock(return_value=predictor)
    return create_app(loader=loader), loader, predictor


def test_lifecycle_metadata_and_reuse(service):
    app, loader, predictor = service
    with TestClient(app) as client:
        assert client.get("/health").json()["model_ready"] is True
        for texts in (["Bonjour"], ["Trois", "Un", "Quatre"]):
            response = client.post("/predict", json={"texts": texts})
            assert response.status_code == 200
            assert response.json() == {
                "predictions": [{"cluster_id": len(text)} for text in texts],
                "model_id": predictor.manifest["logical_id"]}
        info = client.get("/model/info").json()
        assert info == {"model_id": predictor.manifest["logical_id"],
                        "encoder_id": "offline-encoder", "encoder_revision": "a" * 40,
                        "method": "kmeans", "random_seed": 42, "preprocessing_version": 1}
        loader.assert_called_once()
        assert predictor.predict_texts.call_count == 2
    assert app.state.predictor is None


@pytest.mark.parametrize("payload", [None, {}, {"texts": []}, {"texts": [""]},
    {"texts": [" \n\t"]}, {"texts": [None]}, {"texts": [12]}, {"texts": "bonjour"},
    {"texts": ["ok", ""]}, {"texts": ["a" * 5001]}, {"texts": ["ok"] * 65},
    {"texts": ["ok"], "extra": True}])
def test_invalid_input(service, payload):
    app, _, predictor = service
    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 422
        predictor.predict_texts.assert_not_called()


def test_readiness_before_startup_and_failed_initialization(caplog):
    app = create_app(loader=Mock(side_effect=RuntimeError("secret-token /private/path")))
    # Without a context manager TestClient does not execute lifespan.
    assert TestClient(app).get("/health").json()["model_ready"] is False
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json() == {"status": "ok", "model_ready": False, "model_status": "unavailable"}
        for response in (client.get("/model/info"), client.post("/predict", json={"texts": ["Bonjour"]})):
            assert response.status_code == 503
            assert "secret-token" not in response.text
    assert "secret-token" not in caplog.text


def test_prediction_error_is_sanitized(service, caplog):
    app, _, predictor = service
    predictor.predict_texts.side_effect = RuntimeError("secret-token")
    with TestClient(app) as client:
        response = client.post("/predict", json={"texts": ["Bonjour"]})
        assert response.status_code == 500
        assert response.json() == {"detail": "Prediction failed."}
    assert "secret-token" not in caplog.text


@pytest.mark.parametrize("method", ["kmeans", "spherical_kmeans"])
@pytest.mark.parametrize("defect", [None, "checksum", "selection", "revision", "seed", "warmup"])
def test_real_predictor_loading_offline(tmp_path, monkeypatch, config, method, defect):
    """Use the real predictor and estimator; only artifact transport/encoder are fake."""
    import mlflow
    import tracking
    import sentence_transformers

    config["method"] = method
    estimator_class = KMeans if method == "kmeans" else SphericalKMeans
    estimator = estimator_class(**config["hyperparameters"])
    estimator.cluster_centers_ = np.eye(2, dtype=np.float32)
    estimator.n_features_in_ = 2
    if method == "kmeans":
        estimator._n_threads = 1
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": estimator, "method": method, "logical_id": digest(config),
                 "encoder": config["encoder"], "preprocessing": PREPROCESSING, "n_features": 2}, model_path)
    manifest = {"logical_id": digest(config), "configuration": config, "transformations": [],
                "model_sha256": file_hash(model_path), "model_uri": "runs:/train/clustering/model.joblib",
                "mlflow_run_id": "train"}
    selection = {**manifest, "final_seed": 42, "train_mlflow_run_id": "train",
                 "tracking_uri": "https://example.test/repo.mlflow",
                 "manifest_uri": "runs:/train/clustering/manifest.json"}
    if defect == "checksum":
        manifest["model_sha256"] = "invalid"
    elif defect == "selection":
        selection["model_sha256"] = "different"
    elif defect == "revision":
        selection = copy.deepcopy(selection)
        selection["configuration"]["encoder"]["revision"] = "main"
    elif defect == "seed":
        selection["final_seed"] = 43
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    download = Mock(side_effect=lambda *, artifact_uri, dst_path: str(
        manifest_path if artifact_uri.endswith("manifest.json") else model_path))
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", download)
    monkeypatch.setattr(tracking, "configure_tracking", Mock())
    monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: selection["tracking_uri"])
    encoder = Mock()
    encoder.encode.side_effect = lambda texts, **kwargs: np.asarray(
        [[1, 0] if text == "positif" else [0, 1] for text in texts], dtype=np.float32)
    if defect == "warmup":
        encoder.encode.side_effect = RuntimeError("encoder failed")
    constructor = Mock(return_value=encoder)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", constructor)
    monkeypatch.setattr(estimator_class, "fit", Mock(side_effect=AssertionError("fit forbidden")))
    monkeypatch.setattr(estimator_class, "fit_predict", Mock(side_effect=AssertionError("fit_predict forbidden")))
    if defect:
        with pytest.raises((ValueError, RuntimeError)):
            load_selected_model(selection_path, tmp_path / "download")
        return
    with TestClient(create_app(selection_path=selection_path)) as client:
        assert client.get("/health").status_code == 200
        for _ in range(2):
            response = client.post("/predict", json={"texts": ["positif", "négatif", "positif"]})
            assert response.status_code == 200
            assert response.json()["predictions"] == [{"cluster_id": 0}, {"cluster_id": 1}, {"cluster_id": 0}]
        assert download.call_count == 2
        constructor.assert_called_once_with("offline-encoder", revision="a" * 40, device=None,
            trust_remote_code=False, model_kwargs={"use_safetensors": True})
        assert encoder.max_seq_length == 128
        estimator_class.fit.assert_not_called()
        estimator_class.fit_predict.assert_not_called()


def test_missing_selection(tmp_path):
    with TestClient(create_app(selection_path=tmp_path / "missing.json")) as client:
        assert client.get("/health").status_code == 503
