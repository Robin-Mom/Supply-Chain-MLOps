"""Isolated metrics tests, without artifact transport or real encoders."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from api.main import create_app


def ready_app():
    predictor = SimpleNamespace(manifest={"logical_id": "offline", "configuration": {
        "encoder": {"id": "offline", "revision": "a" * 40}, "method": "kmeans",
        "hyperparameters": {"random_state": 42}, "preprocessing": {"version": 1}}},
        predict_texts=Mock(return_value=[6]))
    return create_app(loader=Mock(return_value=predictor))


def sample(app, name, **labels):
    return app.state.metrics.registry.get_sample_value(name, labels) or 0


def test_exposition_deltas_and_scrape_exclusion():
    app = ready_app()
    labels = {"method": "POST", "route": "/predict", "status_code": "200"}
    with TestClient(app) as client:
        before = sample(app, "http_requests_total", **labels)
        assert client.post("/predict", json={"texts": ["private-review-sentinel"]}).json() == {
            "predictions": [{"cluster_id": 6}], "model_id": "offline"}
        assert sample(app, "http_requests_total", **labels) == before + 1
        assert sample(app, "http_request_duration_seconds_count", **labels) == 1
        assert sample(app, "http_request_duration_seconds_sum", **labels) >= 0
        assert sample(app, "http_requests_in_progress", method="POST") == 0
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain;")
        families = {f.name: f.type for f in text_string_to_metric_families(response.text)}
        assert families["http_requests"] == "counter"
        assert families["http_request_duration_seconds"] == "histogram"
        assert families["http_requests_in_progress"] == families["model_ready"] == "gauge"
        assert "private-review-sentinel" not in response.text
        assert client.get("/metrics").text == response.text


def test_routes_methods_and_failures_have_bounded_labels():
    app = ready_app()

    @app.get("/items/{item_id}")
    def item(item_id: str):
        return {"id": item_id}

    @app.get("/boom")
    def boom():
        raise RuntimeError("private-error-sentinel")

    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/items/private-id-sentinel").status_code == 200
        assert client.get("/unmatched/private-path-sentinel").status_code == 404
        assert client.request("PRIVATE-METHOD", "/health").status_code == 405
        assert client.post("/predict", json={"texts": []}).status_code == 422
        assert client.get("/boom").status_code == 500
        text = client.get("/metrics").text
        assert all(value not in text for value in ("private-id-sentinel", "private-path-sentinel",
                                                   "private-error-sentinel", "PRIVATE-METHOD"))
        assert sample(app, "http_requests_total", method="GET", route="/items/{item_id}", status_code="200") == 1
        assert sample(app, "http_requests_total", method="GET", route="__unmatched__", status_code="404") == 1
        assert sample(app, "http_requests_total", method="GET", route="/boom", status_code="500") == 1
        assert sample(app, "http_requests_total", method="POST", route="/predict", status_code="422") == 1
        assert sample(app, "http_requests_in_progress", method="GET") == 0


def test_readiness_and_registry_isolation():
    app = ready_app()
    assert sample(app, "model_ready") == 0
    with TestClient(app) as client:
        assert sample(app, "model_ready") == 1
        client.get("/health")
    assert sample(app, "model_ready") == 0
    failed = create_app(loader=Mock(side_effect=RuntimeError("no model")))
    with TestClient(failed) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/metrics").status_code == 200
        assert sample(failed, "model_ready") == 0
        assert sample(failed, "http_requests_total", method="GET", route="/health", status_code="503") == 1
    assert ready_app().state.metrics.registry is not app.state.metrics.registry
    assert not list(ready_app().state.metrics.requests.collect())[0].samples


def test_concurrent_requests_in_progress():
    app = ready_app()
    entered = Barrier(3)
    release = Event()

    @app.get("/blocked")
    def blocked():
        entered.wait(timeout=10)
        assert release.wait(timeout=10)
        return {"ok": True}

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(client.get, "/blocked") for _ in range(2)]
        try:
            entered.wait(timeout=10)
            response = client.get("/metrics")
            assert response.status_code == 200
            assert sample(app, "http_requests_in_progress", method="GET") == 2
        finally:
            release.set()
        assert all(future.result(timeout=10).status_code == 200 for future in futures)
        assert sample(app, "http_requests_in_progress", method="GET") == 0
