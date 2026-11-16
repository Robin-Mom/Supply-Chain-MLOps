"""Opt-in checks against the running local stack; never starts/stops services."""
import json
import os
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_MONITORING_INTEGRATION") != "1",
                                reason="Requires the explicitly started Docker monitoring stack")
BASE = "http://127.0.0.1:8080"
PROM = "http://127.0.0.1:9090"


def get_json(url):
    with urlopen(url, timeout=10) as response:
        return json.load(response)


def query(expression):
    return get_json(PROM + "/api/v1/query?" + urlencode({"query": expression}))["data"]["result"]


def wait_for(check):
    deadline = time.monotonic() + 50
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(1)
    raise AssertionError("Prometheus did not observe the expected metrics within 50 seconds")


@pytest.mark.parametrize("path", ["/metrics", "/metrics/", "/metrics?x=1", "//metrics",
    "/%6detrics", "/metrics%2f", "/METRICS", "/metrics/extra", "/metrics;extra", "/x/../metrics"])
def test_nginx_blocks_metrics_variants(path):
    with pytest.raises(HTTPError) as error:
        urlopen(BASE + path, timeout=10)
    assert error.value.code == 403


def test_internal_scrape_and_reference_predictions():
    targets = wait_for(lambda: [t for t in get_json(PROM + "/api/v1/targets")["data"]["activeTargets"]
                               if t["labels"].get("job") == "fastapi" and t["health"] == "up"])
    assert targets[0]["scrapeUrl"] == "http://api:8000/metrics"
    assert float(query('model_ready{job="fastapi"}')[0]["value"][1]) == 1
    selector = 'http_requests_total{job="fastapi",route="/predict",method="POST",status_code="200"}'
    before = sum(float(s["value"][1]) for s in query(selector))
    request = Request(BASE + "/predict", data=json.dumps({"texts": [
        "Livraison rapide et produit conforme", "Commande jamais reçue", "Service client très réactif"
    ]}).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=130) as response:
        result = json.load(response)
    assert result["model_id"] == "c93cb37fac6cbebb9f3cce20cb962722c0190a5e588afffbb80f6512eebafab0"
    assert [p["cluster_id"] for p in result["predictions"]] == [6, 7, 6]
    wait_for(lambda: sum(float(s["value"][1]) for s in query(selector)) >= before + 1)
    observations = query('http_request_duration_seconds_count{job="fastapi",route="/predict",status_code="200"}')
    assert sum(float(s["value"][1]) for s in observations) > 0
    labels = get_json(PROM + "/api/v1/series?" + urlencode({"match[]": '{job="fastapi"}'}))["data"]
    allowed = {"__name__", "job", "instance", "method", "route", "status_code", "le"}
    assert all(set(series) <= allowed for series in labels)
