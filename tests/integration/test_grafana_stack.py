"""Opt-in real Grafana API checks. This validates query data, not browser rendering."""
import base64
import json
import math
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dotenv import dotenv_values
import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:3000"
pytestmark = pytest.mark.skipif(os.environ.get("RUN_GRAFANA_INTEGRATION") != "1",
                                reason="Requires the explicitly started Grafana stack")


def grafana(path, data=None, method=None):
    values = dotenv_values(ROOT / ".env.grafana.local", interpolate=False)
    username = os.environ.get("GF_SECURITY_ADMIN_USER") or values["GF_SECURITY_ADMIN_USER"]
    password = os.environ.get("GF_SECURITY_ADMIN_PASSWORD") or values["GF_SECURITY_ADMIN_PASSWORD"]
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    request = Request(BASE + path, data=None if data is None else json.dumps(data).encode(),
                      headers={"Authorization": "Basic " + token, "Content-Type": "application/json"}, method=method)
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def test_provisioned_datasource_and_dashboards():
    assert grafana("/api/health")["database"] == "ok"
    source = grafana("/api/datasources/uid/supply-chain-prometheus")
    assert source["isDefault"] is True and source["url"] == "http://prometheus:9090"
    assert source["readOnly"] is True
    assert grafana("/api/datasources/uid/supply-chain-prometheus/health")["status"] == "OK"
    for uid, title in (("supply-chain-api", "API Performance"), ("supply-chain-model", "Model Serving Health")):
        result = grafana("/api/dashboards/uid/" + uid)
        assert result["dashboard"]["title"] == title
        assert result["meta"]["provisioned"] is True
    with pytest.raises(HTTPError) as error:
        urlopen(BASE + "/api/search", timeout=10)
    assert error.value.code == 401


def reference_request():
    body = {"texts": ["Livraison rapide et produit conforme", "Commande jamais reçue", "Service client très réactif"]}
    request = Request("http://127.0.0.1:8080/predict", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=130) as response:
        result = json.load(response)
    assert [p["cluster_id"] for p in result["predictions"]] == [6, 7, 6]
    assert result["model_id"] == "c93cb37fac6cbebb9f3cce20cb962722c0190a5e588afffbb80f6512eebafab0"


def test_all_panel_queries_return_real_data():
    # Repeat only the three explicit reference reviews as validation traffic.
    # Allow multiple 15-second scrapes so rate/quantile queries have observations.
    queries = []
    for path in sorted((ROOT / "monitoring/grafana/dashboards").glob("*.json")):
        for panel in json.loads(path.read_text())["panels"]:
            for target in panel["targets"]:
                queries.append({**target, "expr": target["expr"].replace("$__rate_interval", "1m"),
                                "refId": "Q" + str(len(queries)), "intervalMs": 15000, "maxDataPoints": 300})
    deadline = time.monotonic() + 80
    missing = []
    while time.monotonic() < deadline:
        reference_request()
        now = int(time.time() * 1000)
        results = grafana("/api/ds/query", {"from": str(now - 300000), "to": str(now), "queries": queries})["results"]
        missing = []
        for query in queries:
            result = results[query["refId"]]
            assert not result.get("error"), query["refId"]
            numeric = [value for frame in result.get("frames", [])
                       for field, values in zip(frame["schema"]["fields"], frame["data"]["values"])
                       if field["type"] == "number" for value in values
                       if isinstance(value, (int, float)) and math.isfinite(value)]
            if not numeric:
                missing.append(query["refId"])
        if not missing:
            return
        time.sleep(5)
    raise AssertionError(f"Panel queries without collected numeric samples: {missing}")
