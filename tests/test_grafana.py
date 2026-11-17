"""Validate provisioned definitions against the metrics actually collected by the API."""
import json
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from api.metrics import Metrics


def test_datasource_and_provider():
    base = ROOT / "monitoring/grafana"
    source = yaml.safe_load((base / "provisioning/datasources/prometheus.yml").read_text())["datasources"][0]
    assert source["uid"] == "supply-chain-prometheus"
    assert source["type"] == "prometheus" and source["isDefault"] is True
    assert source["url"] == "http://prometheus:9090" and source["access"] == "proxy"
    assert source["editable"] is False and "secureJsonData" not in source
    provider = yaml.safe_load((base / "provisioning/dashboards/dashboards.yml").read_text())["providers"][0]
    assert provider["options"]["path"] == "/var/lib/grafana/dashboards"
    assert provider["allowUiUpdates"] is False


def test_dashboards_use_real_metrics_and_consistent_layout():
    names = {"up"}
    for family in Metrics(lambda: 0).registry.collect():
        names.add(family.name)
        if family.type == "counter":
            names.add(family.name + "_total")
        if family.type == "histogram":
            names.update(family.name + suffix for suffix in ("_bucket", "_count", "_sum"))
    dashboards = [json.loads(p.read_text()) for p in sorted((ROOT / "monitoring/grafana/dashboards").glob("*.json"))]
    assert {d["uid"]: d["title"] for d in dashboards} == {
        "supply-chain-api": "API Performance", "supply-chain-model": "Model Serving Health"}
    for dashboard in dashboards:
        assert dashboard["editable"] is False
        assert dashboard["time"] == {"from": "now-1h", "to": "now"}
        assert len({p["id"] for p in dashboard["panels"]}) == len(dashboard["panels"])
        occupied = set()
        for panel in dashboard["panels"]:
            assert panel["datasource"]["uid"] == "supply-chain-prometheus"
            assert panel["fieldConfig"]["defaults"]["noValue"] == "No samples"
            pos = panel["gridPos"]
            cells = {(x,y) for x in range(pos["x"],pos["x"]+pos["w"]) for y in range(pos["y"],pos["y"]+pos["h"])}
            assert pos["x"] + pos["w"] <= 24 and not cells & occupied
            occupied.update(cells)
            for target in panel["targets"]:
                assert target["datasource"]["uid"] == "supply-chain-prometheus"
                assert set(re.findall(r'([a-zA-Z_:][a-zA-Z0-9_:]*)\{', target["expr"])) <= names
                assert 'job="fastapi"' in target["expr"]
                assert "vector(0)" not in target["expr"]
            if "latency" in panel["title"].lower() or "duration" in panel["title"].lower():
                assert panel["fieldConfig"]["defaults"]["unit"] == "s"
                assert len(panel["targets"]) == 3
                assert "Not pure model" in panel["description"]


def test_grafana_service_security():
    services = yaml.safe_load((ROOT / "compose.yaml").read_text())["services"]
    grafana = services["grafana"]
    assert grafana["ports"] == ["127.0.0.1:3000:3000"]
    assert grafana["env_file"] == [".env.grafana.local"]
    assert grafana["environment"]["GF_AUTH_ANONYMOUS_ENABLED"] == "false"
    assert grafana["environment"]["GF_PUBLIC_DASHBOARDS_ENABLED"] == "false"
    assert "GF_SECURITY_ADMIN_PASSWORD" not in grafana["environment"]
    assert not services["api"].get("ports")
