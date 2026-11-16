"""Offline replay tests: synthetic rows and controlled HTTP transport only."""
import csv
import json
from io import BytesIO
from pathlib import Path
import sys
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import simulate_production as sim


@pytest.fixture
def args(tmp_path):
    path = tmp_path / "production.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "commentaire"])
        writer.writerows((str(i), f"private-review-sentinel-{i}") for i in range(5))
    return sim.parser().parse_args(["--input", str(path), "--output-root", str(tmp_path / "outputs"),
                                    "--run-id", "test", "--batch-size", "2"])


def healthy(method, url, payload, timeout):
    if method == "GET":
        return 200, {"model_ready": True}
    return 200, {"model_id": sim.MODEL_ID, "predictions": [{"cluster_id": 6} for _ in payload["texts"]]}


def records(directory):
    return [row for path in sorted((directory / "requests").glob("batch-*.json"))
            for row in sim.read_json(path)["records"]]


def test_batches_output_and_summary(args):
    transport, sleep = Mock(side_effect=healthy), Mock()
    result = sim.simulate(args, transport=transport, sleep=sleep)
    assert [len(c.args[2]["texts"]) for c in transport.call_args_list if c.args[0] == "POST"] == [2, 2, 1]
    assert transport.call_args_list[0].args == ("GET", "http://127.0.0.1:8080/health", None, 60)
    assert transport.call_args_list[1].args[2] == {"texts": ["private-review-sentinel-0", "private-review-sentinel-1"]}
    directory = Path(result["output_directory"])
    saved = records(directory)
    assert [r["source_row_id"] for r in saved] == list(map(str, range(5)))
    assert [r["source_row_index"] for r in saved] == list(range(5))
    assert all(r["success"] and r["cluster_id"] == 6 and r["model_id"] == sim.MODEL_ID for r in saved)
    assert all(r["request_latency_seconds"] >= 0 and r["http_status"] == 200 for r in saved)
    assert saved[0]["request_id"] == saved[1]["request_id"]
    assert saved[0]["request_latency_seconds"] == saved[1]["request_latency_seconds"]
    assert result["attempted_reviews"] == result["successful_reviews"] == 5
    assert result["failed_reviews"] == result["pending_reviews"] == result["unattempted_reviews"] == 0
    assert result["http_requests"] == 4 and result["prediction_http_requests"] == 3
    assert result["cluster_distribution"] == {"6": 5}
    assert result["prediction_request_latency_seconds"]["count"] == 3
    assert sleep.call_count == 2
    assert "private-review-sentinel" not in "".join(p.read_text() for p in directory.rglob("*.json"))


def test_deterministic_subset_and_dry_run(args):
    args.limit, args.shuffle_seed, args.dry_run = 3, 21, True
    first = sim.load_rows(args.input, limit=3, shuffle_seed=21)
    assert first == sim.load_rows(args.input, limit=3, shuffle_seed=21)
    assert first != sim.load_rows(args.input, limit=3, shuffle_seed=22)
    transport = Mock()
    result = sim.simulate(args, transport=transport)
    assert result["selected_reviews"] == 3 and result["batches"] == 2
    transport.assert_not_called()
    assert not args.output_root.exists()


@pytest.mark.parametrize("content", [",other\n1,foo\n", ",commentaire\n1, \n",
    ",commentaire\n1,a\n1,b\n", ",commentaire\n,a\n", ",commentaire\n", ",commentaire\n1\n",
    ",commentaire\n1,a,extra\n", ",commentaire\n1," + "a" * 5001 + "\n"])
def test_invalid_csv(args, content):
    args.input.write_text(content)
    with pytest.raises(sim.SimulationError):
        sim.load_rows(args.input)


@pytest.mark.parametrize("failure", [TimeoutError("private-token"), URLError("private-token"), 503, 429])
def test_retry_backoff_and_http_accounting(args, failure):
    args.limit, args.retries = 1, 2
    calls = []
    def transport(method, url, payload, timeout):
        if method == "GET":
            return healthy(method, url, payload, timeout)
        calls.append(payload)
        if len(calls) < 3:
            if isinstance(failure, Exception):
                raise failure
            return failure, {"detail": "private-token"}
        return healthy(method, url, payload, timeout)
    sleep = Mock()
    result = sim.simulate(args, transport=transport, sleep=sleep)
    assert [c.args[0] for c in sleep.call_args_list] == [1, 2]
    assert result["prediction_http_requests"] == 3 and result["successful_reviews"] == 1
    assert records(Path(result["output_directory"]))[0]["attempt_count"] == 3


def test_validation_errors_not_retried(args):
    def transport(method, url, payload, timeout):
        return (422, {"detail": "private-review-sentinel"}) if method == "POST" else healthy(method,url,payload,timeout)
    result = sim.simulate(args, transport=Mock(side_effect=transport), sleep=Mock())
    assert result["prediction_http_requests"] == 3
    assert result["failed_reviews"] == 5 and result["successful_reviews"] == 0
    assert all(r["error_category"] == "http_error" and r["http_status"] == 422 for r in records(Path(result["output_directory"])))
    args.resume = Path(result["output_directory"])
    transport = Mock()
    resumed = sim.simulate(args, transport=transport)
    transport.assert_not_called()
    assert resumed["failed_reviews"] == 5


@pytest.mark.parametrize("body, category", [
    ({"model_id": "wrong-private-token", "predictions": []}, "model_mismatch"),
    ({"model_id": sim.MODEL_ID, "predictions": []}, "malformed_response"),
    ({"model_id": sim.MODEL_ID, "predictions": [{"cluster_id": True}] * 2}, "malformed_response"),
    ({"model_id": sim.MODEL_ID, "predictions": [{"cluster_id": 10}] * 2}, "malformed_response"),
    (None, "malformed_response")])
def test_bad_responses_stop_without_switching_models(args, body, category):
    transport = Mock(side_effect=lambda method,url,payload,timeout: (200, body) if method == "POST" else (200, {"model_ready": True}))
    with pytest.raises(sim.SimulationError, match=category):
        sim.simulate(args, transport=transport, sleep=Mock())
    summary = sim.read_json(args.output_root / "test/summary.json")
    assert summary["failed_reviews"] == 2 and summary["unattempted_reviews"] == 3
    assert transport.call_count == 2
    assert "wrong-private-token" not in "".join(p.read_text() for p in args.output_root.rglob("*.json"))


def test_unavailable_preflight_sends_no_reviews(args):
    transport = Mock(return_value=(503, None))
    with pytest.raises(sim.SimulationError, match="api_unavailable"):
        sim.simulate(args, transport=transport, sleep=Mock())
    assert all(c.args[0] == "GET" for c in transport.call_args_list)
    assert transport.call_count == 4
    summary = sim.read_json(args.output_root / "test/summary.json")
    assert summary["attempted_reviews"] == 0 and summary["health_http_requests"] == 4


def test_resume_skips_successes_and_validates_input(args):
    posts = 0
    def interrupted(method,url,payload,timeout):
        nonlocal posts
        if method == "POST":
            posts += 1
            if posts == 2:
                raise KeyboardInterrupt
        return healthy(method,url,payload,timeout)
    with pytest.raises(KeyboardInterrupt):
        sim.simulate(args, transport=interrupted, sleep=Mock())
    args.resume = args.output_root / "test"
    before = (args.resume / "requests/batch-000000.json").read_bytes()
    transport = Mock(side_effect=healthy)
    result = sim.simulate(args, transport=transport, sleep=Mock())
    assert (args.resume / "requests/batch-000000.json").read_bytes() == before
    assert result["successful_reviews"] == 5
    assert len({r["source_row_id"] for r in records(args.resume)}) == 5
    assert transport.call_count == 3
    transport.reset_mock()
    sim.simulate(args, transport=transport, sleep=Mock())
    transport.assert_not_called()
    args.input.write_text(args.input.read_text().replace("sentinel-0", "changed-0"))
    with pytest.raises(sim.SimulationError, match="incompatible_resume"):
        sim.simulate(args, transport=transport)


def test_resume_pending_retry_preserves_attempt_budget(args):
    def temporary(method,url,payload,timeout):
        return (503, None) if method == "POST" else healthy(method,url,payload,timeout)
    with pytest.raises(KeyboardInterrupt):
        sim.simulate(args, transport=temporary, sleep=Mock(side_effect=KeyboardInterrupt))
    args.resume = args.output_root / "test"
    result = sim.simulate(args, transport=healthy, sleep=Mock())
    assert result["successful_reviews"] == 5
    assert records(args.resume)[0]["attempt_count"] == 2


def test_resume_rejects_corrupted_checkpoint(args):
    sim.simulate(args, transport=healthy, sleep=Mock())
    args.resume = args.output_root / "test"
    path = args.resume / "requests/batch-000000.json"
    checkpoint = sim.read_json(path)
    checkpoint["records"] = []
    sim.atomic_json(path, checkpoint)
    with pytest.raises(sim.SimulationError, match="incompatible_checkpoint"):
        sim.simulate(args, transport=Mock())


def test_no_overwrite_and_no_sensitive_cli_logs(args, capsys, monkeypatch):
    sim.simulate(args, transport=healthy, sleep=Mock())
    with pytest.raises(FileExistsError):
        sim.simulate(args, transport=Mock())
    monkeypatch.setattr(sim, "simulate", Mock(side_effect=URLError("private-token private-review")))
    assert sim.main([]) == 1
    output = capsys.readouterr().out
    assert "private-token" not in output and "private-review" not in output


@pytest.mark.parametrize("field,value", [("api_url", "http://user:secret@localhost:8080"),
    ("api_url", "http://localhost:8080?token=secret"), ("batch_size",65), ("limit",0),
    ("delay",-1), ("timeout",float("nan")), ("retries",11)])
def test_bad_configuration(args, field, value):
    setattr(args, field, value)
    with pytest.raises(sim.SimulationError):
        sim.make_config(args)


@pytest.mark.parametrize("field,value", [("model_id", "b" * 64), ("batch_size", 1), ("shuffle_seed", 7)])
def test_resume_rejects_changed_model_or_plan(args, field, value):
    sim.simulate(args, transport=healthy, sleep=Mock())
    args.resume = args.output_root / "test"
    setattr(args, field, value)
    transport = Mock()
    with pytest.raises(sim.SimulationError, match="incompatible_resume"):
        sim.simulate(args, transport=transport)
    transport.assert_not_called()


def test_real_http_adapter_encoding_and_error_sanitization(monkeypatch):
    response = BytesIO(json.dumps({"model_id": sim.MODEL_ID, "predictions": [{"cluster_id": 6}]}).encode())
    response.status = 200
    opener = Mock()
    opener.open.return_value = response
    monkeypatch.setattr(sim, "build_opener", Mock(return_value=opener))
    status, body = sim.send_http("POST", "http://127.0.0.1:8080/predict", {"texts": ["avis synthétique"]}, 3)
    request = opener.open.call_args.args[0]
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data) == {"texts": ["avis synthétique"]}
    assert opener.open.call_args.kwargs["timeout"] == 3
    assert status == 200 and body["model_id"] == sim.MODEL_ID
    error_body = Mock()
    opener.open.side_effect = HTTPError(request.full_url, 422, "private-token", {}, error_body)
    assert sim.send_http("POST", request.full_url, {"texts": ["private-review"]}, 3) == (422, None)
    error_body.read.assert_not_called()
    error_body.close.assert_called_once()


def test_prediction_outage_stops_after_bounded_attempts(args):
    def transport(method,url,payload,timeout):
        return (503, None) if method == "POST" else healthy(method,url,payload,timeout)
    with pytest.raises(sim.SimulationError, match="transient_retries_exhausted"):
        sim.simulate(args, transport=transport, sleep=Mock())
    summary = sim.read_json(args.output_root / "test/summary.json")
    assert summary["prediction_http_requests"] == 4 and summary["failed_reviews"] == 2
    assert summary["unattempted_reviews"] == 3
