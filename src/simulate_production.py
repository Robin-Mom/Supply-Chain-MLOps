"""Sequential host-side production replay; no model loading or training."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

MODEL_ID = "c93cb37fac6cbebb9f3cce20cb962722c0190a5e588afffbb80f6512eebafab0"
TRANSIENT = {408, 429, 500, 502, 503, 504}


class SimulationError(Exception):
    """Messages are fixed categories, never server responses or review contents."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, data):
    """Same-directory replace: a checkpoint is either complete or absent."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".checkpoint-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_rows(path, text_column="commentaire", id_column="", limit=None, shuffle_seed=None):
    """Validate every CSV row before selecting; never truncate texts or drop rows."""
    rows, seen = [], set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if (len(set(fields)) != len(fields) or text_column not in fields or id_column not in fields
                or text_column == id_column):
            raise SimulationError("invalid_csv_columns")
        for index, row in enumerate(reader):
            if None in row or any(value is None for value in row.values()):
                raise SimulationError("invalid_csv_row_shape")
            source_id, text = row[id_column], row[text_column]
            if not source_id.strip() or source_id in seen:
                raise SimulationError("missing_or_duplicate_source_id")
            if not text.strip() or len(text) > 5000:
                raise SimulationError("review_outside_api_limits")
            seen.add(source_id)
            rows.append({"source_row_id": source_id, "source_row_index": index, "text": text})
    if not rows:
        raise SimulationError("empty_csv")
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)
    return rows if limit is None else rows[:limit]


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def send_http(method, url, payload, timeout):
    """Do not forward reviews to redirects or retain error bodies."""
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, method=method,
                      headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with build_opener(NoRedirects()).open(request, timeout=timeout) as response:
            data = response.read(1024 * 1024 + 1)
            if len(data) > 1024 * 1024:
                return response.status, None
            try:
                return response.status, json.loads(data)
            except (ValueError, UnicodeError):
                return response.status, None
    except HTTPError as error:
        status = error.code
        error.close()
        return status, None


def validate_predictions(body, count, config):
    if not isinstance(body, dict):
        raise SimulationError("malformed_response")
    if body.get("model_id") != config["model_id"]:
        raise SimulationError("model_mismatch")
    predictions = body.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != count:
        raise SimulationError("malformed_response")
    labels = []
    for prediction in predictions:
        cluster = prediction.get("cluster_id") if isinstance(prediction, dict) else None
        if type(cluster) is not int or not 0 <= cluster < config["n_clusters"]:
            raise SimulationError("malformed_response")
        labels.append(cluster)
    return labels


def make_config(args):
    url = urlsplit(args.api_url)
    try:
        url.port
    except ValueError:
        raise SimulationError("invalid_api_url_no_credentials_allowed") from None
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username is not None
            or url.password is not None or url.query or url.fragment):
        raise SimulationError("invalid_api_url_no_credentials_allowed")
    if (not 1 <= args.batch_size <= 64 or args.limit is not None and args.limit < 1
            or not 0 <= args.retries <= 10 or args.n_clusters < 1
            or any(not math.isfinite(v) or v <= 0 for v in (args.timeout, args.backoff))
            or not math.isfinite(args.delay) or args.delay < 0
            or not re.fullmatch(r"[0-9a-f]{64}", args.model_id)):
        raise SimulationError("invalid_simulation_options")
    return {"schema_version": 1, "input_path": str(args.input.resolve()),
            "input_sha256": fingerprint(args.input), "text_column": args.text_column,
            "id_column": args.id_column, "limit": args.limit, "shuffle_seed": args.shuffle_seed,
            "batch_size": args.batch_size, "delay": args.delay, "timeout": args.timeout,
            "retries": args.retries, "backoff": args.backoff, "api_url": args.api_url.rstrip("/"),
            "model_id": args.model_id, "n_clusters": args.n_clusters}


@contextmanager
def run_lock(directory):
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SimulationError("simulation_already_running") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def perform_request(path, kind, batch, run_id, config, transport, sleep):
    """Checkpoint each completed attempt, then commit all rows of a batch together."""
    checkpoint = read_json(path) if path.exists() else {
        "request_id": path.stem, "run_id": run_id, "kind": kind, "state": "pending",
        "source_row_ids": [row["source_row_id"] for row in batch], "attempts": [], "records": []}
    labels = None
    while len(checkpoint["attempts"]) <= config["retries"]:
        if checkpoint["attempts"]:
            sleep(min(config["backoff"] * 2 ** (len(checkpoint["attempts"]) - 1), 60))
        timestamp, started = utc_now(), time.perf_counter()
        status, category = None, None
        transient = False
        try:
            status, body = transport("GET" if kind == "health" else "POST",
                config["api_url"] + ("/health" if kind == "health" else "/predict"),
                None if kind == "health" else {"texts": [row["text"] for row in batch]}, config["timeout"])
            if status != 200:
                category, transient = "http_error", status in TRANSIENT
            elif kind == "health":
                if not isinstance(body, dict) or body.get("model_ready") is not True:
                    category, transient = "model_not_ready", True
            else:
                labels = validate_predictions(body, len(batch), config)
        except SimulationError as error:
            category = str(error)
            # Parsing errors arise only after a successful HTTP response.
            if category == "malformed_response" and status is None:
                status = 200
        except (TimeoutError, URLError, ConnectionError, OSError, HTTPException) as error:
            category = "timeout" if isinstance(error, TimeoutError) or isinstance(getattr(error, "reason", None), TimeoutError) else "network_error"
            transient = True
        elapsed = time.perf_counter() - started
        checkpoint["attempts"].append({"timestamp": timestamp, "latency_seconds": elapsed,
                                        "http_status": status, "error_category": category})
        terminal = category is None or not transient or len(checkpoint["attempts"]) > config["retries"]
        if terminal:
            checkpoint["state"] = "complete"
            checkpoint["records"] = [{
                "run_id": run_id, "request_id": path.stem, "source_row_id": row["source_row_id"],
                "source_row_index": row["source_row_index"], "cluster_id": labels[i] if category is None else None,
                "model_id": config["model_id"], "request_timestamp": timestamp,
                "request_latency_seconds": elapsed, "success": category is None,
                "http_status": status, "error_category": category,
                "attempt_count": len(checkpoint["attempts"])} for i, row in enumerate(batch)]
        atomic_json(path, checkpoint)
        if terminal:
            break
    if checkpoint["state"] != "complete":
        raise SimulationError("invalid_pending_checkpoint")
    category = checkpoint["attempts"][-1]["error_category"]
    if kind == "health" and category:
        raise SimulationError("api_unavailable")
    if category in {"model_mismatch", "malformed_response"}:
        raise SimulationError(category)
    if category and transient:
        raise SimulationError("transient_retries_exhausted")
    return checkpoint


def summarize(directory, manifest, state):
    requests = [read_json(p) for p in sorted((directory / "requests").glob("*.json"))]
    prediction_requests = [r for r in requests if r["kind"] == "prediction"]
    records = [row for request in prediction_requests for row in request["records"]]
    attempted = {source_id for r in prediction_requests if r["attempts"] for source_id in r["source_row_ids"]}
    successful = [r for r in records if r["success"]]
    sessions = [read_json(p) for p in (directory / "sessions").glob("*.json")]
    elapsed = sum(s["elapsed_seconds"] for s in sessions)
    latencies = sorted(a["latency_seconds"] for r in prediction_requests for a in r["attempts"])
    def percentile(fraction):
        return latencies[max(0, math.ceil(fraction * len(latencies)) - 1)] if latencies else None
    if state == "completed" and len(records) != len(successful):
        state = "completed_with_errors"
    summary = {"run_id": manifest["run_id"], "state": state, "selected_reviews": manifest["selected_reviews"],
        "attempted_reviews": len(attempted), "successful_reviews": len(successful),
        "failed_reviews": len(records) - len(successful), "pending_reviews": len(attempted) - len(records),
        "unattempted_reviews": manifest["selected_reviews"] - len(attempted),
        "http_requests": sum(len(r["attempts"]) for r in requests),
        "prediction_http_requests": len(latencies),
        "health_http_requests": sum(len(r["attempts"]) for r in requests if r["kind"] == "health"),
        "total_elapsed_seconds": elapsed, "successful_reviews_per_second": len(successful) / elapsed if elapsed else None,
        "prediction_request_latency_seconds": {"count": len(latencies),
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": percentile(.5), "p95": percentile(.95), "max": max(latencies) if latencies else None},
        "cluster_distribution": dict(sorted(Counter(str(r["cluster_id"]) for r in successful).items()))}
    atomic_json(directory / "summary.json", summary)
    return summary


def simulate(args, *, transport=send_http, sleep=time.sleep):
    session_started = time.perf_counter()
    config = make_config(args)
    rows = load_rows(args.input, args.text_column, args.id_column, args.limit, args.shuffle_seed)
    if fingerprint(args.input) != config["input_sha256"]:
        raise SimulationError("input_changed_during_loading")
    order_hash = hashlib.sha256(json.dumps([r["source_row_id"] for r in rows]).encode()).hexdigest()
    batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
    if args.dry_run:
        return {"dry_run": True, "selected_reviews": len(rows), "batches": len(batches),
                "input_sha256": config["input_sha256"], "order_sha256": order_hash, "model_id": config["model_id"]}
    if args.resume:
        directory = args.resume
        manifest = read_json(directory / "run.json")
        if (manifest["configuration"] != config or manifest["order_sha256"] != order_hash
                or manifest["selected_reviews"] != len(rows)):
            raise SimulationError("incompatible_resume_configuration_or_input")
    else:
        run_id = args.run_id or uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
            raise SimulationError("invalid_run_id")
        args.output_root.mkdir(parents=True, exist_ok=True)
        directory = args.output_root / run_id
        directory.mkdir(mode=0o700, exist_ok=False)
        (directory / "requests").mkdir()
        (directory / "sessions").mkdir()
        manifest = {"run_id": run_id, "created_at": utc_now(), "configuration": config,
                    "order_sha256": order_hash, "selected_reviews": len(rows)}
        atomic_json(directory / "run.json", manifest)
    with run_lock(directory):
        # Never trust a checkpoint from a different batch/input/model.
        for path in (directory / "requests").glob("batch-*.json"):
            saved = read_json(path)
            index = int(path.stem.removeprefix("batch-"))
            if (not 0 <= index < len(batches) or saved["run_id"] != manifest["run_id"]
                    or saved["kind"] != "prediction" or saved["request_id"] != path.stem
                    or saved["state"] not in {"complete", "pending"}
                    or not 1 <= len(saved["attempts"]) <= config["retries"] + 1
                    or saved["source_row_ids"] != [r["source_row_id"] for r in batches[index]]
                    or saved["state"] == "pending" and (saved["records"] or len(saved["attempts"]) > config["retries"])
                    or saved["state"] == "complete" and len(saved["records"]) != len(batches[index])):
                raise SimulationError("incompatible_checkpoint")
            for record, row in zip(saved["records"], batches[index]):
                if (record["model_id"] != config["model_id"] or record["run_id"] != manifest["run_id"]
                        or record["request_id"] != path.stem or type(record["success"]) is not bool
                        or record["source_row_id"] != row["source_row_id"]
                        or record["source_row_index"] != row["source_row_index"]
                        or record["success"] and (type(record["cluster_id"]) is not int
                            or not 0 <= record["cluster_id"] < config["n_clusters"])):
                    raise SimulationError("incompatible_checkpoint")
        state = "interrupted"
        try:
            remaining = [i for i in range(len(batches)) if not (directory / "requests" / f"batch-{i:06d}.json").exists()
                         or read_json(directory / "requests" / f"batch-{i:06d}.json")["state"] != "complete"]
            if remaining:
                perform_request(directory / "requests" / f"health-{uuid4().hex}.json", "health", [],
                                manifest["run_id"], config, transport, sleep)
            for number, index in enumerate(remaining):
                if number:
                    sleep(config["delay"])
                perform_request(directory / "requests" / f"batch-{index:06d}.json", "prediction", batches[index],
                                manifest["run_id"], config, transport, sleep)
            state = "completed"
        except SimulationError as error:
            state = str(error)
            raise
        finally:
            atomic_json(directory / "sessions" / f"{uuid4().hex}.json", {
                "finished_at": utc_now(), "elapsed_seconds": time.perf_counter() - session_started, "state": state})
            summary = summarize(directory, manifest, state)
    return {"output_directory": str(directory), **summary}


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", type=Path, default=Path("data/processed/avis_8k_production.csv"))
    cli.add_argument("--api-url", default="http://127.0.0.1:8080")
    cli.add_argument("--text-column", default="commentaire")
    cli.add_argument("--id-column", default="", help="Empty name = original unnamed CSV index column.")
    cli.add_argument("--model-id", default=MODEL_ID)
    cli.add_argument("--n-clusters", type=int, default=10)
    cli.add_argument("--batch-size", type=int, default=16)
    cli.add_argument("--delay", type=float, default=1, help="Seconds between logical batches (sequential).")
    cli.add_argument("--timeout", type=float, default=60)
    cli.add_argument("--retries", type=int, default=3, help="Additional attempts after the first request, at most 10.")
    cli.add_argument("--backoff", type=float, default=1, help="Exponential retry base in seconds; capped at 60s.")
    cli.add_argument("--limit", type=int)
    cli.add_argument("--shuffle-seed", type=int)
    cli.add_argument("--output-root", type=Path, default=Path("reports/production_simulation"))
    cli.add_argument("--run-id")
    cli.add_argument("--resume", type=Path)
    cli.add_argument("--dry-run", action="store_true")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = simulate(args)
    except KeyboardInterrupt:
        print("Simulation interrupted; resume using the same configuration and run directory.")
        return 130
    except SimulationError as error:
        # Only fixed application categories are emitted, never transport details.
        known = {"invalid_csv_columns", "invalid_csv_row_shape", "missing_or_duplicate_source_id",
                 "review_outside_api_limits", "empty_csv", "invalid_api_url_no_credentials_allowed",
                 "invalid_simulation_options", "simulation_already_running", "incompatible_checkpoint",
                 "incompatible_resume_configuration_or_input", "invalid_run_id", "invalid_pending_checkpoint",
                 "model_mismatch", "malformed_response", "api_unavailable", "transient_retries_exhausted",
                 "input_changed_during_loading"}
        category = str(error) if str(error) in known else "simulation_error"
        print(f"Simulation stopped: {category}. Check input/options and the run summary if created.")
        return 1
    except (OSError, ValueError, KeyError, TypeError):
        # Never echo exceptions, CSV contents, URLs with secrets or HTTP bodies.
        print("Simulation stopped; check input/options and the run summary if created. No server error body is logged.")
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result.get("dry_run") or not result.get("failed_reviews") else 1


if __name__ == "__main__":
    raise SystemExit(main())
