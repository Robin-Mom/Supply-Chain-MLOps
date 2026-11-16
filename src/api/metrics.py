"""Bounded, per-application operational metrics; no inference data is collected."""
from time import perf_counter

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE", "CONNECT"})


class Metrics:
    def __init__(self, readiness):
        self.registry = CollectorRegistry()
        labels = ("method", "route", "status_code")
        self.requests = Counter("http_requests_total", "Completed HTTP requests, excluding scrapes.",
                                labels, registry=self.registry)
        self.duration = Histogram(
            "http_request_duration_seconds", "End-to-end HTTP duration, including CPU inference and queuing.",
            labels, buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
            registry=self.registry)
        # Route/status are not known until routing/response; keep this gauge by method only.
        self.in_progress = Gauge("http_requests_in_progress", "Active HTTP requests, excluding scrapes.",
                                 ("method",), registry=self.registry)
        self.ready = Gauge("model_ready", "Whether the selected predictor is initialized.", registry=self.registry)
        self.ready.set_function(readiness)


class HTTPMetricsMiddleware:
    """Pure ASGI middleware: preserve response bodies, errors and streaming behavior."""
    def __init__(self, app, metrics):
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"].rstrip("/") == "/metrics":
            return await self.app(scope, receive, send)
        method = scope["method"] if scope["method"] in METHODS else "OTHER"
        status = 500
        started = perf_counter()
        active = self.metrics.in_progress.labels(method)
        active.inc()

        async def measured_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, measured_send)
        finally:
            active.dec()
            # Read the route *after* the router ran; never label with scope['path'].
            route = getattr(scope.get("route"), "path", "__unmatched__")
            labels = (method, route, str(status))
            self.metrics.requests.labels(*labels).inc()
            self.metrics.duration.labels(*labels).observe(perf_counter() - started)
