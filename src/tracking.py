"""Configure local MLflow or DagsHub tracking without persisting credentials."""
from __future__ import annotations

import os
from urllib.parse import urlsplit


def configure_tracking(tracking_uri=None, dagshub_repo=None):
    """Resolve CLI/environment settings and return the configured URI, or None.

    An explicit URI takes precedence over environment settings. A repository
    shortcut and an explicit URI are mutually exclusive.
    """
    if tracking_uri and dagshub_repo:
        raise ValueError("Use either --tracking-uri or --dagshub-repo.")
    if dagshub_repo:
        parts = dagshub_repo.split("/")
        if len(parts) != 2 or any(
            not part or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in part)
            or part in {".", ".."} for part in parts
        ):
            raise ValueError("--dagshub-repo must have the form OWNER/REPOSITORY.")
        tracking_uri = f"https://dagshub.com/{dagshub_repo}.mlflow"
    tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return None
    parsed = urlsplit(tracking_uri)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Supply tracking credentials through environment variables, not the URI.")
    if parsed.hostname == "dagshub.com":
        if parsed.scheme != "https":
            raise ValueError("DagsHub tracking requires HTTPS.")
        username = os.environ.get("MLFLOW_TRACKING_USERNAME")
        token = (os.environ.get("MLFLOW_TRACKING_PASSWORD")
                 or os.environ.get("DAGSHUB_USER_TOKEN")
                 or os.environ.get("DAGSHUB_CLIENT_TOKEN"))
        if not username or not token:
            raise ValueError("DagsHub requires MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD "
                             "(or DAGSHUB_USER_TOKEN / DAGSHUB_CLIENT_TOKEN).")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri
