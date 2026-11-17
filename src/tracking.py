"""Configure local MLflow or DagsHub tracking without persisting credentials."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

ENV_FILE = ".env.dagshub.local"
TRACKING_VARIABLES = (
    "SUPPLY_CHAIN_TRACKING_URI", "MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD", "DAGSHUB_USER_TOKEN", "DAGSHUB_CLIENT_TOKEN",
)


def load_tracking_environment(env_file=None):
    """Read only tracking settings; never execute a local file or override exports.

    The absolute file path and loaded variables are inherited by DVC children.
    Secrets stay outside Git and DVC dependencies, including isolated workspaces.
    A missing file is allowed when tracking is configured through exports.
    """
    from dotenv import dotenv_values

    path = Path(os.environ.get("SUPPLY_CHAIN_ENV_FILE") or env_file or ENV_FILE).resolve()
    os.environ["SUPPLY_CHAIN_ENV_FILE"] = str(path)
    if path.is_file():
        values = dotenv_values(path, interpolate=False)
        for key in TRACKING_VARIABLES:
            if values.get(key) and not os.environ.get(key):
                os.environ[key] = values[key]
    return path


def configure_tracking(tracking_uri=None, dagshub_repo=None, *, env_file=None):
    """Resolve CLI/environment settings and return the configured URI, or None.

    An explicit URI takes precedence over environment settings. A repository
    shortcut and an explicit URI are mutually exclusive. Read local credentials
    first, while preserving explicitly exported environment settings.
    """
    load_tracking_environment(env_file)
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
    tracking_uri = (tracking_uri or os.environ.get("SUPPLY_CHAIN_TRACKING_URI")
                    or os.environ.get("MLFLOW_TRACKING_URI"))
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


def main(argv=None):
    """Check server access without creating experiments, runs or artifacts."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=ENV_FILE)
    parser.add_argument("--check", action="store_true", required=True,
                        help="Check read-only MLflow access without training.")
    args = parser.parse_args(argv)
    try:
        uri = configure_tracking(os.environ.get("SUPPLY_CHAIN_TRACKING_URI"), env_file=args.env_file)
        if not uri:
            parser.error("Set MLFLOW_TRACKING_URI in the local env file or shell.")
    except ValueError as error:
        parser.error(str(error))
    import mlflow
    try:
        mlflow.MlflowClient().search_experiments(max_results=1)
    except Exception as error:
        # Server responses can contain sensitive details: do not echo them.
        raise SystemExit(f"MLflow access check failed ({type(error).__name__}); check the URI, credentials and network.") from None
    print("MLflow access verified (read-only; no run created).")


if __name__ == "__main__":
    main()
