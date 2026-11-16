"""Run with: uvicorn api.main:app --app-dir src."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from experiment import PREPROCESSING, digest, read_json
from predict import ExperimentPredictor
from tracking import configure_tracking

logger = logging.getLogger(__name__)
UNAVAILABLE = "Selected model unavailable; check selection, tracking access and encoder cache."
Review = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=5000)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: list[Review] = Field(min_length=1, max_length=64)

    @field_validator("texts")
    @classmethod
    def reject_blank(cls, texts):
        if any(not text.strip() for text in texts):
            raise ValueError("Reviews must contain non-whitespace text.")
        return texts


class Prediction(BaseModel):
    cluster_id: int


class PredictionResponse(BaseModel):
    predictions: list[Prediction]
    model_id: str


class ModelInfo(BaseModel):
    model_id: str
    encoder_id: str
    encoder_revision: str
    method: str
    random_seed: int
    preprocessing_version: int


def load_selected_model(selection_path, destination):
    """Read artifacts only; never create runs or use the current DVC model."""
    selection = read_json(selection_path)
    config = selection["configuration"]
    if (selection["logical_id"] != digest(config)
            or config["preprocessing"] != PREPROCESSING
            or not re.fullmatch(r"[0-9a-f]{40}", config["encoder"]["revision"])
            or selection["final_seed"] != config["hyperparameters"]["random_state"]):
        raise ValueError("Invalid selected configuration.")
    # The recorded tracking store resolves runs:/ URIs. Credentials are loaded
    # by the existing helper, with exported values taking precedence.
    configure_tracking(selection["tracking_uri"])
    predictor = ExperimentPredictor.from_mlflow(selection["manifest_uri"], destination)
    manifest = predictor.manifest
    for key in ("logical_id", "configuration", "model_sha256", "model_uri"):
        if manifest[key] != selection[key]:
            raise ValueError("Selected artifact and manifest differ.")
    if manifest["mlflow_run_id"] != selection["train_mlflow_run_id"]:
        raise ValueError("Selected training run and manifest differ.")
    estimator_params = predictor.bundle["model"].get_params()
    if any(estimator_params.get(key) != value for key, value in config["hyperparameters"].items()):
        raise ValueError("Bundle hyperparameters differ from selection.")
    # Exercise the complete frozen path, including lazy pinned encoder loading,
    # preprocessing and dimension checks, before reporting readiness.
    predictor.predict_texts(["Initialisation du service"])
    return predictor


def create_app(*, selection_path=None, loader=None):
    """Factory with an injectable loader for offline tests."""
    selected_path = Path(selection_path or os.environ.get(
        "SUPPLY_CHAIN_SELECTION_PATH", "reports/experiment_selection/selection.json"))
    model_loader = loader or load_selected_model

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = None
        app.state.info = None
        with TemporaryDirectory(prefix="supply-chain-inference-") as destination:
            try:
                predictor = model_loader(selected_path, destination)
                config = predictor.manifest["configuration"]
                info = ModelInfo(
                    model_id=predictor.manifest["logical_id"],
                    encoder_id=config["encoder"]["id"],
                    encoder_revision=config["encoder"]["revision"],
                    method=config["method"],
                    random_seed=config["hyperparameters"]["random_state"],
                    preprocessing_version=config["preprocessing"]["version"],
                )
                app.state.info = info
                app.state.predictor = predictor
            except Exception:
                # Exceptions from remote services may contain tokens or paths.
                logger.error(UNAVAILABLE)
            try:
                yield
            finally:
                app.state.predictor = None
                app.state.info = None

    app = FastAPI(title="Selected review clustering model", lifespan=lifespan)
    app.state.predictor = None
    app.state.info = None
    prediction_lock = Lock()

    def require_model():
        if app.state.predictor is None:
            raise HTTPException(status_code=503, detail=UNAVAILABLE)
        return app.state.predictor

    @app.get("/health")
    def health():
        ready = app.state.predictor is not None
        return JSONResponse(status_code=200 if ready else 503, content={
            "status": "ok", "model_ready": ready,
            "model_status": "ready" if ready else "unavailable",
        })

    @app.get("/model/info", response_model=ModelInfo)
    def model_info():
        require_model()
        return app.state.info

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest):
        predictor = require_model()
        try:
            # Serialize access to the shared encoder/estimator in this process.
            with prediction_lock:
                labels = predictor.predict_texts(request.texts)
            if len(labels) != len(request.texts):
                raise ValueError("Prediction count mismatch.")
            return PredictionResponse(
                predictions=[Prediction(cluster_id=int(label)) for label in labels],
                model_id=app.state.info.model_id,
            )
        except Exception:
            logger.error("Selected model prediction failed.")
            raise HTTPException(status_code=500, detail="Prediction failed.") from None

    return app


app = create_app()
