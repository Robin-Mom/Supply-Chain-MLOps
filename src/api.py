#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

import subprocess
import joblib
import dagshub
import mlflow
import json
import os
import numpy as np

from sentence_transformers import SentenceTransformer
from mlflow.tracking import MlflowClient


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

MODEL_NAME = "trustpilot_bertopic"

METRICS_PATH = "metrics/silhouette.json"

app = FastAPI(
    title="Oscaro Trustpilot API",
    description="Production API using MLflow Registry",
    version="2.0.0"
)


# ---------------------------------------------------------
# GLOBAL VARIABLES
# ---------------------------------------------------------

kmeans = None
meta_labels = None
embedding_model = None
current_model_version = None
current_model_uri = None


# ---------------------------------------------------------
# MLflow / DagsHub INIT
# ---------------------------------------------------------

token = os.getenv("DAGSHUB_USER_TOKEN")

if token is None:
    raise Exception(
        "DAGSHUB_USER_TOKEN missing"
    )

dagshub.init(
    repo_owner='schmilblick-ai',
    repo_name='Supply-Chain-MLOps',
    mlflow=True
)


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

def load_latest_production_model():

    global kmeans
    global meta_labels
    global embedding_model
    global current_model_version
    global current_model_uri

    client = MlflowClient()

    versions = client.get_latest_versions(
        MODEL_NAME,
        stages=["Production"]
    )

    if len(versions) == 0:
        raise Exception(
            f"No Production model found for '{MODEL_NAME}'"
        )

    latest_model = versions[0]

    model_version = latest_model.version
    artifact_uri = latest_model.source

    print(f"Loading Production model v{model_version}")
    print(f"Artifact URI: {artifact_uri}")

    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=artifact_uri
    )

    # ---------------------------------------------------------
    # LOAD CONFIG
    # ---------------------------------------------------------

    config_path = os.path.join(
        local_path,
        "config.json"
    )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    st_model_name = config["sentence_transformer"]

    # ---------------------------------------------------------
    # LOAD ARTIFACTS
    # ---------------------------------------------------------

    kmeans = joblib.load(
        os.path.join(local_path, "kmeans.pkl")
    )

    meta_labels = joblib.load(
        os.path.join(local_path, "meta_labels.pkl")
    )

    embedding_model = SentenceTransformer(
        st_model_name
    )

    current_model_version = model_version
    current_model_uri = artifact_uri

    print("✅ Production model successfully loaded")


# ---------------------------------------------------------
# STARTUP EVENT
# ---------------------------------------------------------

@app.on_event("startup")
async def startup_event():

    try:
        load_latest_production_model()

    except Exception as e:

        print(f"❌ Error loading production model: {e}")


# ---------------------------------------------------------
# DATA SCHEMAS
# ---------------------------------------------------------

class AvisInput(BaseModel):
    commentaire: str


class PredictResponse(BaseModel):
    text: str
    meta_topic: int
    meta_label: str
    model_version: str


# ---------------------------------------------------------
# ROOT
# ---------------------------------------------------------

@app.get("/")
def home():

    return {
        "status": "online",
        "model_name": MODEL_NAME,
        "model_version": current_model_version
    }


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictResponse
)
async def predict_endpoint(data: AvisInput):

    try:

        if embedding_model is None:
            raise HTTPException(
                status_code=500,
                detail="No production model loaded"
            )

        if not data.commentaire.strip():
            raise HTTPException(
                status_code=400,
                detail="Empty comment"
            )

        # ---------------------------------------------------------
        # EMBEDDING
        # ---------------------------------------------------------

        embedding = embedding_model.encode(
            [data.commentaire]
        )

        # ---------------------------------------------------------
        # META-CLUSTER PREDICTION
        # ---------------------------------------------------------

        meta_topic = int(
            kmeans.predict(embedding)[0]
        )

        # ---------------------------------------------------------
        # LABEL
        # ---------------------------------------------------------

        if meta_topic in meta_labels:

            label = " | ".join(
                meta_labels[meta_topic]
            )

        else:

            label = "unknown"

        return {
            "text": data.commentaire,
            "meta_topic": meta_topic,
            "meta_label": label,
            "model_version": str(current_model_version)
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

@app.get("/metrics")
def get_metrics():

    if not os.path.exists(METRICS_PATH):

        return {
            "error": (
                "Metrics file not found."
            )
        }

    with open(METRICS_PATH, "r") as f:

        return json.load(f)


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------

@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "model_version": current_model_version
    }


# ---------------------------------------------------------
# RELOAD MODEL
# ---------------------------------------------------------

@app.post("/reload_model")
async def reload_model():

    try:

        load_latest_production_model()

        return {
            "status": "success",
            "model_version": current_model_version
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------------
# DVC TRAINING
# ---------------------------------------------------------

@app.post("/train", tags=["MLOps"])
async def trigger_train(
    background_tasks: BackgroundTasks
):

    def run_dvc():

        try:

            result = subprocess.run(
                ["dvc", "repro", "--force"],
                capture_output=True,
                text=True,
                check=True
            )

            print(
                f"✅ DVC repro success:\n"
                f"{result.stdout}"
            )

        except subprocess.CalledProcessError as e:

            print(
                f"❌ DVC repro error:\n"
                f"{e.stderr}"
            )

    background_tasks.add_task(run_dvc)

    return {
        "status": "training_started",
        "message": (
            "DVC pipeline launched "
            "in background"
        )
    }
