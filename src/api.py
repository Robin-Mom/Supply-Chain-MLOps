#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import subprocess
import joblib
import dagshub
import mlflow
import numpy as np
import os
import json
from contextlib import asynccontextmanager
from fastapi.responses import Response, FileResponse
from sentence_transformers import SentenceTransformer
from mlflow.tracking import MlflowClient

from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST


# 1. Déclaration du compteur pour l'histogramme
PREDICTION_COUNTER = Counter(
    'model_predictions_total', 
    'Nombre de prédictions par topic', 
    ['topic_id', 'meta_label']
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MODEL_NAME = "trustpilot_bertopic_v4"
METRICS_PATH = "metrics/best_score.json"

# Variables globales gérées par le cycle de vie de l'app
state = {
    "kmeans": None,
    "meta_labels": None,
    "embedding_model": None,
    "version": None,
    "uri": None
}

# ---------------------------------------------------------
# INITIALISATION MLFLOW
# ---------------------------------------------------------
if os.getenv("DAGSHUB_USER_TOKEN") is None:
    raise Exception("DAGSHUB_USER_TOKEN missing")

dagshub.init(repo_owner='schmilblick-ai', repo_name='Supply-Chain-MLOps', mlflow=True)

# ---------------------------------------------------------
# LOGIQUE DE CHARGEMENT
# ---------------------------------------------------------
def load_latest_production_model():
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    
    if not versions:
        raise Exception(f"No Production model found for '{MODEL_NAME}'")

    latest = versions[0]
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=latest.source)

    with open(os.path.join(local_path, "config.json"), "r") as f:
        config = json.load(f)

    state["kmeans"] = joblib.load(os.path.join(local_path, "kmeans.pkl"))
    state["meta_labels"] = joblib.load(os.path.join(local_path, "meta_labels.pkl"))
    state["embedding_model"] = SentenceTransformer(config["sentence_transformer"])
    state["version"] = latest.version
    state["uri"] = latest.source
    print(f"✅ Model v{latest.version} chargé.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_latest_production_model()
    except Exception as e:
        print(f"❌ Erreur critique au démarrage : {e}")
    yield
    print("🛑 Arrêt API")

app = FastAPI(lifespan=lifespan, title="Oscaro Trustpilot API")

# ---------------------------------------------------------
# SCHÉMAS
# ---------------------------------------------------------
class AvisInput(BaseModel):
    commentaire: str

class PredictResponse(BaseModel):
    text: str
    meta_topic: int
    meta_label: str
    model_version: str

# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------
@app.get("/health", tags=["Toolbox"])
def health():
    if state["embedding_model"] is None:
        return Response(status_code=503, content="Model not loaded")
    return {"status": "healthy", "version": state["version"]}

@app.post("/predict", response_model=PredictResponse, tags=["MLOps"])
async def predict(data: AvisInput):
    if state["embedding_model"] is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not data.commentaire.strip():
        raise HTTPException(status_code=400, detail="Empty comment")

    embedding = state["embedding_model"].encode([data.commentaire])
    
    meta_topic = int(state["kmeans"].predict(embedding)[0])
    label = " | ".join(state["meta_labels"][meta_topic]) if meta_topic in state["meta_labels"] else "unknown"

    # Incrémenter ici
    PREDICTION_COUNTER.labels(topic_id=str(meta_topic), meta_label=label).inc()

    return {
        "text": data.commentaire,
        "meta_topic": meta_topic,
        "meta_label": label,
        "model_version": str(state["version"])
    }

@app.post("/reload_model", tags=["MLOps"])
async def reload_model():
    try:
        load_latest_production_model()
        return {"status": "success", "version": state["version"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", tags=["MLOps"])
async def trigger_train(background_tasks: BackgroundTasks):
    def run_dvc():
        subprocess.run(["dvc", "repro", "--force"], check=True)
    background_tasks.add_task(run_dvc)
    return {"status": "Training started"}

@app.get("/metrics", tags=["MLOps"])
def get_metrics():
    """Expose les métriques du dernier entraînement validé."""
    if not os.path.exists(METRICS_PATH):
        return {"error": "No metrics available"}
    with open(METRICS_PATH, "r") as f:
        return json.load(f)
    
# 3. Nouvel endpoint pour Prometheus
@app.get("/metrics_prom", tags=["MLOps"])
def get_metrics_prom():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)