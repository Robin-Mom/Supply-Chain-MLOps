#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import subprocess
import joblib
import dagshub
import mlflow
from sentence_transformers import SentenceTransformer
from mlflow.tracking import MlflowClient
import numpy as np
import os
import json
from contextlib import asynccontextmanager
from fastapi.responses import Response, FileResponse

# Illustration collect par le petit container PushGateway pour une alimentation à la demande
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from numpy import random
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

# Chemins vers les artefacts
MODEL_NAME = "trustpilot_bertopic_v4"
METRICS_PATH = "metrics/best_score.json"

# MODEL_PATH = "models/BERTopic"
# KMEANS_PATH = MODEL_PATH + "_kmeans.pkl"
# LABELS_PATH = MODEL_PATH + "_meta_labels.pkl"

# ST_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

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

    # latest is a model - consider assigning a type
    latest = versions[0]
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=latest.source)
    # LOAD CONFIG config.json
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
        print("✅ API prête : Inférence directe via KMeans chargée.")
    except Exception as e:
        print(f"❌ Erreur critique au démarrage : {e}")
    yield  # <- la boucle d'attente de l'api tourne dans ce yield
    print("🛑 Arrêt API")
    #ici on libère les ressources

app = FastAPI(lifespan=lifespan,
    title="Oscaro Trustpilot API",
    description="API optimisée : Embedding -> KMeans Meta-Topic -> MLFlow Registry -> nginx",
    version="2.0.1"
)


# ---------------------------------------------------------
# SCHEMAS pydantic
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
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    response = FileResponse("app/static/trustp.png")#,media_type="image/png")
    response.headers["Cache-Control"] = "no-cache, no-store" 
    return response

@app.get("/", tags=["Toolbox"], summary="🏡 Home et Context du model")
def home():
    return { "status": "online", 
        "method": "Direct KMeans Inference" ,
        "model_name": MODEL_NAME,
        "model_version": state["version"]
    }

# Health check qui vérifie que le modèle est bien chargé
@app.get("/health", tags=["Toolbox"], summary="💖🩺 Santé du serveur")
def health():
    if state["embedding_model"] is None:
        return Response(status_code=503, content="Model not loaded")
    return {"status": "healthy", "version": state["version"]}


@app.post("/predict0", response_model=PredictResponse,  tags=["Z_Test"], summary="⚙️ test area Prédiction du model")
async def predict_endpoint0(data: AvisInput):

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

@app.post("/train", tags=["MLOps"], summary="🚂🚃 Entrainement du model")
async def trigger_train(background_tasks: BackgroundTasks):
    """
    Lance le pipeline DVC (preprocess -> train -> evaluate) 
    en arrière-plan pour ne pas bloquer l'API.
    """
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
        "status": "Training started",
        "message": "DVC pipeline launched in background. Please check logs with follow up server."
    }



@app.get("/metrics", tags=["MLOps"], summary="🔎 Expose les métriques du dernier entraînement validé.")
def get_metrics():
    """Expose les métriques du dernier entraînement validé."""
    if not os.path.exists(METRICS_PATH):
        return {"error": "No metrics available"}
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


# 3. Nouvel endpoint pour Prometheus
@app.get("/metrics_prom", tags=["MLOps"],summary="🔎 endpoint prometheus pour generate_latest")
def get_metrics_prom():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)



@app.get("/metrics_pushgateway", tags=["Z_Test"], summary="🔎 test Metriques du model et test promotheus PushGateway")
def get_metrics_pushgateway():
    """Retourne le score de silhouette calculé lors de l'évaluation
       Et Push le score silhouette dans les metrics pour grafana via pushgateway prometheus
    """

    if not os.path.exists(METRICS_PATH):
        return {"error": "Metrics file not found. Run dvc repro first."}
    
    with open(METRICS_PATH, "r") as f:
        data = json.load(f)
        registry = CollectorRegistry()
        g = Gauge(
            "model_silhouette_score",
            "Silhouette score du modèle",
            registry=registry
        )
        g.set(data["silhouette_score"] + random.rand())  #a bit of random just for the sake of demo and avoid monotony

        push_to_gateway(
            "pushgateway:9091",   # ← service docker:port
            job="trainer",
            registry=registry
        )

        return data
