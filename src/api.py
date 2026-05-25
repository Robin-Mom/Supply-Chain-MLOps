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
# ---------------------------------------------------------
# CONFIGURATION & CHARGEMENT
# ---------------------------------------------------------

# Chemins vers les artefacts
MODEL_NAME = "trustpilot_bertopic_v4"
METRICS_PATH = "metrics/best_score.json"

MODEL_PATH = "models/BERTopic"
KMEANS_PATH = MODEL_PATH + "_kmeans.pkl"
LABELS_PATH = MODEL_PATH + "_meta_labels.pkl"

ST_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# ---------------------------------------------------------
# GLOBAL VARIABLES
# ---------------------------------------------------------
kmeans, meta_labels, embedding_model, current_model_version,current_model_uri   = None, None, None, None, None

# ---------------------------------------------------------
# MLflow / DagsHub INIT
# ---------------------------------------------------------
token = os.getenv("DAGSHUB_USER_TOKEN")

if token is None:
    raise Exception("DAGSHUB_USER_TOKEN missing")

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

    versions = client.get_latest_versions( MODEL_NAME, stages=["Production"] )

    if len(versions) == 0:
        raise Exception( f"No Production model found for '{MODEL_NAME}'" )

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

    config_path = os.path.join(local_path,"config.json" )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    st_model_name = config["sentence_transformer"]

    # ---------------------------------------------------------
    # LOAD ARTIFACTS
    # ---------------------------------------------------------

    kmeans = joblib.load(os.path.join(local_path, "kmeans.pkl"))
    meta_labels = joblib.load(os.path.join(local_path, "meta_labels.pkl"))
    embedding_model = SentenceTransformer(st_model_name)

    current_model_version = model_version
    current_model_uri = artifact_uri

    print("✅ Production model successfully loaded")


# ---------------------------------------------------------
# STARTUP EVENT - via asynccontextmanager
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ← startup : tout ce qui était dans on_event("startup")
    global kmeans,meta_labels,embedding_model
    # Chargement des artefacts au démarrage
    try:
        load_latest_production_model()
        print("✅ API prête : Inférence directe via KMeans chargée.")
    except Exception as e:
        print(f"❌ Erreur de chargement des artefacts : {e}")

    yield              # ← l'API tourne ici

    # ← shutdown : tout ce qui était dans on_event("shutdown")
    print("🛑 Arrêt API")
    app.state.model = None

app = FastAPI(lifespan=lifespan,
    title="Oscaro Trustpilot API - Direct Meta-Clustering",
    description="API optimisée : Embedding -> KMeans Meta-Topic -> MLFlow Registry -> nginx",
    version="2.0.1"
)

# ---------------------------------------------------------
# SCHÉMAS DE DONNÉES
# ---------------------------------------------------------

class AvisInput(BaseModel):
    commentaire: str

class PredictResponse(BaseModel):
    text: str
    meta_topic: int
    meta_label: str
    model_version: str

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    response = FileResponse("app/static/trustp.png")#,media_type="image/png")
    response.headers["Cache-Control"] = "no-cache, no-store" 
    return response

@app.get("/", tags=["Toolbox"], summary="🏡 Home et Context du model")
def home():
    return {
        "status": "online",
        "model_name": MODEL_NAME,
        "model_version": current_model_version
    }

# Health check qui vérifie que le modèle est bien chargé
@app.get("/health", tags=["Toolbox"], summary="💖🩺 Santé du serveur")
def health():
    #tempo config health check switch if embedding_model is None:
    if embedding_model is None:
        return Response(status_code=503, content="embedding_model not loaded")
    return {"status": "healthy", "model": "loaded", "model_version": current_model_version}


# ---------------------------------------------------------
# ENDPOINTS DE PREDICTION
# ---------------------------------------------------------

@app.post("/predict", response_model=PredictResponse,  tags=["MLOps"], summary="🚀 Prédiction du model")
async def predict_endpoint(data: AvisInput):
    """
    Reproduction exacte de la logique de predict.py
    """
    try:
        if not data.commentaire.strip():
            raise HTTPException(status_code=400, detail="Le commentaire est vide")

        # 1. Embedding (Vectorisation du texte)
        embedding = embedding_model.encode([data.commentaire])

        # 2. Prédiction directe via KMeans (Meta-clustering)
        # On projette le point directement dans les 10 clusters de Robin
        meta_topic = int(kmeans.predict(embedding)[0])

        # 3. Récupération du label
        if meta_topic in meta_labels:
            label = " | ".join(meta_labels[meta_topic])
        else:
            label = "unknown"

        return {
            "text": data.commentaire,
            "meta_topic": meta_topic,
            "meta_label": label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict",  response_model=PredictResponse,  tags=["MLOps"], summary="🚀 Prédiction du model NEW")
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
# RELOAD MODEL
# ---------------------------------------------------------
@app.post("/reload_model", tags=["MLOps"], summary="🚚 Rechargement du model")
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
# ENDPOINTS D'ENTRAÎNEMENT (DVC)
# ---------------------------------------------------------

@app.post("/train", tags=["MLOps"], summary="🚂🚃 Entrainement du model")
async def trigger_train(background_tasks: BackgroundTasks):
    """
    Lance le pipeline DVC (preprocess -> train -> evaluate) 
    en arrière-plan pour ne pas bloquer l'API.
    """
    def run_dvc():
        try:
            # On lance dvc repro comme tu le ferais dans le terminal
            result = subprocess.run(
                ["dvc", "repro", "--force"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            print(f"✅ DVC repro terminé avec succès :\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur lors de DVC repro :\n{e.stderr}")

    # On lance la fonction dans une tâche de fond
    background_tasks.add_task(run_dvc)
    
    return {
        "status": "Training started",
        "message": "DVC pipeline launched in background. Please check logs with follow up server."
    }

# ---------------------------------------------------------
# ENDPOINTS DE MONITORING & MÉTRIQUES
# ---------------------------------------------------------

@app.get("/metrics", tags=["MLOps"], summary="🔎 Metriques du model et test promotheus PushGateway")
def get_metrics():
    """Retourne le score de silhouette calculé lors de l'évaluation
       Et Push le score silhouette dans les metrics pour grafan via pushgateway prometheus
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
