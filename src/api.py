from fastapi import FastAPI, HTTPException, BackgroundTasks
import subprocess
from pydantic import BaseModel
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from contextlib import asynccontextmanager
from fastapi.responses import Response, FileResponse

# Illustration collect par le petit container PushGateway pour une alimentation à la demande
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from numpy import random
# ---------------------------------------------------------
# CONFIGURATION & CHARGEMENT (Allégé)
# ---------------------------------------------------------


# Chemins vers les artefacts
MODEL_PATH = "models/BERTopic"
KMEANS_PATH = MODEL_PATH + "_kmeans.pkl"
LABELS_PATH = MODEL_PATH + "_meta_labels.pkl"
METRICS_PATH = "metrics/silhouette.json"
ST_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

kmeans, meta_labels, embedding_model = None, None, None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ← startup : tout ce qui était dans on_event("startup")
    global kmeans,meta_labels,embedding_model
    # Chargement des artefacts au démarrage
    try:
        kmeans = joblib.load(KMEANS_PATH)
        meta_labels = joblib.load(LABELS_PATH)
        #embedding_model = SentenceTransformer(ST_MODEL_NAME)
        print("✅ API prête : Inférence directe via KMeans chargée.")
    except Exception as e:
        print(f"❌ Erreur de chargement des artefacts : {e}")

    yield              # ← l'API tourne ici

    # ← shutdown : tout ce qui était dans on_event("shutdown")
    print("🛑 Arrêt API")
    app.state.model = None

app = FastAPI(lifespan=lifespan,
    title="Oscaro Trustpilot API - Direct Meta-Clustering",
    description="API optimisée : Embedding -> KMeans Meta-Topic",
    version="1.3.0"
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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    response = FileResponse("app/static/trustp.png")#,media_type="image/png")
    response.headers["Cache-Control"] = "no-cache, no-store" 
    return response

@app.get("/", tags=["Toolbox"], summary="🏡 tester le home")
def home():
    return {"status": "online", "method": "Direct KMeans Inference"}

# Health check qui vérifie que le modèle est bien chargé
@app.get("/health", tags=["Toolbox"], summary="💖🩺 Santé du serveur")
def health():
    #tempo config health check switch if embedding_model is None:
    if kmeans is None:
        return Response(status_code=503, content="embedding_model not loaded")
    return {"status": "ok", "model": "loaded"}


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
                ["dvc", "repro"], 
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
        "message": "Le pipeline DVC a été lancé en arrière-plan. Vérifie les logs du serveur pour le suivi."
    }

# ---------------------------------------------------------
# ENDPOINTS DE MONITORING & MÉTRIQUES
# ---------------------------------------------------------

@app.get("/metrics", tags=["MLOps"], summary="🔎 Metriques du model")
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



