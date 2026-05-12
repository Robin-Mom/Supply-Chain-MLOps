from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import subprocess
from fastapi import BackgroundTasks

# ---------------------------------------------------------
# CONFIGURATION & CHARGEMENT (Allégé)
# ---------------------------------------------------------

app = FastAPI(
    title="Oscaro Trustpilot API - Direct Meta-Clustering",
    description="API optimisée : Embedding -> KMeans Meta-Topic",
    version="1.3.0"
)

# Chemins vers les artefacts
MODEL_PATH = "models/BERTopic"
KMEANS_PATH = MODEL_PATH + "_kmeans.pkl"
LABELS_PATH = MODEL_PATH + "_meta_labels.pkl"
METRICS_PATH = "metrics/silhouette.json"
ST_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


# Chargement des artefacts au démarrage
try:
    kmeans = joblib.load(KMEANS_PATH)
    meta_labels = joblib.load(LABELS_PATH)
    embedding_model = SentenceTransformer(ST_MODEL_NAME)
    print("✅ API prête : Inférence directe via KMeans chargée.")
except Exception as e:
    print(f"❌ Erreur de chargement des artefacts : {e}")

# ---------------------------------------------------------
# SCHÉMAS DE DONNÉES
# ---------------------------------------------------------

class AvisInput(BaseModel):
    commentaire: str

class PredictResponse(BaseModel):
    text: str
    meta_topic: int
    meta_label: str

# ---------------------------------------------------------
# ENDPOINTS DE PREDICTION
# ---------------------------------------------------------

@app.get("/")
def home():
    return {"status": "online", "method": "Direct KMeans Inference"}

@app.post("/predict", response_model=PredictResponse)
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
# ENDPOINTS DE MONITORING & MÉTRIQUES
# ---------------------------------------------------------

@app.get("/metrics")
def get_metrics():
    """Retourne le score de silhouette calculé lors de l'évaluation"""
    if not os.path.exists(METRICS_PATH):
        return {"error": "Metrics file not found. Run dvc repro first."}
    
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ---------------------------------------------------------
# ENDPOINTS D'ENTRAÎNEMENT (DVC)
# ---------------------------------------------------------

@app.post("/train", tags=["MLOps"])
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