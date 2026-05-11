from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

# ---------------------------------------------------------
# CONFIGURATION & CHARGEMENT (Allégé)
# ---------------------------------------------------------

app = FastAPI(
    title="Oscaro Trustpilot API - Direct Meta-Clustering",
    description="API optimisée : Embedding -> KMeans Meta-Topic",
    version="1.2.0"
)

# Chemins vers les artefacts
MODEL_PATH = "models/BERTopic"
KMEANS_PATH = MODEL_PATH + "_kmeans.pkl"
LABELS_PATH = MODEL_PATH + "_meta_labels.pkl"
ST_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

try:
    # 1. On ne charge QUE ce qui est dans predict.py
    kmeans = joblib.load(KMEANS_PATH)
    meta_labels = joblib.load(LABELS_PATH)
    
    # 2. Modèle d'embedding (le cœur du moteur désormais)
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
# ENDPOINTS
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

@app.get("/health")
async def health_check():
    return {"status": "healthy"}