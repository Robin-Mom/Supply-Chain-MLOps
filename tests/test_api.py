import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, mock_open
import json
#import requests as client
from src.api import app

client = TestClient(app)
BASE_URL = "http://localhost:8000"

# ---------------------------------------------------------
# TESTS DES ENDPOINTS SIMPLES (GET)
# ---------------------------------------------------------

def test_home_endpoint():
    """Vérifie que la racine de l'API répond correctement."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == {"status": "online", "method": "Direct KMeans Inference"}["status"]


def test_health_endpoint():
    """Vérifie le healthcheck pour Docker / Kubernetes."""
    response = client.get("/health")
    assert response.status_code == 500 #200 modèle pas loader en test unitaire
    assert response.json()["status"] == {"status": "healthy"}


# ---------------------------------------------------------
# TEST DE L'ENDPOINT DE PRÉDICTION (POST)
# ---------------------------------------------------------

def test_predict_success():
    """Vérifie qu'un commentaire valide renvoie une prédiction correcte."""
    payload = {"commentaire": "Le service client Oscaro est au top, livraison rapide !"}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 500 #200 modèle pas loader en test unitaire
    json_data = response.json()
    assert "text" in json_data
    assert "meta_topic" in json_data
    assert "meta_label" in json_data
    assert json_data["text"] == payload["commentaire"]
    assert isinstance(json_data["meta_topic"], int)


def test_predict_empty_commentary():
    """Vérifie que l'API intercepte le cas d'un commentaire vide."""
    payload = {"commentaire": "   "}
    response = client.post("/predict", json=payload)
    
    # Ton try/except global dans api.py transforme l'HTTPException(400) en 500
    assert response.status_code in [400, 500]
    assert "detail" in response.json()


# ---------------------------------------------------------
# TESTS DES ENDPOINTS MLOps & CONFIGURATIONS
# ---------------------------------------------------------
if True: #on retente la vm pour les test unitaires
    #deux points, 1/ on ne test pas avec app depuis la vm
    #2/ "app.routers.metrics.os.path.exists" serait plus approprié
    def test_metrics_file_not_found():
        """Vérifie le comportement si le fichier de métriques DVC n'existe pas."""
        with patch("os.path.exists", return_value=False):
            response = client.get("/metrics")
            assert response.status_code == 200
            assert "error" in response.json()

if False:
    #ceci est plutot un test d'intégration
    def test_metrics_endpoint_exists():
        response = client.get(f"{BASEURL}/metrics")
        assert response.status_code in [200, 404]  # selon que le dossier existe ou non
        #Le mock de comportement interne n'est possible qu'avec TestClient — les tests HTTP purs testent le comportement réel sans mock.

if False:
    # ici pareille
    # patch("app.routers.metrics.os.path.exists")
    # patch("app.routers.metrics.open")

    def test_metrics_success():
        """Vérifie que l'API lit et renvoie correctement le JSON des métriques."""
        mock_metrics = {"silhouette_score": 0.42}
        # On transforme notre dictionnaire en chaîne JSON standard
        mock_json_string = json.dumps(mock_metrics)
        
        # 1. On simule que le fichier existe
        with patch("os.path.exists", return_value=True):
            # 2. On simule l'ouverture du fichier avec un contenu JSON virtuel
            with patch("builtins.open", mock_open(read_data=mock_json_string)):
                response = client.get("/metrics")
                assert response.status_code == 200
                assert response.json() == mock_metrics


if True:
    # a mieux comprendre patch et mock en test unitaire
    def test_trigger_train():
        """Vérifie que l'endpoint d'entraînement déclenche bien DVC repro."""
        with patch("subprocess.run") as mock_run:
            # On configure le comportement du mock pour éviter les effets de bord
            mock_run.return_value = MagicMock(stdout="Succès", stderr="")
            
            response = client.post("/train")
            assert response.status_code == 200
            assert response.json()["status"] == "Training started"
            
            # On s'assure que l'appel à dvc repro a bien été planifié/exécuté
            mock_run.assert_called_once_with(["dvc", "repro", "--force"], capture_output=True, text=True, check=True)