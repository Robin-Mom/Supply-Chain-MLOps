import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from src.api import app, state

# On injecte des mocks directement dans l'état de l'API
state["embedding_model"] = MagicMock()
state["kmeans"] = MagicMock()
state["meta_labels"] = {0: ["topic_test"]}
state["version"] = "1.0.0"

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200 # Maintenant ça passe !

def test_predict_success():
    # Simulation des retours des modèles
    state["embedding_model"].encode.return_value = [[0.1] * 384]
    state["kmeans"].predict.return_value = [0]
    
    payload = {"commentaire": "Ceci est un test"}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    assert response.json()["text"] == "Ceci est un test"

def test_trigger_train():
    # On mock juste le subprocess.run pour ne pas lancer DVC en vrai
    with pytest.mock.patch("subprocess.run"):
        response = client.post("/train")
        assert response.status_code == 200
        assert response.json()["status"] == "Training started"