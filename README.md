# 📦 Supply Chain MLOps - Analyse d'Avis Clients (Oscaro)

Ce projet implémente un pipeline MLOps complet pour le clustering et l'analyse de sentiment des avis clients. L'architecture est conçue pour être collaborative (Git/DVC), trackée (MLflow) et prête pour la production (Docker).

## 🛠️ Architecture du Projet

Le projet est décomposé en micro-services orchestrés par Docker :

    Service API : Inférence en temps réel via FastAPI.

    Service Trainer : Exécution du pipeline de réentraînement via DVC.

    Tracking : Expériences et métriques centralisées sur DagsHub (MLflow).

    Storage : Modèles et données volumineux stockés sur le remote DVC de DagsHub.

## 🚀 Guide de Démarrage Rapide

1. Prérequis

    Docker & Docker Compose

    uv (pour la gestion locale des dépendances)

    Un compte DagsHub avec accès au dépôt.

2. Configuration (Secrets)

Créez un fichier .env à la racine (non suivi par Git) :

    Extrait de code

    MLFLOW_TRACKING_URI=https://dagshub.com/votre-username/Supply-Chain-MLOps.mlflow
    MLFLOW_TRACKING_USERNAME=votre-username
    MLFLOW_TRACKING_PASSWORD=votre-token-dagshub
    DAGSHUB_USER_TOKEN=votre-token-dagshub

3. Synchronisation
Avant de lancer l'infrastructure, récupérez le code et les fichiers lourds :

    Bash
    git pull origin main
    dvc pull
4. précautions supplémentaires pour éviter les bugs avec docker-compose v1 avant de lancer le build:

    docker-compose down --volumes --remove-orphans
    docker system prune -af

## 🏗️ Étapes Globales du Pipeline

### Étape 1 : Préparation & Cleaning
Le script src/train.py récupère les données brutes (data/avis_40k.csv), nettoie le texte (suppression des stop-words, lemmatisation avec Spacy) et prépare les features.

### Étape 2 : Entraînement & Clustering
    Embeddings : Utilisation de SentenceTransformers pour vectoriser les avis.

    Modèle : Utilisation de BERTopic pour l'extraction de thématiques.

    Sérialisation : Le modèle est sauvegardé au format safetensors dans models/BERTopic/.

### Étape 3 : Tracking & Versioning
    Chaque run enregistre le Score de Silhouette et les hyperparamètres sur MLflow.

    DVC versionne les fichiers .pkl et les datasets, garantissant que le code sur Git correspond exactement au modèle utilisé.

### Étape 4 : Déploiement en Micro-services
L'application est conteneurisée pour garantir la portabilité :

    Lancer l'API : docker-compose up --build api (disponible sur http://localhost:8000/docs).

    Lancer un Training : docker-compose run --rm trainer.

## 👥 Workflow Collaboratif (Équipe de 3)

Pour maintenir la stabilité du projet, l'équipe suit ces règles :

    1. Branches : Toute modification se fait sur une branche feature/nom.

    2. DVC First : Si le modèle change, on fait dvc push avant le git push.

    3. Review : Les modifications de l'infrastructure Docker doivent être testées localement par au moins deux membres avant le merge sur main.

## 📂 Structure des fichiers
```
Plaintext
.
├── src/                # Scripts source (api.py, train.py)
├── models/             # Modèles (gérés par DVC)
├── data/               # Datasets (gérés par DVC)
├── metrics/            # Scores JSON
├── Dockerfile          # Image Python optimisée avec uv
├── docker-compose.yml  # Orchestration des services
├── dvc.yaml            # Définition du pipeline DVC
└── pyproject.toml      # Dépendances du projet
```
## 💡 Tips pour l'équipe
Si vous obtenez une erreur de fichier manquant au lancement de Docker, vérifiez que vous avez bien fait un dvc pull sur votre machine hôte. Le volume Docker partage vos fichiers locaux avec le container !
