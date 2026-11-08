# Supply Chain MLOps - Client reviews unsupervised clustering 

This project stems from a collective work produced in the context of Liora Machine Learning engineering formation. The current version of it has been reframed entirely for learning purposes.
This work has been both an organizational and technical challenge. Both aspects are documented in the present README file.

## Technicalities

The main purpose was to implement the complete MLops chain from project design up to production and monitoring of the application and models. The starting point was 40 000 scrapped Trustpilot reviews.
We aimed at implementing a clustering tool enabling automatic reviews clustering coupled with sentiment analysis. This reframed version of the project is focused on unsupervised semantic clustering only.
The main point is to display the stability and structure of the whole MLops chain. 

### Rebuild step 1: Exploratory phase

Four pipelines were explored during this exploratory phase. Using SentenceTransformers paraphrase-multilingual-mpnet-base-v2 to generate embeddings (with L2 normalization):
* KMeans
* Spherical KMeans
* HDBSCAN
* UMAP dimensional reduction followed by HDBSCAN

All pipelines were tracked using MLflow. Geometric, semantic, statistical and algorithmic coherence and stability have been investigated.

### Suivi MLflow hébergé sur DagsHub

DagsHub fournit le serveur MLflow du dépôt : aucun serveur local à démarrer.
Les scripts d'entraînement prennent en charge KMeans et Spherical KMeans.
L'intégration utilise directement le client MLflow, sans nécessiter le SDK DagsHub.

Copier `.env.example` vers `.env`, puis renseigner son nom d'utilisateur DagsHub
et son jeton d'accès personnel (avec accès en écriture au dépôt). Le propriétaire
du dépôt peut être différent de l'utilisateur authentifié. Le fichier `.env`
est ignoré par Git ; les scripts lisent les variables d'environnement, sans
charger automatiquement ce fichier.

```bash
cp .env.example .env
# Modifier .env avant de poursuivre, puis exporter ses variables :
set -a
source .env
set +a

uv run python src/train.py \
  --input data/processed/reviews.csv \
  --output-dir models/run-001 \
  --experiment Supply-chain-baseline

uv run python src/evaluate.py \
  --artifact-dir models/run-001 \
  --output-dir reports/run-001 \
  --log-mlflow
```

Adapter le chemin du CSV (colonne `commentaire` par défaut). Utiliser de nouveaux
répertoires de sortie à chaque exécution. L'entraînement envoie paramètres,
métriques, modèles de clustering et encodeur à DagsHub. L'évaluation ajoute ses
résultats aux mêmes runs grâce aux fichiers locaux `mlflow_run.json`.
Les identifiants restent dans l'environnement et ne sont pas enregistrés dans
ces fichiers. `DAGSHUB_USER_TOKEN` ou `DAGSHUB_CLIENT_TOKEN` peuvent aussi fournir
le jeton si `MLFLOW_TRACKING_PASSWORD` n'est pas défini.

Pour choisir un autre dépôt, utiliser `--dagshub-repo OWNER/REPOSITORY` lors de
l'entraînement ; ce paramètre remplace l'URL de l'environnement. L'option
`--tracking-uri URL` permet également de choisir explicitement un serveur MLflow.
Sans ces options et sans `MLFLOW_TRACKING_URI`, l'entraînement reste local, sans
suivi MLflow. La prédiction utilise les artefacts locaux de l'entraînement et
ne nécessite aucune connexion à DagsHub.

Interface du dépôt : [MLflow sur DagsHub](https://dagshub.com/schmilblick-ai/Supply-Chain-MLOps.mlflow).
Configuration : [documentation officielle DagsHub](https://dagshub.com/docs/integration_guide/mlflow_tracking/).

### Pipeline DVC : entraînement → évaluation

`dvc.yaml` enchaîne `train` puis `evaluate` pour les deux méthodes. Les paramètres
sont dans `params.yaml`. L'entrée par défaut est `data/raw/avis_24k_train.csv`,
utilisée dans le notebook d'expérimentation ; adapter `data.input` si nécessaire.

```bash
uv sync
source .venv/bin/activate
dvc pull data/raw.dvc
dvc repro
dvc metrics show
```

Les modèles et l'encodeur sont produits dans `models/clustering`, les rapports
dans `reports/clustering`, et les métriques comparables avec DVC dans
`metrics/clustering.json`. DVC gère les sorties et les remplace lorsqu'une étape
doit être recalculée. Une modification des paramètres d'évaluation relance
uniquement cette étape ; une modification de l'entraînement relance les deux.
Ne pas utiliser ces répertoires pour des fichiers créés manuellement.

Pour enregistrer aussi les résultats sur DagsHub, exporter les variables du
fichier `.env` comme décrit ci-dessus et définir `evaluate.extra_args` à
`"--log-mlflow"` avant `dvc repro`. Les variables d'environnement ne sont pas
suivies par DVC : après un changement de serveur ou l'activation du suivi,
utiliser `dvc repro --force` pour exécuter à nouveau les étapes. Les étapes
restaurées depuis le cache ne renvoient pas leurs artefacts à MLflow.

Les mots-clés sont extraits par défaut. `evaluate.extra_args` accepte aussi
`--skip-keywords` ou `--stability statistical algorithmic global --plot`.
`train.extra_args` permet notamment de fixer la révision de l'encodeur avec
`--revision <commit>` pour stabiliser les téléchargements ultérieurs.

Après une exécution réelle réussie, versionner `dvc.yaml`, `params.yaml`,
`dvc.lock` et `metrics/clustering.json` dans Git ; `dvc push` envoie les sorties
en cache au stockage DVC configuré. `dvc.lock` est généré par DVC à l'exécution.
Voir la [documentation des pipelines DVC](https://doc.dvc.org/user-guide/pipelines/defining-pipelines).
 
