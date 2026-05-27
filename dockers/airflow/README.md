
# question positionnement des outils
Avec dvc on fait des repro de pipeline déjà initialisé, donc on relance des stages/ des étapes exposés avec la définition de leurs entrées et sorties, et dvc choisi en fonction de ce qui a changé la partie qu'il va relancer, c'est ca (dvc repro) ? et il y a du versionnement intermédiaire avec des dvc add, dvc push bien placé.

Avec MLFlow on versionne des models et leurs métriques, au milieu du python, on déclare des étapes, on fait de la registration fine.

Avec AIRflow on lance un peu tout, des bash, des pythons, des containers, ou des sensors on peut aussi réorganiser des pipelines avec des étapes load/transform/train/evaluate et load/transform/predict. on branche et on séquence les noeuds des dags de manière un peu arbitraire mais si on capture la logique, ca va bien.

Alors comment on combine du dvc dans airflow ou mlflow ? mon impression est que si on veut profiter de la "repro" de dvc, on doit simplifier nos dags airflow, ou bien ?


Claude a répondu : Voici comment ça s'articule.

Les trois outils ont des rôles distincts
Outil | Rôle | Granularité
------|------|------------
DVC   |Reproductibilité, versionnement data/modèles, cache des stages | Pipeline ML
MLflow|Tracking expériences, métriques, registry modèles|Run/expérience
Airflow|Orchestration, scheduling, dépendances entre tâches|Workflow

Ton intuition est correcte
Si tu veux profiter de dvc repro dans Airflow, tu as deux approches :

## Option A — Airflow délègue tout à DVC (DAG simplifié)
```python
# Un seul BashOperator qui lance dvc repro
dvc_repro = BashOperator(
    task_id="dvc_repro",
    bash_command="cd /opt/project && dvc repro",
)

```
DVC gère lui-même quels stages relancer. Airflow ne fait qu'appeler DVC et gérer le scheduling. C'est simple mais tu perds la visibilité fine dans l'UI Airflow.

## Option B — Un BashOperator par stage DVC (DAG détaillé)

```python
load = BashOperator(
    task_id="load",
    bash_command="dvc repro load",
)
train = BashOperator(
    task_id="train",
    bash_command="dvc repro train",
)
evaluate = BashOperator(
    task_id="evaluate",
    bash_command="dvc repro evaluate",
)

load >> train >> evaluate
```

Tu gardes la visibilité Airflow stage par stage, mais tu perds une partie de l'intelligence de dvc repro — c'est Airflow qui séquence, pas DVC.

Comment MLflow s'intègre ?

MLflow se branche à l'intérieur des scripts Python appelés par DVC ou Airflow :

```python
# train.py — appelé par un stage DVC ou un PythonOperator Airflow
import mlflow

with mlflow.start_run():
    mlflow.log_param("lr", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

Ni Airflow ni DVC ne savent que MLflow tourne — il est transparent.

L'architecture qui fait sens en pratique
```
Airflow
  └── BashOperator "dvc repro"
        └── DVC
              ├── stage load    ─┐
              ├── stage train   ─┼─ scripts Python avec MLflow tracking
              └── stage evaluate─┘
                                    └── MLflow Registry (modèle validé)

Airflow : scheduling, triggers, alerting, dépendances inter-pipelines
DVC : reproductibilité, cache, ne relance que ce qui a changé
MLflow : tracking des métriques, comparaison des runs, registry
```

La vraie question de design

Est-ce que tu veux que Airflow ou DVC soit le chef d'orchestre du pipeline ML ?


DVC chef → DAG Airflow minimal (1-2 tâches), toute la logique dans dvc.yaml. Idéal si la reproductibilité et le cache sont prioritaires.
Airflow chef → DAG détaillé, DVC utilisé juste pour le versionnement data (dvc push/pull), pas pour repro. Plus de visibilité, moins de cache intelligent.

La plupart des équipes choisissent DVC chef pour le pipeline ML, et Airflow chef pour l'orchestration globale (ingestion data → pipeline ML → déploiement → monitoring).

---

# Combiner dvc add et dvc push en prod ?

Est-ce dans l'orchestration airflow qu'on combine les repro dvc avec les dvc add/dvc push ? y a t'il lieu, en production de faire des dvc add/dvc push sur dagshub ?

la vraie question de ce qu'on versionne et quand.

Ce qu'on versionne avec DVC en production
* dvc repro          → recalcule les stages qui ont changé
* dvc add            → versionne un fichier/dossier non géré par un stage
* dvc push           → envoie les données/modèles vers le remote (DagsHub, S3, etc.)

dvc add est surtout un outil de développement - en production, si tout est défini dans dvc.yaml, les outputs des stages sont automatiquement trackés par DVC. Tu n'as pas besoin de dvc add explicitement.

Ce qui a du sens dans un DAG Airflow en prod
```python
# Après le repro, pousser les artefacts vers DagsHub
dvc_repro = BashOperator(
    task_id="dvc_repro",
    bash_command="cd /opt/project && dvc repro",
)

dvc_push = BashOperator(
    task_id="dvc_push",
    bash_command="cd /opt/project && dvc push",
)

dvc_repro >> dvc_push
```
`dvc push` en production est légitime — tu veux que les modèles et données issus du run soient persistés sur le remote.

`dvc add` en production : rarement justifié
Situation	| dvc add utile ?
---|---
Data brute qui arrive en dehors du pipeline	|✅ oui
Outputs de stages définis dans dvc.yaml	|❌ automatique
Modèle final à archiver manuellement	|⚠️ plutôt MLflow registry
Le seul cas où dvc add a du sens dans un DAG Airflow c'est si tu as une donnée source brute (un dump SQL, un fichier FTP) qui arrive en dehors de DVC et que tu veux versionner avant de lancer le pipeline :

```python
ingest = BashOperator(
    task_id="ingest",
    bash_command="python ingest.py && dvc add data/raw && dvc push",
)

repro = BashOperator(
    task_id="repro",
    bash_command="dvc repro",
)

push = BashOperator(
    task_id="push",
    bash_command="dvc push",
)

ingest >> repro >> push
```

## DagsHub en production : oui ou non ?
DagsHub est très bien pour le dev et les petites équipes — c'est un remote DVC + MLflow tracking en un seul endroit. En production sérieuse :

Remote	|Usage typique
---|---
DagsHub	|Dev, expérimentation, petites équipes
S3 / GCS / Azure Blob	|Production, gros volumes
MLflow Registry	|Modèles validés prêts au déploiement

>En prod tu peux tout à fait garder DagsHub si les volumes de données restent raisonnables - il supporte S3 en backend. La vraie limite est le coût et la bande passante si tu pushes des gros modèles à chaque run.
```
Architecture complète qui fait sens
Airflow DAG
  ├── ingest        → récupère la data brute + dvc add + dvc push
  ├── dvc repro     → relance les stages modifiés
  │     ├── load
  │     ├── transform
  │     ├── train    → MLflow tracking (métriques, params)
  │     └── evaluate → MLflow tracking (métriques finales)
  ├── dvc push      → artefacts → DagsHub/S3
  └── mlflow promote → modèle validé → MLflow Registry

```
Le dvc add disparaît du DAG dès que l'ingestion est elle-même un stage DVC — ce qui est l'objectif final en prod.


## Petit rappel - Résumé des rôles
Fichier|Rôle
---|---
`.dvc/config`|Définit où pousser (remote URL)
`dvc.yaml`|Définit les stages et leurs outputs
`dvc.lock`|Snapshot des hash des outputs actuels
`.dvc/cache`|/Stockage local des artefacts à pousser



## sur un autre host si une mise à jour de docker compose est requise
 un host en docker-compose v1.28, le script d'initialisation de airflow ne sera pas compatible
 il faut passer à la version 2.1>

 bien comparer les versions
 ```bash
 cat /etc/os-release
```

 si host récent
```bash
    sudo apt-get update
    sudo apt-get install docker-compose-plugin
```
si host ancien il se peut que docker-compose en v1.29 soit installé en tant que binaire
en faire un backup
```bash
sudo mv /usr/local/bin/docker-compose /usr/local/bin/docker-compose-v1.bak
```

puis retenter
```bash
sudo apt-get update
sudo apt-get install docker-compose-plugin
```

si E: Unable to locate package docker-compose-plugin

Sur Ubuntu 20.04, le plugin n'est pas dans les repos apt par défaut. Il faut ajouter le repo officiel Docker :
```bash
# Ajouter le repo Docker officiel
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Mettre à jour et installer
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Vérifier
docker compose version

# après on peut repasser en docker compose -f dockers/airflow/docker-compose.yaml up airflow-init

```