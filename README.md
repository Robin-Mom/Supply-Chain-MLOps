# Supply Chain MLOps — clustering local et suivi MLflow DagsHub

La branche `rebuild_step2_scripting` utilise Python, DVC Experiments et MLflow
pour comparer deux SentenceTransformers (MPNet et MiniLM), KMeans et Spherical
KMeans sur cinq seeds. Les configurations Git/DVC distantes et les infrastructures historiques
ne sont pas modifiées.

## Préparation

```bash
uv sync --frozen
source .venv/bin/activate
# Seulement si les deux CSV sont absents :
dvc pull data/train_eval.dvc

# Vérifier la connexion DagsHub en lecture seule, sans créer de run.
python src/tracking.py --check
# Facultatif : limiter les threads BLAS/OpenMP selon la machine.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

Le fichier `.env.dagshub.local` existant fournit `MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME` et `DAGSHUB_USER_TOKEN`. Le token est transmis à MLflow
comme `MLFLOW_TRACKING_PASSWORD` ; cette variable peut aussi être fournie
directement. Aucun `dagshub.init()` ni changement des remotes Git/DVC n'est
nécessaire. Le format attendu est illustré par `.env.example`, sans valeurs.

Les scripts chargent automatiquement ce fichier (chemin `tracking.env_file`
dans `params.yaml`). Ils lisent uniquement les variables de tracking, sans
exécuter le fichier. Les variables déjà exportées restent prioritaires. Pour
utiliser la configuration DagsHub du fichier, retirer d'abord d'éventuels anciens
overrides SQLite (`SUPPLY_CHAIN_TRACKING_URI` ou `MLFLOW_TRACKING_URI`) du shell.
Le fichier local est ignoré par Git et n'est ni une dépendance ni une sortie DVC.

`params.yaml` contient deux encodeurs, deux méthodes, cinq seeds, soit **20
pipelines train → predict → evaluate**, et la seed finale fixée à
42 avant comparaison. Les deux encodeurs sont figés à des commits Hugging Face.
Pour résoudre des révisions manquantes, sans télécharger les poids :

```bash
python src/run_experiments.py --pin-revisions
```

Cette commande ne remplace pas les révisions déjà définies. Pour changer une
révision, indiquer un autre commit explicitement, ou mettre sa valeur à `null`
puis la résoudre à nouveau. Versionner les scripts, `dvc.yaml`, `params.yaml` et
`uv.lock` avant de lancer les expériences, particulièrement avec `--temp`.
DVC ne transporte pas automatiquement les nouveaux fichiers source non suivis.

Les seules données utilisées sont `data/train_eval/train_20k.csv` (20 000 lignes)
et `data/train_eval/eval_4k.csv` (4 000 lignes). La colonne texte est
`commentaire` ; `Unnamed: 0` contient l'identifiant d'origine. Les identifiants
manquants, dupliqués ou communs aux deux jeux et les avis vides sont rejetés.
Aucune ligne n'est supprimée silencieusement.

## Lancer les expériences

```bash
# Afficher les 20 commandes sans entraîner de modèle.
python src/run_experiments.py --dry-run

# Exécuter séquentiellement les 20 expériences, puis la sélection globale.
python src/run_experiments.py

# Variante : workspaces isolés, même cache DVC et même stockage MLflow.
python src/run_experiments.py --temp

# Reprendre un sweep interrompu en vérifiant les expériences déjà enregistrées.
python src/run_experiments.py --temp --resume
```

Choisir une variante d'exécution pour un même sweep. La reprise utilise les noms
DVC déterministes ; elle exige que les runs MLflow associés soient encore
présents. Pour recommencer un sweep déjà terminé, conserver ses résultats et
utiliser un nouveau dossier `--output` avec `--resume`. Les scripts refusent
d'écraser les résultats de sélection existants.

L'orchestrateur charge les credentials avant de lancer DVC. Les processus
isolés héritent des variables et du chemin absolu du fichier local ; aucune
copie de ce fichier n'est placée dans les expériences.

L'orchestrateur cible `evaluate` : DVC ne calcule que les dépendances nécessaires
à l'encodeur actif. Pour une expérience individuelle dans le workspace courant :

```bash
SUPPLY_CHAIN_DVC_NAME=essai-minilm-43 dvc exp run evaluate \
  --name essai-minilm-43 \
  -S experiment.encoder=minilm \
  -S experiment.method=spherical_kmeans \
  -S experiment.seed=43

dvc exp show
dvc metrics show
dvc exp diff
# Consulter les runs dans l'onglet Experiments du repository DagsHub.
```

L'orchestrateur complète les tags MLflow avec le commit de l'expérience DVC
après chaque exécution et écrit un index dans `reports/dvc_run_index/`. Une
commande DVC individuelle conserve le nom de l'expérience et la révision Git
de base dans le manifeste ; la révision finale peut être associée ensuite avec
`run_experiments.record_revision()`.

Le premier calcul réel régénère `dvc.lock` avec les nouveaux stages et leurs
empreintes. Le lock historique n'a pas été remplacé par des résultats
synthétiques. Versionner le lock obtenu après cette première exécution.

## Étapes et stockage

1. **`embed@<encodeur>`** charge un seul transformer, encode successivement train
   puis eval par lots et écrit deux matrices L2 en `float32` sous
   `models/embeddings/<encodeur>/`. Les métadonnées enregistrent la révision,
   le prétraitement, les empreintes et les identifiants ordonnés. Le cache DVC
   réutilise ces quatre matrices entre méthodes et seeds. La VRAM est libérée à la
   fin du stage. Le cache Hugging Face partagé conserve les poids ; aucune copie
   de transformer n'est incluse dans les runs MLflow.
2. **`train`** ajuste une seule configuration, exclusivement sur les embeddings
   train chargés en mémoire mappée. Le bundle, les hyperparamètres, les métriques
   d'apprentissage et le manifeste sont enregistrés dans `Supply_chain_train`.
   Les labels train ne sont pas conservés dans le bundle. Les tableaux de
   travail nécessaires aux estimateurs peuvent occuper de la RAM supplémentaire.
3. **`predict`** vérifie le manifeste, le modèle et le cache eval, puis appelle
   seulement `predict()`. Il écrit les colonnes `observation_id`, `logical_id`
   et `cluster_id`, ainsi qu'un manifeste de prédiction. Aucun transformer n'est
   rechargé pendant les 20 inférences utilisant le cache.
4. **`evaluate`** lit ces prédictions et le cache eval, calcule la silhouette
   cosine et crée un nouveau run `Supply_chain_eval` avec
   `train_mlflow_run_id`. Les prédictions ne sont journalisées que dans ce run,
   sans copie du modèle ou du transformer.

Les sorties de chaque configuration sont relatives au workspace, sous
`models/runs/`, `models/predictions/` et `models/evaluations/`, puis organisées par
encodeur/méthode/seed. DVC gère leur remplacement et leur restauration. Ne pas y
placer de fichiers indépendants du pipeline.

Avec DagsHub, les paramètres, métriques et artefacts sont journalisés dans le
repository indiqué par l'URI fournie dans le fichier local. Les noms restent
`Supply_chain_train` et `Supply_chain_eval`. La sélection récupère les résultats
et les prédictions par MLflow, puis conserve les références du modèle. Une future
inférence charge le modèle de clustering depuis ces URI ; les poids
des transformers restent uniquement dans le cache Hugging Face local.

Pour lancer directement `dvc exp run --temp` sans l'orchestrateur, exporter
`SUPPLY_CHAIN_ENV_FILE` avec le chemin absolu de `.env.dagshub.local` avant la
commande. Les stages pourront ainsi lire le fichier depuis leur workspace isolé.

Le tracking local reste possible avec un override explicite :

```bash
export SUPPLY_CHAIN_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"
python src/run_experiments.py --temp
```

Avec SQLite, l'URI doit être absolue. Les nouvelles expériences MLflow stockent
leurs artefacts sous `mlartifacts/<nom-expérience>` à côté de la base, hors des
workspaces temporaires. Une expérience MLflow déjà existante conserve son
emplacement d'artefacts. Pour un serveur distant, fournir son URI exacte dans
`tracking.uri` ou `SUPPLY_CHAIN_TRACKING_URI` et exporter les identifiants séparément.
L'ordre de priorité est `SUPPLY_CHAIN_TRACKING_URI`, puis `tracking.uri`, puis
`MLFLOW_TRACKING_URI`. Les exports du shell sont prioritaires sur les valeurs
homonymes du fichier local. Aucun secret n'est écrit
dans les fichiers versionnés ; les URI contenant des identifiants sont refusées.

Le cache DVC ne rejoue pas les écritures MLflow. Préserver ensemble la base et
ses artefacts ; `dvc push` ne les sauvegarde pas. Changer l'URI d'environnement
n'invalide pas le cache DVC : si le stockage MLflow a été supprimé ou déplacé,
restaurer ce stockage ou forcer les stages concernés. Ne pas utiliser `--force`
sur les embeddings pour simplement régénérer les runs MLflow : reproduire
séquentiellement `train`, `predict`, puis `evaluate` avec `--single-item --force`.

## Évaluation et sélection

Pour comparer les runs dans DagsHub, afficher les colonnes MLflow
`silhouette_cosine` (à maximiser), `ari_mean` (à maximiser) et `inertia`
(à minimiser au sein d'une même méthode et d'un même espace d'embeddings).
La silhouette et l'inertie sont disponibles dans les expériences d'entraînement
et d'évaluation après `evaluate`. L'inertie est l'objectif sur les données
d'entraînement : somme des distances euclidiennes au carré pour KMeans,
somme des distances cosine pour Spherical KMeans ; leurs valeurs ne sont donc
pas directement comparables entre méthodes.
Après le sweep complet, la sélection ajoute à chaque run `ari_mean`, moyenne
des ARI de sa partition eval avec les quatre autres seeds du même encodeur et
de la même méthode, ainsi que `ari_std` et `ari_pairs` (4). Cet ARI mesure la
stabilité, sans labels de référence. Les anciens runs ne sont pas mis à jour
automatiquement : ces métriques concernent les nouvelles exécutions du pipeline.

La silhouette est exacte sur les 4 000 avis par défaut (`sample_size: 0`).
Scikit-learn calcule les distances par blocs, avec `working_memory_mb: 32` pour
limiter leur mémoire de travail. Une seule configuration est évaluée à la fois.
Ce budget concerne les distances, pas la mémoire totale du processus.

Si `sample_size` est réduit, les positions sont tirées avec `sample_seed`, triées
et identiques pour tous les encodeurs, méthodes et seeds. Leur empreinte est
journalisée. Une silhouette impossible est enregistrée à `null` avec sa raison.

La sélection est extérieure au DAG de chaque expérience :

```bash
# Déjà appelée automatiquement après les 20 expériences ; commande indépendante :
python src/select_best.py --output reports/selection_alternative
```

Elle recherche dans `Supply_chain_eval` le sweep correspondant aux données,
paramètres, sources et lock actuels. Elle vérifie les 20 configurations finies,
leurs liens aux runs train et l'alignement des observations. En cas de répétition
d'une configuration dans un sweep, le dernier run terminé est retenu.
Elle calcule, pour chacun des quatre groupes :

- la moyenne et l'écart-type de silhouette sur cinq seeds (`ddof=1`) ;
- les dix ARI deux à deux et leur moyenne/écart-type sur les partitions eval.

Un groupe dont une silhouette est indéfinie est signalé et exclu du classement.
Le classement maximise la silhouette moyenne, puis l'ARI moyen en cas d'égalité
exacte ; les noms départagent les égalités restantes. La seed finale est toujours
celle déclarée à l'avance, indépendamment de ses performances individuelles.

`comparison.csv`, `comparison.json`, `ari_pairs.csv` et `selection.json` sont
produits dans `reports/experiment_selection/` et journalisés dans un run MLflow
de rôle `selection`. Le fichier de sélection conserve les URI du modèle et du
manifeste, leurs références MLflow/DVC et un plan de réentraînement futur sur
les 24 000 avis. Ce réentraînement n'est pas exécuté. Aucun score train n'entre
dans le classement et aucun modèle n'est recopié pendant la sélection.

Les identifiants logiques sont les SHA-256 des configurations d'entraînement
(encodeur, révision, prétraitement, données, algorithme, hyperparamètres et seed).
L'identifiant de sweep distingue les ensembles de résultats selon leurs
paramètres d'évaluation, leurs sources et leur lock. Les runs MLflow distinguent
les exécutions successives d'une même configuration.

## Inférence réutilisable et compatibilité

`ExperimentPredictor` dans `src/predict.py` charge une seule fois le clustering
et, pour du texte nouveau, charge paresseusement le transformer figé. Il réutilise
ce transformer entre appels, applique le même prétraitement et la même
normalisation, puis vérifie les dimensions. Exemple pour une future API :

```python
import json
import mlflow
from predict import ExperimentPredictor  # avec src dans PYTHONPATH

selection = json.load(open("reports/experiment_selection/selection.json"))
mlflow.set_tracking_uri(selection["tracking_uri"])
predictor = ExperimentPredictor.from_mlflow(selection["manifest_uri"], "models/inference")
labels = predictor.predict_texts(["Livraison rapide et produit conforme"])
```

`from_mlflow()` charge également les credentials locaux avant de télécharger
les artefacts, tout en conservant l'URI sélectionnée avec `set_tracking_uri()`.

Seules les transformations effectivement sauvegardées peuvent être utilisées.
Le pipeline actuel n'en ajuste aucune en plus de la normalisation L2 ; un
manifeste annonçant des transformations inconnues est rejeté.

Les anciennes interfaces CLI et fonctions de `train.py`, `predict.py`,
`evaluate.py` et `select_model.py` sont conservées pour les bundles historiques.
Elles gardent leur fonctionnement historique, y compris les analyses
optionnelles de stabilité sur les données train. **Le nouveau protocole utilise
`--params params.yaml` et `select_best.py`** ; les anciennes commandes ne doivent
pas servir à alimenter sa comparaison des 20 expériences.

## API FastAPI locale

Depuis la racine du dépôt :

```bash
uv sync --frozen
uv run --frozen uvicorn api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

`SUPPLY_CHAIN_SELECTION_PATH` permet de remplacer le chemin par défaut
`reports/experiment_selection/selection.json`. Ce fichier est un résultat local
de `select_best.py` et peut être absent de Git : utiliser la sélection déjà
produite, sans relancer les expériences ni la sélection pour démarrer l'API.
Les chemins relatifs sont résolus depuis le répertoire courant.

Au démarrage, le lifespan lit cette sélection et configure le tracking avec son
`tracking_uri` enregistré, puis `ExperimentPredictor.from_mlflow()` télécharge
son `manifest_uri` et le bundle référencé dans un répertoire temporaire, nettoyé
à l'arrêt. Aucun run ni artefact distant n'est créé ou modifié. Le checksum,
l'identifiant logique, la configuration, le run d'entraînement et les
hyperparamètres de l'estimateur sont vérifiés avant une inférence d'initialisation.
Cette dernière charge l'encodeur à sa révision Hugging Face figée et vérifie le
chemin complet de prédiction. Le prétraitement et la normalisation restent ceux
de `ExperimentPredictor`. Aucun fit n'est appelé.

Les credentials sont lus par `tracking.py` dans `.env.dagshub.local` ou dans les
variables exportées, prioritaires : `MLFLOW_TRACKING_USERNAME` et
`MLFLOW_TRACKING_PASSWORD` (ou `DAGSHUB_USER_TOKEN` / `DAGSHUB_CLIENT_TOKEN`).
`SUPPLY_CHAIN_ENV_FILE` permet de choisir un autre fichier local ignoré. L'URI
du tracking est celle de la sélection par défaut ; `SUPPLY_CHAIN_TRACKING_URI`
permet un override explicite, indépendamment des paramètres de la dernière
expérience DVC. Les contrôles du modèle sélectionné restent obligatoires.
Ne placer aucun identifiant dans les URI.
Les poids Hugging Face doivent être en cache ou téléchargeables au démarrage ;
`HF_HOME` permet de choisir le cache et `HF_HUB_OFFLINE=1` impose son usage hors
ligne. Le clustering et l'encodeur sont réutilisés entre requêtes. Utiliser un
seul worker local ; chaque worker supplémentaire chargerait sa propre copie.

```bash
curl -i http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model/info
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Livraison rapide et produit conforme","Commande jamais reçue"]}'
```

`/health` distingue la disponibilité de l'application (`status: "ok"`) de celle
du modèle (`model_ready`). Il répond 200 seulement après initialisation réussie,
sinon 503. Une initialisation échouée laisse l'application accessible ;
`/predict` et `/model/info` répondent alors 503. Corriger la sélection, l'accès
MLflow ou le cache de l'encodeur, puis redémarrer le processus. Aucun modèle de
secours n'est choisi. Les erreurs d'exécution donnent une réponse 500 générique,
sans trace interne. Les journaux d'échec évitent aussi le contenu des exceptions
distantes, qui pourrait contenir des secrets.

`/predict` accepte de 1 à 64 chaînes de 1 à 5 000 caractères chacune, rejette
les avis blancs et les champs supplémentaires (422), et conserve l'ordre des
avis. Les accès concurrents à l'encodeur partagé sont sérialisés. Les textes ne
sont ni journalisés ni conservés par l'application. `/model/info` expose seulement
l'identifiant logique, l'encodeur et sa révision, la méthode, la seed et la
version du prétraitement. La documentation interactive est disponible à `/docs`.
Ce service est destiné à un usage local, sans authentification ni déploiement.

Tests API hors ligne (transport MLflow et encodeur simulés, sans credentials ni
poids réels) et régression complète :

```bash
MLFLOW_DISABLE_AGENT_HINT=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --frozen pytest tests/test_api.py -q
MLFLOW_DISABLE_AGENT_HINT=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --frozen pytest tests -q
```

## Docker : inférence locale

Le déploiement utilise quatre services sur le réseau Compose par défaut :
`client → 127.0.0.1:8080 → nginx:80 → api:8000`. NGINX résout le nom de service
`api` avec le DNS Docker ; `localhost` dans NGINX désignerait le proxy lui-même.
La résolution dynamique permet de retrouver l'API si son adresse change après
une recréation. Aucune configuration réseau supplémentaire n'est nécessaire.
Le Dockerfile et la commande Uvicorn restent utilisables comme auparavant ;
seule l'entrée réseau du déploiement Compose change.
Prometheus collecte directement `api:8000/metrics`, indépendamment de NGINX,
et publie son interface uniquement sur `127.0.0.1:9090`.

Prérequis : Docker Engine avec BuildKit et Docker Compose v2+, accès au moteur
Docker, accès réseau aux registres pendant le build et assez d'espace pour les
dépendances ML verrouillées. Aucun GPU ni environnement Python hôte n'est requis.

```bash
cp docker.env.example .env.docker.local
chmod 600 .env.docker.local
# Au premier démarrage seulement : créer puis renseigner les credentials Grafana.
test -f .env.grafana.local || cp grafana.env.example .env.grafana.local
chmod 600 .env.grafana.local
# Renseigner les credentials existants dans ce fichier local, sans les afficher.
docker compose config -q
docker compose build api
# Vérification syntaxique autonome, sans démarrer le modèle :
docker compose run --rm --no-deps nginx nginx -t
docker compose up -d --wait --wait-timeout 660
docker compose ps
```

`.env.docker.local` est déjà ignoré par la règle `.env.*`. Compose injecte ses
variables uniquement à l'exécution. Le fichier n'entre pas dans le contexte de
build. Pour DagsHub, fournir `MLFLOW_TRACKING_USERNAME` et `DAGSHUB_USER_TOKEN`
ou `MLFLOW_TRACKING_PASSWORD`. Un `SUPPLY_CHAIN_TRACKING_URI` optionnel dans ce
fichier remplace l'URI par défaut sans réécrire la sélection ; les identifiants
et checksums restent vérifiés. Éviter `docker compose config` sans `-q`,
qui afficherait les variables résolues, credentials inclus.

Le fichier hôte monté par défaut est
`config/selection.docker.json`, en lecture seule. Il référence le run DagsHub
`c650a6675bf5438c81d90b96ee4e154c` sur
`https://dagshub.com/mom.robin.63/Supply-Chain-MLOps.mlflow`. Le manifeste et le
bundle distant ont été vérifiés contre la configuration de la sélection locale,
l'identifiant logique
`c93cb37fac6cbebb9f3cce20cb962722c0190a5e588afffbb80f6512eebafab0` et le SHA-256
`0576bf251a3b4daad93598bc971669d32b18e71d9fd5a9b1b98034486ca3f4af`.
Ce fichier Docker ne contient ni credentials ni références SQLite locales.
`reports/experiment_selection/selection.json` reste inchangé, et aucun montage
de base ou d'artefacts locaux n'est nécessaire pour ce run distant. Pour utiliser
un autre chemin **vers la même sélection validée**, définir une variable shell :

```bash
export SUPPLY_CHAIN_SELECTION_FILE=/chemin/vers/selection.json
docker compose up -d --wait --wait-timeout 660
```

`SUPPLY_CHAIN_SELECTION_PATH` configure le chemin **dans le conteneur**, par
défaut `/app/config/selection.json` ; Compose l'utilise aussi comme cible du
montage. Ces deux variables doivent être exportées dans le shell ou fournies
via `docker compose --env-file .env.docker.local ...` pour l'interpolation
Compose. Un simple `env_file` du service fournit les variables au processus,
pas à l'interpolation du YAML. Le montage refuse de créer un répertoire si le
fichier source manque. Ne pas générer une nouvelle sélection pour démarrer.

La sélection locale historique peut référencer SQLite et des artefacts `file://` locaux.
Dans ce cas, le Compose standard ne suffit pas : les chemins absolus contenus
dans `tracking_uri` et dans la base doivent exister dans le conteneur. Ne pas
remplacer ces références par celles d'un autre modèle. Une sélection DagsHub
portable doit être vérifiée avant usage ; aucun fichier de sélection n'est
réécrit par Docker.

Pour une sélection SQLite, utiliser un override local hors dépôt, par exemple
`/tmp/compose.mlflow.local.yaml`, avec les chemins réellement lus dans les
métadonnées. Préférer une sauvegarde SQLite cohérente et jetable pour que
l'initialisation MLflow ne puisse pas modifier la base historique. Conserver son
nom de fichier et monter son dossier au chemin parent attendu par l'URI ; le
dossier de copie doit être accessible en écriture à l'UID 10001 pour les fichiers
SQLite auxiliaires. Monter le dossier d'artefacts original en lecture seule :

```yaml
services:
  api:
    volumes:
      - type: bind
        source: /chemin/hote/copie-sqlite
        target: /chemin/absolu/parent-attendu-par-tracking-uri
        bind:
          create_host_path: false
      - type: bind
        source: /chemin/hote/artefacts-du-run-selectionne
        target: /chemin/absolu/artefacts-enregistre-dans-mlflow
        read_only: true
        bind:
          create_host_path: false
```

```bash
docker compose -f compose.yaml -f /tmp/compose.mlflow.local.yaml config -q
docker compose -f compose.yaml -f /tmp/compose.mlflow.local.yaml \
  up -d --wait --wait-timeout 660
```

Utiliser ces mêmes options `-f` pour toutes les commandes suivantes si cet
override est nécessaire. Les chemins hôtes restent dans l'override local, jamais
dans l'image. La sélection est montée telle quelle, et le modèle conserve ses
vérifications de checksum, configuration, seed et révision d'encodeur.

L'image comporte un étage de construction avec uv 0.10.2 et un étage d'exécution
Python 3.12.12 slim Bookworm, avec les deux images épinglées par digest.
`uv sync --locked --no-dev --no-install-project`
installe les dépendances de production depuis `uv.lock`, sans le modifier.
Le compilateur et uv restent dans le builder ; seul `libgomp1` est ajouté à
l'étage final avec l'environnement Python construit. La liste de fichiers copiés
et `.dockerignore` excluent données, modèles, `.venv`, Git et secrets.
Le processus utilise l'UID/GID 10001, sans capacités Linux supplémentaires,
avec `no-new-privileges`. FastAPI reste sur le port 8000 du réseau Compose,
sans publication sur l’hôte ; seul NGINX publie `127.0.0.1:8080:80`.

Cette première image conserve tout le groupe de dépendances de production du
projet pour respecter le lock existant. Celui-ci inclut les bibliothèques CUDA
de PyTorch et des bibliothèques d'expérimentation : l'image est volumineuse même
si l'exécution est CPU (`CUDA_VISIBLE_DEVICES` vide, aucun GPU exposé). Un lock
CPU et un groupe d'inférence séparés seraient une optimisation ultérieure.

Le volume nommé `huggingface-cache` est monté sur `/cache/huggingface` et initialisé
avec les permissions de l'utilisateur applicatif. Il survit aux redémarrages,
recréations et à `docker compose down`. Les poids de la révision figée sont
réutilisés ; Hugging Face peut encore vérifier des métadonnées en ligne.
Après un premier chargement réussi, on peut imposer leur usage hors ligne :

```bash
HF_HUB_OFFLINE=1 docker compose up -d --force-recreate --wait --wait-timeout 660
```

Cela désactive uniquement le réseau Hugging Face, pas les téléchargements MLflow.
`ExperimentPredictor.from_mlflow()` écrit dans la destination fournie par le
lifespan, qui est un `TemporaryDirectory` nettoyé à l'arrêt. Il n'implémente pas
de cache d'artefacts persistant : le manifeste et le petit bundle de clustering
sont donc récupérés à chaque démarrage. Monter un volume sur `/tmp` ne changerait
pas ce comportement. Aucun volume MLflow trompeur n'est ajouté et aucun code
d'inférence n'est modifié pour cette étape. Les artefacts d'un store local,
lorsqu'ils sont montés, restent bien sur l'hôte.

```bash
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/model/info
curl --fail http://127.0.0.1:8080/docs
curl --fail http://127.0.0.1:8080/openapi.json
curl --fail -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Livraison rapide et produit conforme","Commande jamais reçue","Service client très réactif"]}'
docker compose logs --tail=100 api nginx
docker compose restart api
docker compose ps
docker compose stop
docker compose down
```

Pour ces trois avis dans cet ordre, la référence locale validée est `[6, 7, 6]`.
Vérifier aussi `model_id` dans `/model/info` et `/predict`. Toute différence
d'identité, checksum ou prédiction doit être investiguée ; ne pas choisir un
autre modèle automatiquement. Le volume Hugging Face est conservé, tandis que
les deux petits artefacts MLflow sont récupérés depuis DagsHub à chaque démarrage.

NGINX utilise l'image officielle `nginx:1.28.3-alpine`, épinglée par version et digest.
Sa configuration `nginx/nginx.conf` est montée en lecture seule et transmet
les chemins, paramètres de requête et réponses FastAPI sans réécriture.
Swagger UI est accessible à `http://127.0.0.1:8080/docs` et utilise
`/openapi.json` sur la même origine. Ses ressources JavaScript/CSS externes
nécessitent un accès Internet depuis le navigateur, comme avec FastAPI seul.
Les en-têtes `Host` (port inclus), `X-Real-IP`, `X-Forwarded-For` et
`X-Forwarded-Proto` sont transmis. La limite de corps est de 2 Mio, le délai
de connexion de 5 secondes et les délais de lecture/envoi de 120 secondes.
Les réponses d'inférence ne sont pas mises en cache et les requêtes ne sont
pas rejouées en cas d'erreur upstream. Le numéro de version du serveur est masqué.
Les access logs contiennent méthode, chemin sans query string, statut, durée
et adresse cliente, sans corps de requête ni en-tête d'autorisation.
Seul `api` reçoit `.env.docker.local` ; NGINX ne reçoit aucun credential MLflow.

Les quatre services utilisent `restart: unless-stopped`. NGINX attend initialement
que l'API soit saine (`depends_on: condition: service_healthy`). Son healthcheck
appelle ensuite `/health` à travers le proxy : si l'API devient indisponible,
NGINX devient également non sain après les échecs configurés. La condition de
dépendance n'est pas une surveillance continue et Docker ne redémarre pas un
conteneur simplement parce qu'il devient non sain. Les réponses 422/503 de
FastAPI sont transmises telles quelles ; une connexion upstream impossible
donne typiquement 502 et un délai de réponse dépassé donne 504.

Pour diagnostiquer 502/504 :

```bash
docker compose ps
docker compose logs --tail=100 api nginx
docker compose exec nginx nginx -t
docker compose exec api python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
```

Vérifier le chargement du modèle et les credentials côté API avant d'augmenter
les timeouts. Après une recréation de l'API, laisser au DNS Docker quelques
secondes pour actualiser l'adresse. Pour vérifier l'échec upstream localement,
arrêter seulement `api`, appeler `/health` via le port 8080 et constater 502
ou 504 selon que la connexion est refusée ou expire,
puis relancer `docker compose up -d --wait --wait-timeout 660`.
Les healthchecks continuent à constater l'état réel après le démarrage.
Une extension future pourrait ajouter TLS au niveau NGINX avec une configuration
dédiée ; aucun certificat ni domaine public n'est ajouté ici. La collecte
Prometheus interne est décrite dans la section monitoring ci-dessous.
Les directives proxy suivent la [documentation officielle NGINX](https://nginx.org/en/docs/http/ngx_http_proxy_module.html).

Le healthcheck Python n'exige pas curl dans l'image. Il vérifie HTTP 200 et
`model_ready: true` sur `/health` toutes les 30 secondes, avec un timeout de
5 secondes, trois échecs tolérés et une grâce de 10 minutes pour le premier
téléchargement. Pendant le lifespan, Uvicorn peut ne pas encore accepter les
connexions ; cela ne signifie jamais que le modèle est prêt. Un démarrage raté
reste non sain et les endpoints de modèle répondent 503.

En cas d'erreur d'authentification, vérifier le fichier runtime et les droits de
lecture du run sélectionné, puis recréer le service. Si la sélection manque,
vérifier son chemin hôte et ne pas lancer de nouveau sweep. Pour un store local,
vérifier séparément le montage SQLite et le chemin des artefacts. Pour les poids,
vérifier le réseau et l'espace disque, retirer `HF_HUB_OFFLINE=1` si le cache
n'a jamais été rempli. Avec un volume hôte personnalisé, assurer l'accès à
l'UID 10001 ; les volumes Docker nommés utilisent les permissions de l'image.
`docker compose down --volumes` supprime le cache et impose de télécharger les
poids à nouveau : ne l'utiliser que pour une remise à zéro volontaire.

Les tests Python restent exécutables avec la commande de la section précédente.
La syntaxe Compose,
le build et les appels réels ci-dessus constituent les vérifications Docker.
La construction suit le [guide Docker de uv](https://docs.astral.sh/uv/guides/integration/docker/)
et les montages la [référence Compose](https://docs.docker.com/reference/compose-file/services/).

## Monitoring opérationnel : Prometheus

```bash
docker compose config -q
docker compose build api
docker compose run --rm --no-deps nginx nginx -t
docker compose run --rm --no-deps --entrypoint promtool prometheus \
  check config /etc/prometheus/prometheus.yml
docker compose up -d --wait --wait-timeout 660
docker compose logs --tail=100 api nginx prometheus
```

L'interface est disponible sur `http://127.0.0.1:9090`. Vérifier les cibles dans
la page **Status → Target health**, ou via `http://127.0.0.1:9090/api/v1/targets` :
le job `fastapi` doit être UP et son URL `http://api:8000/metrics`. La collecte
est définie dans `monitoring/prometheus/prometheus.yml` : intervalle 15 secondes,
timeout 5 secondes. Elle ne dépend pas du proxy et continue à tenter les scrapes
si le modèle ou NGINX est indisponible. Le healthcheck Prometheus vérifie son
endpoint `/-/ready` ; il ne garantit pas que chaque cible est UP.
L'image officielle `prom/prometheus:v3.5.0` est épinglée par version et digest.

L'instrumentation utilise la [bibliothèque officielle prometheus-client](https://prometheus.github.io/client_python/registry/)
avec un registre par application, sans collecteurs globaux ni métadonnées du
modèle. Le middleware ASGI conserve les réponses et compte les erreurs aussi.
Un seul processus Uvicorn est pris en charge ; pour plusieurs workers, il faudra
une configuration multiprocess dédiée. Les compteurs de l'API repartent de zéro
au redémarrage de son processus ; `rate()` gère ces resets.

| Métrique | Type | Labels / signification |
|---|---|---|
| `http_requests_total` | Counter | `method`, `route`, `status_code` ; requêtes terminées, y compris erreurs |
| `http_request_duration_seconds` | Histogram | mêmes labels ; durée complète, files d'attente et inférence comprises |
| `http_requests_in_progress` | Gauge | `method` ; requêtes actives, décrémentées même en cas d'exception |
| `model_ready` | Gauge | sans label ; 1 si le prédicteur est initialisé, 0 sinon |

Les buckets de latence vont de 5 ms à 300 secondes, plus `+Inf`. Les labels de
route utilisent les templates du routeur ; les chemins inconnus sont regroupés
sous `__unmatched__`. Les méthodes inconnues sont regroupées sous `OTHER`.
Les labels ne contiennent ni texte d'avis, URL brute, query string, credential,
identifiant individuel, embedding ou prédiction. La readiness lit seulement
l'état existant du prédicteur : aucun téléchargement ni contrôle coûteux au scrape.
Les scrapes `/metrics` et leur redirection avec slash final sont exclus des
compteurs, histogrammes et jauges HTTP, pour éviter la récursion et le bruit.
Les healthchecks `/health` sont comptés ; filtrer `route="/predict"` pour étudier
l'inférence. Les requêtes rejetées par NGINX (403/413) ne touchent pas FastAPI
et ne figurent donc pas dans ces métriques.

PromQL utile, avec quelques minutes de trafic pour les fenêtres de 5 minutes :

```promql
# Débit d'inférence (requêtes/s)
sum(rate(http_requests_total{job="fastapi",route="/predict"}[5m]))

# Fraction d'erreurs serveur d'inférence ; sans trafic le ratio peut être NaN
sum(rate(http_requests_total{job="fastapi",route="/predict",status_code=~"5.."}[5m]))
/
sum(rate(http_requests_total{job="fastapi",route="/predict"}[5m]))

# Latence p95 d'inférence (secondes)
histogram_quantile(0.95,
  sum by (le) (rate(http_request_duration_seconds_bucket{job="fastapi",route="/predict"}[5m])))

# Disponibilité de la collecte et du modèle : deux notions distinctes
up{job="fastapi"}
model_ready{job="fastapi"}
```

NGINX bloque `/metrics` avec HTTP 403, y compris variantes de casse, slash final,
suffixes, query strings et chemins normalisés/encodés. FastAPI n'a toujours aucun
port publié. Prometheus et NGINX n'ont ni fichier d'environnement de credentials
ni montage des artefacts. Les données sont accessibles aux services du réseau
Compose et à l'utilisateur local via l'UI Prometheus, sans authentification.

```bash
curl -i http://127.0.0.1:8080/metrics
curl -i http://127.0.0.1:8080/metrics/
curl -i --path-as-is http://127.0.0.1:8080/%6detrics
# Vérification interne, sans passer par NGINX :
docker compose exec prometheus wget -q -O - http://api:8000/metrics
```

Le volume nommé `prometheus-data` conserve la TSDB sur `/prometheus` avec une
rétention de 15 jours. Il survit aux redémarrages/recréations et à
`docker compose down`. Les nouveaux volumes héritent des permissions de l'image.
Conserver le même nom de projet Compose pour retrouver les mêmes volumes.
`docker compose down --volumes` supprimerait les données et le cache Hugging Face.
Si 9090 est occupé, ne pas arrêter un service tiers ; résoudre le conflit avant
de démarrer ce stack. La configuration suit la [documentation Prometheus](https://prometheus.io/docs/prometheus/latest/configuration/configuration/).

Tests hors ligne :

```bash
MLFLOW_DISABLE_AGENT_HINT=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --frozen pytest tests -q
```

Les tests d'intégration sont ignorés par défaut. Une fois le stack explicitement
démarré et prêt, ils vérifient les variantes bloquées par NGINX, la cible UP,
les incréments observés après un scrape et les trois prédictions de référence :

```bash
RUN_MONITORING_INTEGRATION=1 uv run --frozen pytest tests/integration/test_monitoring_stack.py -q
docker compose restart prometheus
docker compose up -d --wait --wait-timeout 660
docker compose down
```

Ces métriques décrivent le fonctionnement du service, pas la qualité sémantique
des clusters, la dérive des données ou la pertinence des prédictions. Aucun jeu
de simulation, référence monitoring ou test final n'est utilisé. Evidently, alertes et simulation restent hors de cette phase. Grafana utilise
ces métriques opérationnelles selon la section suivante.

## Grafana : dashboards provisionnés

L'image officielle `grafana/grafana:13.2.2` est épinglée par version et digest.
Le quatrième service accède à `http://prometheus:9090` sur le réseau Compose
existant. Son interface est limitée à `http://127.0.0.1:3000`, avec un volume
`grafana-data` pour la base, les utilisateurs et leurs préférences. Les credentials
DagsHub restent exclusivement dans `api` ; Grafana utilise un fichier séparé.

Avant le premier démarrage, copier `grafana.env.example` vers
`.env.grafana.local`, ignoré par Git, et définir `GF_SECURITY_ADMIN_USER` et
`GF_SECURITY_ADMIN_PASSWORD` avec un mot de passe fort et unique. Protéger le
fichier avec `chmod 600 .env.grafana.local`, puis se connecter avec ces valeurs.
Un fichier vide empêche le démarrage ; aucun mot de passe n'est livré dans Git.
Lors de la validation par l'agent, un fichier local a été créé avec un mot de
passe aléatoire ; consulter ce fichier dans son éditeur, sans le partager.
Ces variables initialisent uniquement une nouvelle base Grafana : modifier le
fichier ne réinitialise pas le mot de passe d'un utilisateur existant. Utiliser
ensuite la gestion des utilisateurs dans Grafana pour le changer, sans supprimer
le volume. Les accès anonymes, inscriptions et dashboards publics sont désactivés.
Ne pas publier `docker compose config` sans `-q`, qui afficherait les secrets.

```bash
docker compose config -q
docker compose up -d --wait --wait-timeout 660
docker compose logs --tail=100 grafana
```

Le healthcheck `/api/health` confirme le service et sa base ; le statut de la
source Prometheus se vérifie séparément. Si 3000 est occupé, résoudre le conflit
avant de démarrer, sans arrêter de service tiers.

Le provisioning en lecture seule est versionné sous `monitoring/grafana/` :

- `provisioning/datasources/prometheus.yml` : source par défaut, UID stable
  `supply-chain-prometheus`, accès proxy interne et intervalle minimum 15 s.
- `provisioning/dashboards/dashboards.yml` : dossier **Supply Chain MLOps**, lecture
  des fichiers JSON toutes les 30 s, sauvegarde UI désactivée.
- `dashboards/api-performance.json` : dashboard **API Performance**, UID
  `supply-chain-api`. Statut du scrape, requêtes actives, cumul depuis le démarrage,
  fraction d'erreurs HTTP 4xx/5xx, débits par route et statut, durées p50/p95/p99.
- `dashboards/model-serving-health.json` : dashboard **Model Serving Health**, UID
  `supply-chain-model`. Readiness, statut du scrape, cumul des requêtes POST
  `/predict`, fraction d'erreurs, débit et durées p50/p95/p99 d'inférence HTTP.

Ouvrir `http://127.0.0.1:3000/d/supply-chain-api` et
`http://127.0.0.1:3000/d/supply-chain-model` après connexion. Les durées sont en
secondes, les débits en requêtes/s et les fractions en pourcentage. Le cumul
compte des requêtes HTTP, tous statuts, pas des avis ni des prédictions
individuelles ; il repart de zéro au redémarrage de l'API. La latence inclut la
file d'attente et le traitement HTTP complet, pas uniquement le calcul du modèle.

La plage initiale est d'une heure et le rafraîchissement de 15 s. Les expressions
`$__rate_interval` s'adaptent au zoom, avec plusieurs scrapes pour calculer les
débits. Les valeurs absentes restent **No samples** ou des interruptions de courbe.
Une fraction d'erreurs nulle est déduite uniquement si du trafic est effectivement
observé ; sans trafic elle reste absente. Les quantiles sans observations ne
deviennent pas artificiellement une latence nulle. Le compteur de readiness
et `up` sont deux jauges distinctes : un scrape UP n'assure pas un modèle prêt.

Pour générer quelques observations de validation, répéter manuellement les avis
de référence avec un intervalle d'au moins 15 s, sans jeu de simulation :

```bash
curl --fail http://127.0.0.1:8080/health
curl --fail -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Livraison rapide et produit conforme","Commande jamais reçue","Service client très réactif"]}'
```

Les clusters attendus sont `[6, 7, 6]`. Laisser deux à quatre scrapes à Prometheus
pour que les courbes et quantiles aient des échantillons. En cas de panneaux
vides, vérifier le job `fastapi` UP dans Prometheus, la readiness, la plage de
temps et les observations de `/predict`, puis le statut de la source dans Grafana.
Les healthchecks seuls alimentent les panneaux API, pas les panneaux `/predict`.
Une absence de trafic ou une cible DOWN ne doit pas être corrigée en remplaçant
les valeurs manquantes par zéro.

Pour modifier les dashboards, éditer les JSON dans le dépôt puis attendre le
provisioning ou redémarrer Grafana. Garder les UID stables. Les modifications UI
ne sont pas sauvegardées sur les fichiers ; une copie de travail exportée de
Grafana doit être reportée dans les JSON versionnés. La suppression d'un fichier
provisionné supprime son dashboard géré. Le fonctionnement suit la
[documentation officielle de provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/).

```bash
# Tests de structure et régression hors ligne
MLFLOW_DISABLE_AGENT_HINT=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --frozen pytest tests -q
# Tests réels, stack explicitement démarré, credentials lus sans affichage
RUN_MONITORING_INTEGRATION=1 RUN_GRAFANA_INTEGRATION=1 \
  uv run --frozen pytest tests/integration -q
docker compose restart grafana
docker compose down
```

Les tests JSON vérifient les UID, métriques existantes et dispositions. Les tests
Grafana API vérifient le provisioning, la source accessible et les résultats
numériques des requêtes de tous les panneaux ; cela ne constitue pas une
inspection visuelle dans un navigateur. Le volume Grafana survit aux
redémarrages et à `down`, avec le même projet Compose ; `down --volumes` le
supprimerait, ainsi que la TSDB et le cache Hugging Face. Ce monitoring ne mesure
ni dérive des données ni qualité des clusters. Aucun Evidently, nouveau metric,
jeu de production simulé ou mécanisme d'alerte n'est ajouté.

## Simulation de production : client hôte

`src/simulate_production.py` lit uniquement le CSV choisi et envoie des lots
séquentiels à NGINX, sans charger de modèle ni écrire dans MLflow. Pour cette
phase, utiliser exclusivement `data/processed/avis_8k_production.csv` : 8 000
lignes, texte `commentaire`, identifiant stable dans la première colonne sans
nom (`--id-column ''`). Ni auteurs, titres, notes ni textes ne sont écrits dans
les résultats. Les jeux de référence et de test final ne sont pas utilisés.
Le client utilise uniquement la bibliothèque standard Python, sans dépendance
supplémentaire ni nouveau service Docker.

Un timeout récent au démarrage Docker n'a pas été diagnostiqué dans cette phase.
Le simulateur ne démarre pas le stack et ne résout pas cette panne. Le dry-run
et les tests hors ligne ne nécessitent aucune API disponible.

```bash
# Inspecter le plan d'un petit sous-ensemble, sans réseau et sans sortie sur disque
uv run --frozen python src/simulate_production.py --dry-run --limit 6 --batch-size 2

# Smoke test réel, seulement lorsque l'API est disponible
uv run --frozen python src/simulate_production.py \
  --limit 6 --batch-size 2 --delay 2 --run-id smoke-6

# Reprendre ce même run après interruption, avec les mêmes options
uv run --frozen python src/simulate_production.py \
  --limit 6 --batch-size 2 --delay 2 \
  --resume reports/production_simulation/smoke-6
```

La simulation complète n'est pas exécutée par défaut dans la validation de cette
phase. La commande suivante traite l'intégralité du fichier lorsqu'on décide
explicitement de la lancer ; **ne pas la lancer pour un simple smoke test** :

```bash
uv run --frozen python src/simulate_production.py \
  --batch-size 16 --delay 1 --run-id production-8k
```

Options principales (liste complète avec `--help`) :

| Option | Défaut / comportement |
|---|---|
| `--input` | `data/processed/avis_8k_production.csv` ; chemin d'une copie du même jeu si nécessaire |
| `--api-url` | `http://127.0.0.1:8080` ; aucun credential, query string ou fragment accepté |
| `--text-column`, `--id-column` | `commentaire`, nom vide ; identifiants lus comme chaînes, sans conversion numérique |
| `--batch-size` | 16 ; de 1 à 64, au plus 5 000 caractères par avis |
| `--delay` | 1 seconde entre lots ; pas de concurrence, fréquence approximative 1/(latence + délai) |
| `--limit` | Aucun ; limiter explicitement le nombre d'avis pour un smoke test |
| `--shuffle-seed` | Aucun ; ordre CSV par défaut, mélange déterministe avant application de la limite |
| `--timeout` | 60 secondes par tentative HTTP |
| `--retries`, `--backoff` | 3 tentatives supplémentaires, délais exponentiels 1/2/4 secondes par défaut, plafonnés à 60 s |
| `--model-id`, `--n-clusters` | Identifiant logique sélectionné `c93cb37fac6cbebb9f3cce20cb962722c0190a5e588afffbb80f6512eebafab0`, 10 clusters |
| `--output-root`, `--run-id` | `reports/production_simulation`, identifiant UUID automatique ou nom explicite non existant |
| `--resume` | Répertoire existant ; tous les paramètres de traitement doivent correspondre au run initial |
| `--dry-run` | Validation, empreintes et nombres d'avis/lots uniquement ; aucun réseau ni création de fichiers |

Le CSV entier est validé avant de choisir le sous-ensemble : colonnes requises,
identifiants uniques et non blancs, textes non blancs et limites API. Aucune
ligne n'est supprimée, aucun avis n'est tronqué. L'empreinte SHA-256 du fichier
complet est vérifiée avant et après lecture, et l'ordre des identifiants choisis
possède sa propre empreinte.

Avant d'envoyer des avis, le client exige un `/health` HTTP 200 avec
`model_ready: true`, avec le même budget borné de retries. Il n'interroge pas
indéfiniment une API en initialisation. Les erreurs réseau/timeouts et HTTP
408/429/500/502/503/504 sont réessayées ; les erreurs de validation 4xx restantes
ne le sont pas. Après épuisement des retries transitoires, le run s'arrête pour
éviter de poursuivre la charge sur une API indisponible. Un lot rejeté par une
erreur non transitoire reste enregistré en échec et le traitement peut continuer.
Une réponse incohérente, un cluster hors limites ou un identifiant modèle différent
arrête immédiatement le run, sans modèle de secours. Les redirections ne sont
pas suivies, afin de ne pas transmettre les avis à une autre destination.
Les corps d'erreur et messages d'exceptions réseau ne sont jamais journalisés.

Chaque nouveau run crée un répertoire privé de résultats, ignoré par Git et
refusé s'il existe déjà :

```text
reports/production_simulation/<run-id>/
  run.json                  # configuration immuable, empreintes, ordre et date UTC
  requests/health-*.json    # tentatives du contrôle de readiness, sans avis
  requests/batch-*.json     # checkpoint atomique de chaque lot et ses records
  sessions/*.json           # temps actif et état de chaque invocation/reprise
  summary.json              # résumé agrégé, remplacé atomiquement
```

Les checkpoints de lots contiennent `attempts` (timestamp UTC, latence HTTP,
statut HTTP ou null, catégorie d'erreur) et `records`, un objet par avis :

- `run_id`, `request_id`, `source_row_id`, `source_row_index` (index CSV à partir de zéro) ;
- `cluster_id` (null en échec), `model_id` attendu ;
- `request_timestamp`, `request_latency_seconds`, `attempt_count` ;
- `success`, `http_status` (null si aucun statut reçu), `error_category` (null en succès).

La latence enregistrée par avis est celle de la **dernière tentative HTTP du lot**,
répétée avec le même `request_id` pour chaque avis. Elle ne doit pas être interprétée
comme un temps de calcul individuel ni additionnée une fois par avis. Les autres
tentatives et leurs latences restent dans l'enveloppe du lot. En cas d'échec,
`model_id` désigne le modèle attendu, pas une affirmation sur le modèle distant.
La jointure ultérieure avec le CSV utilise `source_row_id` et l'empreinte du jeu ;
ne pas joindre simplement sur l'ordre après un mélange.

`summary.json` distingue avis tentés, réussis, échoués, en attente d'un checkpoint
terminal et pas encore tentés. Il conserve le nombre de requêtes HTTP physiques
(retries inclus), séparément pour readiness et prédiction, le temps actif cumulé
des sessions, le débit d'avis réussis, les latences des tentatives de prédiction
(moyenne, p50/p95 nearest-rank, maximum) et les effectifs par cluster. Le délai
entre deux invocations n'entre pas dans le débit ; ce n'est pas une moyenne de
latences individuelles. Le code de sortie est 1 en présence d'erreurs et 130 pour
une interruption clavier ; les détails structurés restent disponibles sur disque.

Pour reprendre, conserver le répertoire, le fichier CSV et les mêmes paramètres
(URL, modèle, taille de lot, sous-ensemble, seed, délais et retries). Les
checkpoints sont vérifiés avant toute requête et une instance concurrente du même
run est refusée par verrou POSIX. Les lots terminés, réussis **ou échoués**, ne sont
pas rejoués ; les échecs restent disponibles pour inspection. Les tentatives des
lots interrompus conservent leur budget de retries. Les lots ne sont pas engagés
ligne par ligne : écriture temporaire, fsync et remplacement atomique assurent
des fichiers JSON complets. Ne pas éditer ou supprimer des checkpoints pour
forcer des retries ; lancer un run distinct si une nouvelle campagne est voulue.

Une coupure entre la réponse HTTP et son checkpoint peut imposer de répéter la
requête au serveur lors de la reprise : le contrat API n'a pas de clé d'idempotence.
Cela ne duplique pas les succès déjà enregistrés dans les sorties. De même, une
requête interrompue avant son checkpoint peut ne pas figurer dans le compteur
local de requêtes physiques ; Prometheus reste la mesure côté serveur. Le résumé
est reconstruit à la reprise si un arrêt brutal a empêché sa mise à jour.

Les métriques et dashboards existants observent naturellement `/predict` via
NGINX ; aucun texte ni prédiction individuelle n'est envoyé à Prometheus. Pour
des panneaux vides, laisser plusieurs scrapes s'écouler et vérifier le trafic et
la cible UP. Si l'API est indisponible, consulter le diagnostic du stack dans une
tâche distincte, puis reprendre le run compatible. Cette phase ne tente ni de
réparer le timeout Docker ni de démarrer une simulation complète.

```bash
MLFLOW_DISABLE_AGENT_HINT=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --frozen pytest tests -q
```

Les tests du client emploient uniquement des CSV synthétiques et un transport
HTTP simulé. Aucun téléchargement d'encodeur, credential DagsHub ou service
Docker n'est nécessaire. Aucun Evidently n'est ajouté.

## Vérifications du pipeline hors ligne

```bash
MLFLOW_DISABLE_AGENT_HINT=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

Les tests utilisent un encodeur synthétique avec les vrais estimateurs et
MLflow : matrice complète de 20 configurations, absence de fit en inférence et
évaluation, intégrité du cache, identifiants, silhouette indéfinie, seed finale
et agrégation. Les tests d'authentification vérifient aussi le chargement du fichier local,
la priorité des exports, l'héritage des credentials sans copie du fichier et
le caractère strictement en lecture seule du contrôle de connexion.
Un test exécute deux seeds DVC dans des workspaces isolés, puis
restaure une expérience sous un nouveau nom. Il vérifie que l'encodeur n'est
chargé qu'une fois et que les références MLflow restent correctes. Aucun poids n'est
téléchargé ; ces tests ne mesurent pas la qualité sémantique des modèles réels.

## Phase 6D — Offline statistical monitoring

`src/monitor_drift.py` reads the reference and production CSVs directly for input
monitoring. It reads the simulator's `run.json` and `requests/batch-*.json` for
production cluster assignments. It does not contact the API, fit a model, change
infrastructure, or run the simulator. The independent final test set is excluded.
Only stripped text character length and whitespace-delimited word count enter
Evidently; review content, author names and ratings are never exported.

Install the locked environment with `uv sync --frozen`. Evidently is pinned to
**0.7.23**. This uses the current `Report`, `Dataset.from_pandas`, `DataDefinition`
and `ValueDrift` API; `Report.run()` returns a `Snapshot` saved with `save_html()`
and `save_json()`. See the [official report API](https://evidentlyai.github.io/evidently/api-reference/main/evidently/core/report.html)
and [drift configuration](https://docs.evidentlyai.com/metrics/customize_data_drift).

Input monitoring works before any simulator outputs are available:

```bash
uv run python src/monitor_drift.py report \
  --reference data/processed/avis_4k_reference.csv \
  --production data/processed/avis_8k_production.csv \
  --output reports/drift_monitoring/input-only-001
```

CSV defaults are `--text-column commentaire --id-column ''` (the existing
unnamed original row ID). IDs must be nonblank, unique within each dataset and
stable across CSV/prediction files. Reference and production IDs must be disjoint.
Rows are never silently dropped, deduplicated or truncated. Text must meet the
simulator's nonblank/5,000-character contract. Both CSVs are fingerprinted with
SHA-256; a change while reading/analysing invalidates the run.

### Reference predictions and model identity

Generate reference assignments explicitly, once, with the selected frozen model:

```bash
uv run python src/monitor_drift.py predict-reference \
  --selection config/selection.docker.json \
  --reference data/processed/avis_4k_reference.csv \
  --batch-size 32 \
  --output reports/drift_monitoring/reference-predictions.json
```

This reuses the existing verified MLflow inference loader, checks the manifest,
model SHA-256, logical ID, complete configuration and estimator parameters, then
runs CPU inference with the exact selected encoder revision and preprocessing.
No estimator or dimensionality reduction is fitted. The model and encoder must
be available through the selected tracking store/Hugging Face cache. The tracking
URI remains configurable through `SUPPLY_CHAIN_TRACKING_URI`; credentials come
from the existing environment loader, never from generated reports. The original
local selection file is untouched. Only this explicit reference-generation command
loads/downloads the model; the report command operates offline.

Reference artifacts have schema version 1, `kind: reference_predictions`, model
identity/configuration/checksum, dataset fingerprint/schema, creation time,
records SHA-256 and records containing `source_row_id`, zero-based
`source_row_index`, `model_id`, `success`, `cluster_id`. They contain no text.
Keep these generated artifacts trusted: their hashes detect accidental changes,
not malicious authors who can rewrite both records and provenance. Existing
pipeline assignment CSVs are not accepted without verified compatible provenance.

Once a real simulator run is available:

```bash
uv run python src/monitor_drift.py report \
  --reference-predictions reports/drift_monitoring/reference-predictions.json \
  --production-run reports/production_simulation/REAL_RUN_ID \
  --output reports/drift_monitoring/with-predictions-001
```

Both prediction inputs must be supplied together. Production assignments come
exclusively from recorded simulator checkpoints, never from new inference or
summary cluster counts. The replay plan (including shuffle, limit and batch
membership), input fingerprint, column mapping, model ID, cluster count, row IDs,
row indices and cluster ranges are checked. A cooperative simulator lock prevents
reading an active writer. The simulator records the logical ID returned/validated
by the API, but not an observed binary checksum: production checksum assurance
therefore depends on the existing verified API loader and trusted checkpoints.

By default, every CSV row must have exactly one successful prediction. Empty,
failed, pending, missing, duplicate or incompatible results are rejected. A limited
or interrupted run can be analysed only with explicit `--allow-subset`. This still
rejects incompatible IDs/models/configuration and malformed completed batches;
it reports recorded/successful/failed/missing sample counts and successful-ID
fingerprints. Prediction comparisons then describe those successful subsets,
which may suffer selection bias. Input monitoring always compares the complete
CSV cohorts. A prediction file is never filled with fabricated assignments.

### Statistical protocol and interpretation

The null hypothesis for each input feature is equality of the two population
distributions. A two-sample Kolmogorov–Smirnov statistic (maximum empirical CDF
gap, `KS_D`) is calibrated by independent-label permutation. This handles ties in
integer lengths/counts without assuming the continuous KS null distribution.
Raw-unit Wasserstein distance and count/mean/population standard deviation,
minimum, quartiles, median, p95 and maximum describe magnitude and shape.

For clusters, the null is equality of categorical assignment probabilities.
Pearson's two-sample homogeneity statistic is permutation-calibrated, including
sparse categories; jointly empty clusters contribute no statistic. This avoids
an unreliable asymptotic chi-square approximation when expected counts are low.
Every cluster, including absent clusters, has counts, proportions and
production-minus-reference proportion deltas. Total variation distance
(`0.5 * sum(abs(delta))`, range 0–1) measures overall redistribution.

All available inferential tests (two input tests and, when available, one cluster
test) form one Holm family at `--alpha .05`. The adjusted p-value is reported
separately from practical importance: default `--practical-threshold .1` means
`KS_D >= .1` for input features and total variation `>= .1` for clusters. These
are provisional investigation thresholds, not validated operational limits.
Use effect sizes and domain review before deciding on action.

The protocol uses SciPy's [independent permutation test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html),
`--resamples 1999 --seed 42`, with a greater-tail statistic (large discrepancy is
evidence against equality); small problems enumerate all distinct partitions.
Randomized p-values use the conservative +1 correction and have Monte Carlo
uncertainty and finite resolution (approximately 1/2,000 by default). Increasing
resamples improves resolution, not the validity of sampling assumptions. Large
samples can make tiny effects significant; small subsets have limited power.
Holm handles dependence between features, but the underlying permutation tests
assume exchangeable, independent observations. Repeated authors, duplicate text,
time dependence and sampling differences can violate that assumption. No authors
or timestamps are used to model dependence in this phase. Word count is a simple
whitespace feature, not token count or a semantic representation.

### Report outputs and limitations

Each new output directory contains:

- `results.json`: schema version, timestamp, dataset hashes/column mappings/sample
  sizes, dependency versions, test configuration, descriptive statistics, raw
  and Holm-adjusted p-values, effect sizes and separate significance/practical
  flags. Predictions include model provenance, artifact/checkpoint fingerprints,
  coverage and all cluster frequencies when available; otherwise their status is
  explicitly `unavailable` with a reason. Embeddings are explicitly `deferred`.
- `input.html` and `input.json`: Evidently distributions and reference-standardized
  Wasserstein distances, using its explicit 0.1 distance threshold.
- `predictions.html` and `predictions.json`, only with verified prediction inputs:
  Evidently categorical distributions and Jensen–Shannon distances with a 0.1
  distance threshold. These are descriptive supplementary metrics. Evidently's
  threshold flags are distinct from the permutation/Holm and practical decisions
  in `results.json`; those JSON conclusions are authoritative for this protocol.

Outputs under `reports/drift_monitoring/` are ignored by Git. Existing report
folders and reference prediction files are never overwritten. A failed run can
leave a partial directory; absence of `results.json` means no completed report.
Statistical results are deterministic for the same inputs, seed and locked
versions; generation timestamps and HTML identifiers are not byte-deterministic.

Embedding drift is deferred to a separate phase: generating pinned embeddings
for both cohorts and validating a reference-only projection plus a calibrated
multivariate comparison would add substantial compute and methodological scope.
Independent tests of hundreds of coordinates would not establish semantic drift.
This phase does not evaluate clustering quality, semantic correctness or business
performance. Drift can prompt investigation; it is not evidence of degradation.

Offline verification: `uv run pytest tests/test_monitor_drift.py -q` uses only
synthetic CSVs and a mocked frozen predictor/simulator transport. It covers input
validation, provenance/alignment, incomplete runs, deterministic statistical
results, identical/shifted distributions, sparse clusters and real HTML exports.

## Phase 7A — Frozen independent final evaluation

`src/evaluate_final.py` audits and seals a final protocol before any inference.
Checkpoint **7A-1 requires explicit approval** before `--execute`. The file named
`avis_3k_test.csv` actually contains **2,261** rows in this checkout. The owner
explicitly authorized preparing the protocol for those 2,261 observations; no
rows are added, removed or represented as a 3,000-observation evaluation.

```bash
# Audit only: validates CSV/IDs and fingerprints; never loads/encodes the model.
uv run python src/evaluate_final.py --audit-only \
  --input data/processed/avis_3k_test.csv --expected-rows 2261 \
  --output reports/final_evaluation/checkpoint_7a1_20261116_ready

# AFTER explicit checkpoint approval only; this command has not yet been run:
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  uv run python src/evaluate_final.py --execute \
  --protocol reports/final_evaluation/checkpoint_7a1_20261116_ready/protocol.json \
  --batch-size 32 --output reports/final_evaluation/final_20261116
```

Existing output directories are never overwritten. The audit creates
`protocol.json` and `seal.json` with its SHA-256. Changing any protocol field,
selection, dataset, audited cohort, inference/metric source code or software version invalidates execution. The protocol seals source and uv.lock hashes plus package versions. Before inference,
execution exclusively creates `execution_claim.json` beside the approved protocol,
reserving it across output directories. An interrupted/failed claim is retained:
stop and inspect rather than deleting it or silently rerunning the final test.
The program cannot enforce human approval: the operator must obtain approval
before invoking `--execute`. Batch size affects resource use, not metric sampling.
Create a fresh audit only if a changed input is intentionally reviewed and approved;
never use new protocols to repeatedly evaluate or tune on the final results.

### Independence evidence and limits

The preflight checks unique, nonblank final IDs and disjointness against the actual
20,000 training IDs, 4,000 selection-evaluation IDs, 4,000 reference IDs and 8,000
production IDs. It seals their file hashes, checks that the local original selection
and Docker selection agree on logical ID, configuration and binary checksum, and
retains the historical selection rule/ranking only as context. It also reports how many final rows share exactly the same stripped text with each cohort, without persisting text or dropping rows. Repeated generic reviews can match across different IDs; that overlap limits independence without proving reuse of the same observation. Current pipeline
inputs and prior monitoring configuration exclude the final file; drift thresholds
were fixed in phase 6D, before accessing final identifiers.

This establishes current row-ID disjointness, not a complete proof of all historical
use: there is no immutable complete usage ledger. Unrecorded earlier use, repeated
review text across distinct IDs and dependence across authors/time cannot be ruled
out. The selected training-cache ID was also reconstructed from the current training CSV SHA-256, encoder, preprocessing and column specification; it matches the selected configuration. This binds the declared selected training input to the audited CSV. It does not prove all historical uses of data or that nobody inspected test data earlier. This limitation must be acknowledged
before approval, not hidden in the eventual final scores.

### Frozen inference and metrics

The workflow reuses `api.main.load_selected_model`: MLflow manifest/run selection,
binary SHA-256, logical configuration digest, pinned encoder revision, preprocessing
and estimator hyperparameters are checked before final prediction. The actual
estimator must be `SphericalKMeans` with the selected number of unit centroids.
`ExperimentPredictor.predict_texts(..., return_embeddings=True)` optionally returns
the **same normalized embeddings** used by the existing prediction path. Default
API behavior still returns labels only. No parallel encoder implementation, refit,
transform fitting, keyword naming, semantic annotation or model selection occurs.
Original centroid bytes are checked for mutation after inference.

The sealed metrics are:

- **Exact cosine silhouette**, using all final observations and frozen predicted
  labels, via the existing chunked `held_out_silhouette` helper (32 MiB working
  memory; sample size 0). Silhouette compares mean pairwise intra-/nearest-other
  cluster distances; it is not distance to the trained centroid. It is undefined
  for fewer than two represented clusters or one cluster per observation, with
  explicit null/reason. Scikit-learn assigns singleton-cluster silhouettes zero.
  No Euclidean silhouette or Euclidean Davies–Bouldin score is mixed into this protocol.
- **Cluster population**, all configured IDs including empty clusters: counts,
  proportions, represented/empty IDs, largest proportion, largest/smallest nonempty
  count ratio. Imbalance is descriptive and is not assumed undesirable.
- **Frozen-centroid geometry**: assigned similarity `x dot c`, dispersion
  `1 - x dot c`, assignment margin `x dot c_assigned - max(x dot c_other)`, and
  pairwise original-centroid cosine distances. Unit/finite vectors, cluster ranges
  and agreement with original-centroid argmax are required. Empty cluster summaries
  are explicitly unavailable; singleton summaries are valid descriptive values.
  No centroid is recomputed from final data or used to redefine predictions.
- **Previous metrics**, available only with `--previous-result` during audit and
  verified matching logical ID, complete configuration, binary checksum, selection
  cohort fingerprint and exact cosine sampling protocol/sample count. Different
  cohorts remain different populations: a score difference is descriptive, not
  evidence of quality change. No verified individual selected-seed result was
  found locally at preflight. The five-seed selection mean/ARI remains historical
  context, never substituted for that missing individual result or training quality.
- **Uncertainty** is explicitly not estimated. Deterministic repeated predictions
  do not estimate training stability; no model is refitted. A dependence-aware
  resampling design would require separate justification. Geometric separation
  does not establish semantic correctness, topic usefulness or business value.

### Artifacts and reproducibility

Outputs are ignored under `reports/final_evaluation/`, with private run-directory
permissions. Schema version 1 artifacts include:

- `predictions.csv`: source ID, zero-based source row index, logical model ID,
  success flag, cluster ID, assigned centroid similarity/distance and margin;
  no raw review text or author metadata.
- `embeddings.npy`, `original_centroids.npy`: exact numeric arrays used for metrics,
  saved without pickle, for reproducibility. These are local data artifacts and
  must not be published indiscriminately.
- `metrics.json`: exact cosine silhouette/sample fingerprint and explicit reasons
  when undefined, cluster statistics, original centroid distances, descriptive
  geometry, comparison availability and uncertainty status.
- `protocol.json`, `metadata.json`: dataset/model/configuration fingerprints,
  software versions, full coverage, centroid hash, artifact SHA-256 hashes, runtime
  and Git revision. `execution_state.json` marks started/complete. Only a complete
  state with metadata constitutes a successful evaluation; partial outputs survive
  failures for diagnosis and must not be reported as completed results.
- `figures/`: cluster populations, assigned-centroid similarity boxplots and margin
  histogram, exported as 300-dpi PNG and vector SVG with fixed layouts. A comparison
  figure is omitted when no compatible historical individual result is available.

The exact dataset and selected model stay frozen after evaluation. Do not change
hyperparameters, thresholds, metrics or the model based on final results. Tests in
`tests/test_evaluate_final.py` use synthetic vectors/CSVs only, with mocked inference
and explicit checks against fitting or centroid mutation:

```bash
uv run pytest tests/test_evaluate_final.py -q
uv run pytest tests -q
```
