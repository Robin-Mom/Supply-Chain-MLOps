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

Prérequis : Docker Engine avec BuildKit et Docker Compose v2+, accès au moteur
Docker, accès réseau aux registres pendant le build et assez d'espace pour les
dépendances ML verrouillées. Aucun GPU ni environnement Python hôte n'est requis.

```bash
cp docker.env.example .env.docker.local
chmod 600 .env.docker.local
# Renseigner les credentials existants dans ce fichier local, sans les afficher.
docker compose config -q
docker compose build api
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
avec `no-new-privileges`. Le port hôte est limité à `127.0.0.1:8000`.

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
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/model/info
curl --fail -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Livraison rapide et produit conforme","Commande jamais reçue","Service client très réactif"]}'
docker compose logs --tail=100 api
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

Les tests Python restent exécutables avec la commande de la section précédente ;
pytest n'est pas installé dans l'image de production. La syntaxe Compose,
le build et les appels réels ci-dessus constituent les vérifications Docker.
La construction suit le [guide Docker de uv](https://docs.astral.sh/uv/guides/integration/docker/)
et les montages la [référence Compose](https://docs.docker.com/reference/compose-file/services/).

## Vérifications hors ligne

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
