# Supply Chain MLOps — expériences de clustering locales

La branche `rebuild_step2_scripting` utilise Python, DVC Experiments et MLflow
pour comparer trois SentenceTransformers, KMeans et Spherical KMeans sur cinq
seeds. Les configurations Git/DVC distantes et les infrastructures historiques
ne sont pas modifiées.

## Préparation

```bash
uv sync --frozen
source .venv/bin/activate
# Seulement si les deux CSV sont absents :
dvc pull data/train_eval.dvc

# Le chemin absolu reste valide dans les workspaces temporaires DVC.
export SUPPLY_CHAIN_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"
export MLFLOW_TRACKING_URI="$SUPPLY_CHAIN_TRACKING_URI"
# Facultatif : limiter les threads BLAS/OpenMP selon la machine.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

`params.yaml` contient les hyperparamètres, cinq seeds et la seed finale fixée à
42 avant comparaison. Les trois encodeurs sont figés à des commits Hugging Face.
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
# Afficher les 30 commandes sans entraîner de modèle.
python src/run_experiments.py --dry-run

# Exécuter séquentiellement les 30 expériences, puis la sélection globale.
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

L'orchestrateur cible `evaluate` : DVC ne calcule que les dépendances nécessaires
à l'encodeur actif. Pour une expérience individuelle :

```bash
SUPPLY_CHAIN_DVC_NAME=essai-minilm-43 dvc exp run evaluate \
  --name essai-minilm-43 \
  -S experiment.encoder=minilm \
  -S experiment.method=spherical_kmeans \
  -S experiment.seed=43

dvc exp show
dvc metrics show
dvc exp diff
mlflow ui --backend-store-uri "$SUPPLY_CHAIN_TRACKING_URI"
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
   réutilise ces six matrices entre méthodes et seeds. La VRAM est libérée à la
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
   rechargé pendant les 30 inférences utilisant le cache.
4. **`evaluate`** lit ces prédictions et le cache eval, calcule la silhouette
   cosine et crée un nouveau run `Supply_chain_eval` avec
   `train_mlflow_run_id`. Les prédictions ne sont journalisées que dans ce run,
   sans copie du modèle ou du transformer.

Les sorties de chaque configuration sont relatives au workspace, sous
`models/runs/`, `models/predictions/` et `models/evaluations/`, puis organisées par
encodeur/méthode/seed. DVC gère leur remplacement et leur restauration. Ne pas y
placer de fichiers indépendants du pipeline.

Avec SQLite, l'URI doit être absolue. Les nouvelles expériences MLflow stockent
leurs artefacts sous `mlartifacts/<nom-expérience>` à côté de la base, hors des
workspaces temporaires. Une expérience MLflow déjà existante conserve son
emplacement d'artefacts. Pour un serveur distant, fournir son URI exacte dans
`tracking.uri` ou `SUPPLY_CHAIN_TRACKING_URI` et exporter les identifiants séparément.
L'environnement prend priorité sur `tracking.uri`. Aucun secret n'est écrit
dans les fichiers versionnés ; les URI contenant des identifiants sont refusées.

Le cache DVC ne rejoue pas les écritures MLflow. Préserver ensemble la base et
ses artefacts ; `dvc push` ne les sauvegarde pas. Changer l'URI d'environnement
n'invalide pas le cache DVC : si le stockage MLflow a été supprimé ou déplacé,
restaurer ce stockage ou forcer les stages concernés. Ne pas utiliser `--force`
sur les embeddings pour simplement régénérer les runs MLflow : reproduire
séquentiellement `train`, `predict`, puis `evaluate` avec `--single-item --force`.

## Évaluation et sélection

La silhouette est exacte sur les 4 000 avis par défaut (`sample_size: 0`).
Scikit-learn calcule les distances par blocs, avec `working_memory_mb: 32` pour
limiter leur mémoire de travail. Une seule configuration est évaluée à la fois.
Ce budget concerne les distances, pas la mémoire totale du processus.

Si `sample_size` est réduit, les positions sont tirées avec `sample_seed`, triées
et identiques pour tous les encodeurs, méthodes et seeds. Leur empreinte est
journalisée. Une silhouette impossible est enregistrée à `null` avec sa raison.

La sélection est extérieure au DAG de chaque expérience :

```bash
# Déjà appelée automatiquement après les 30 expériences ; commande indépendante :
python src/select_best.py --output reports/selection_alternative
```

Elle recherche dans `Supply_chain_eval` le sweep correspondant aux données,
paramètres, sources et lock actuels. Elle vérifie les 30 configurations finies,
leurs liens aux runs train et l'alignement des observations. En cas de répétition
d'une configuration dans un sweep, le dernier run terminé est retenu.
Elle calcule, pour chacun des six groupes :

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

Seules les transformations effectivement sauvegardées peuvent être utilisées.
Le pipeline actuel n'en ajuste aucune en plus de la normalisation L2 ; un
manifeste annonçant des transformations inconnues est rejeté.

Les anciennes interfaces CLI et fonctions de `train.py`, `predict.py`,
`evaluate.py` et `select_model.py` sont conservées pour les bundles historiques.
Elles gardent leur fonctionnement historique, y compris les analyses
optionnelles de stabilité sur les données train. **Le nouveau protocole utilise
`--params params.yaml` et `select_best.py`** ; les anciennes commandes ne doivent
pas servir à alimenter sa comparaison des 30 expériences.

## Vérifications hors ligne

```bash
MLFLOW_DISABLE_AGENT_HINT=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

Les tests utilisent un encodeur synthétique avec les vrais estimateurs et
MLflow : matrice complète de 30 configurations, absence de fit en inférence et
évaluation, intégrité du cache, identifiants, silhouette indéfinie, seed finale
et agrégation. Un test exécute deux seeds DVC dans des workspaces isolés, puis
restaure une expérience sous un nouveau nom. Il vérifie que l'encodeur n'est
chargé qu'une fois et que les références MLflow restent correctes. Aucun poids n'est
téléchargé ; ces tests ne mesurent pas la qualité sémantique des modèles réels.
