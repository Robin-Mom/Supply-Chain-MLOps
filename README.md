# Supply Chain MLOps — local clustering experiments

This branch, `rebuild_step2_scripting`, rebuilds review clustering with Python,
DVC, and MLflow. `main` and `feat/grafana` contain historical implementations.
The current scope is local training, evaluation, and model selection using
KMeans and Spherical KMeans. Git and DVC remotes are preserved as configured.

## Run locally

```bash
uv sync --frozen
source .venv/bin/activate
# If the input data is missing, retrieve it from the existing configured remote:
dvc pull data/processed.dvc
dvc repro
dvc metrics show
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Set the CSV path and text column in `params.yaml`. The default input is
`data/processed/avis_24k_train.csv`, with a `commentaire` column. Blank reviews are
rejected rather than silently dropped. Data access uses the existing DVC
configuration and any credentials you supply separately.

## Pipeline

1. `train`: encode reviews once with the pretrained multilingual encoder,
   normalize embeddings, and fit both clustering methods. Save the encoder,
   estimators, assignments, configuration, package versions, and input hash in
   `models/clustering`. Log parameters, metrics, and artifacts to local MLflow.
2. `evaluate`: compute cosine silhouette and Davies–Bouldin scores, cluster
   sizes, French c-TF-IDF keywords, and optional stability analyses. Save reports
   in `reports/clustering`, consolidate DVC metrics in `metrics/clustering.json`,
   and append evaluation results to the original MLflow runs.
3. `select`: rank the evaluated models using the declared metric and direction.
   Write `reports/selection.json` with the chosen method, ranking, artifact
   paths, and hashes of the model, metrics, training metadata, and training data.
   This records an experiment decision; it does not deploy a model.

The default selection maximizes cosine silhouette on training embeddings.
This is an internal geometry comparison, not evidence of generalization or
semantic quality. Undefined scores are excluded; selection fails if no valid
score exists. Equal scores are resolved alphabetically by method name.
Use `selection.metric: davies_bouldin` with `selection.direction: minimize`
when selecting by Davies–Bouldin instead.

## Reproducibility and experiments

`params.yaml` controls cluster count, initialization count, iteration limit,
random seeds, encoder, evaluation sampling, selection rule, and tracking URI.
The DVC command explicitly sets `train.tracking_uri` to `sqlite:///mlflow.db`
by default, so an inherited remote tracking environment does not redirect it.

```bash
dvc exp run -S train.n_clusters=15
dvc exp show
dvc metrics diff
```

For stronger encoder reproducibility, set `train.extra_args` to
`"--revision <encoder-commit>"` using a revision you have verified. An unpinned
encoder identifier may resolve to different weights over time. Fixed seeds
make clustering and metric sampling repeatable within the same environment;
results may vary across numerical libraries or hardware. `uv.lock` is a DVC
stage dependency, and the commands above use its locked environment.

Modify evaluation parameters to rerun evaluation and selection; modify training
parameters to rerun the full pipeline. For optional stability checks, append
`--stability statistical algorithmic global --plot` to `evaluate.extra_args`.
Keep `--log-mlflow` there to log evaluation; append `--skip-keywords` to skip
keyword extraction. Additional JSON hyperparameter overrides are supported by
`train.py --config`; if used in DVC, declare that JSON file as a stage dependency.

DVC removes managed outputs before recomputation. Do not place unrelated files
in the stage output directories. Standalone scripts deliberately reject existing
output paths. Commit `dvc.yaml`, `params.yaml`, the generated `dvc.lock`, and
`metrics/clustering.json` after a real run. `dvc push` sends cached outputs to
the already configured remote; it does not upload the local MLflow database.

DVC cache restoration does not replay MLflow logging. Preserve `mlflow.db` and
`mlartifacts/` together to retain local runs. To regenerate runs after removing
that store, use `dvc repro --force --no-run-cache`. The tracking database and
MLflow artifacts are local and ignored by Git.

## Standalone scripts and optional remote tracking

Run `python src/train.py --help`, `python src/evaluate.py --help`,
`python src/select_model.py --help`, or `python src/predict.py --help` for options.
Prediction uses the saved local encoder and clustering bundle without refitting.

Standalone training accepts `--tracking-uri` or `MLFLOW_TRACKING_URI`. For a
remote service, supply its exact URI and credentials yourself; no owner, URL,
or credential is derived from legacy configuration. `.env.example` contains
empty optional environment settings. The scripts do not load `.env`
automatically. For DVC, set the explicit URI in `train.tracking_uri` and export
credentials in your shell. Never commit credentials. URI credentials are
rejected by the tracking helper. Changing tracking configuration requires a
fresh run to produce new MLflow run references.

## Offline checks

```bash
MLFLOW_DISABLE_AGENT_HINT=1 python -m unittest discover -s tests -v
```

The integration test uses synthetic encoder output with real clustering,
serialization, evaluation, model selection, and a temporary local MLflow store.
It requires no model download or remote service. It does not validate semantic
quality or the full pretrained encoder run.
