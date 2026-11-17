# Reproduction and command verification

Run commands from the repository root. Private inputs, remote permissions and
credentials are prerequisites; this public checkout does not include review data.
The Phase 8 audit reused an installed environment and an already running stack.
It did not replay training, production simulation or the final evaluation.

## Environment and private data

Install Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/),
then use the committed lock:

```bash
uv sync --frozen
```

The network-install variant above is a recipe, not a clean-machine validation.
The audit successfully ran `uv sync --frozen --offline`, `uv lock --check --offline`
and `uv pip check` against the existing cache/environment.

Retrieve private data only with access to the existing configured DVC remote:

```bash
uv run --frozen dvc pull data/train_eval.dvc data/processed.dvc
```

This retrieval was **not executed in Phase 8**. Do not replace remote URLs or
assume public data access. `dvc status --json` was executed: several stages have
changed dependencies and the MPNet embedding output is absent. The current DVC
workspace is not a clean reproduction of every historical stage. Do not run
`dvc repro` simply to clear this warning: it can retrain or overwrite outputs.

## MLflow and host API

Prepare credentials locally, without inserting them in tracked manifests:

```bash
cp -n .env.example .env.dagshub.local
chmod 600 .env.dagshub.local
```

Fill the blank tracking URI, username and token/password privately. Use the known
tracking store from `config/selection.docker.json`; retain the selected run and
checksum. The read-only access check was executed successfully:

```bash
uv run --frozen python src/tracking.py --check
```

For host serving, explicitly select the portable DagsHub reference instead of a
machine-specific local SQLite selection:

```bash
SUPPLY_CHAIN_SELECTION_PATH=config/selection.docker.json \
  uv run --frozen uvicorn api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

This host startup command was not re-executed in Phase 8. Credentials are loaded
through the environment or the ignored tracking environment file; an optional
`SUPPLY_CHAIN_TRACKING_URI` overrides the tracking store. Never embed a token in a
URI. The model loader checks the selected artifact's identity and SHA-256.

## Docker Compose

For an initial setup, copy the blank templates and fill credentials privately:

```bash
cp -n docker.env.example .env.docker.local
cp -n grafana.env.example .env.grafana.local
chmod 600 .env.docker.local .env.grafana.local
```

The Grafana variables initialize a new database; they do not reset an existing
administrator. Container `env_file` settings and Compose interpolation are
separate. Set `SUPPLY_CHAIN_SELECTION_FILE` in the shell or a Compose `--env-file`
to change the host selection path, keeping it consistent with the container path.

```bash
docker compose config -q
# Initial build only; not performed during Phase 8:
docker compose build api
# Reuse existing images and preserve all volumes:
docker compose up -d --no-build --pull never
docker compose ps
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/model/info
```

Only configuration validation, status inspection and existing-service checks were
performed in Phase 8; neither build nor startup was replayed. Cold encoder
acquisition previously exceeded 17 minutes and the old readiness wait budget.
Inspect startup logs and readiness rather than assuming a short cold-start time.
Do not remove volumes: they preserve the Hugging Face cache and monitoring state.

Validated configuration checks:

```bash
docker compose exec -T nginx nginx -t
docker compose exec -T prometheus promtool check config /etc/prometheus/prometheus.yml
```

## Tests

These exact test invocations succeeded during Phase 8:

```bash
MLFLOW_DISABLE_AGENT_HINT=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests -q

RUN_MONITORING_INTEGRATION=1 RUN_GRAFANA_INTEGRATION=1 \
  MLFLOW_DISABLE_AGENT_HINT=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/integration -q
```

Results: **167 passed, 13 skipped, 12 dependency deprecation warnings** in 25.02 s;
then **13 integration tests passed** in 19.69 s. Integration requires the existing
healthy stack and private Grafana credentials. It validates routing, model control
predictions, metrics, datasource/dashboard APIs and queries; it is not a load test
or a visual browser inspection.

## Experimentation and monitoring

Safe planning commands successfully executed in Phase 8:

```bash
.venv/bin/python src/run_experiments.py --dry-run
.venv/bin/python src/simulate_production.py --dry-run --limit 6 --batch-size 2
```

The following are historical-workflow recipes, **not executed again in Phase 8**.
They produce private artifacts and require an approved output location. Reference
prediction performs inference; replay sends live traffic. Reuse completed outputs
where available and use the simulator's `--resume` option for an interrupted run.

```bash
.venv/bin/python src/monitor_drift.py predict-reference \
  --selection config/selection.docker.json \
  --output reports/drift_monitoring/new-reference.json

.venv/bin/python src/simulate_production.py \
  --batch-size 16 --delay 1 --run-id new-production

.venv/bin/python src/monitor_drift.py report \
  --reference-predictions reports/drift_monitoring/new-reference.json \
  --production-run reports/production_simulation/new-production \
  --output reports/drift_monitoring/new-report
```

Verify that the supplied prediction paths match the actual completed run layout
before using these recipes. The successful Phase 6E outputs remain local under
`reports/`; their aggregate results are in [final validation](final_validation.md).
No fresh workload or report generation is claimed here.

## Semantic inspection

Phase 7A is sealed and has already been executed once. **Do not execute it again**
for documentation or resample in place over the annotated CSV. The existing Phase
7B sample and guide are local under
`reports/final_evaluation/semantic_inspection_20261116/`.

The sampler's existing reproducibility entry point is:

```bash
.venv/bin/python src/sample_semantic_inspection.py --seed 42 \
  --output reports/final_evaluation/semantic_inspection_new
```

This recipe was not replayed in Phase 8. It uses frozen per-review geometry rather
than generating embeddings. A new output directory is necessary to preserve human
annotations. Select four high-similarity reviews, three nearest the cluster median
similarity and three remaining lowest-margin reviews per cluster. Seed 42 resolves
ties; identifiers and stripped texts must be distinct within each cluster. The
original 100-row sample met all quotas, skipping four duplicate-text candidates.
This targeted sample cannot estimate global semantic accuracy.
