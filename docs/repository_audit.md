# Repository consolidation audit — Checkpoint 8B

Audit date: 17 November 2026. Branch: `rebuild_step8_finalization`; starting HEAD:
`5308743`. No branch switch, commit, push, merge, tag, model change, retraining,
large workload, image rebuild, volume deletion or Git history cleanup was performed.

## History and working-tree scope

The current history contains the prior implementation chain:

| Phase | Commit |
|---|---|
| DVC experimentation | `a32de55` |
| FastAPI | `3953756` |
| DVC test correction | `4fcd9c8` |
| Docker / remote selected artifacts | `2103921` |
| NGINX | `880efc3` |
| Prometheus | `2d11ace` |
| Grafana | `978f256` |
| Production simulation | `da3c6d2` |
| Evidently | `5239471` |
| Integrated validation | `9287ecd` |
| Final evaluation | `70841a6` |
| Semantic inspection | `5308743` |

Initial untracked items were local `AGENTS.md` instructions, `data/.gitignore`,
`metrics/`, `mlruns/` and `reports/`. The first two are configuration/instructions
and remain untouched, outside the proposed documentation commit. The other three
are generated/private runtime artifacts and are now ignored. No uncommitted
application source or tests were found.

Changes are confined to the README, this documentation directory, `.gitignore`
and saved outputs/execution counters in the two historical notebooks. Notebook
cell sources are preserved. Training/API source, manifests, dependencies, DVC
configuration, parameters and results are unchanged.

## Privacy and storage

- Added root ignores for `mlruns/`, `metrics/`, `reports/` and SQLite sidecars.
  Existing rules already exclude private dotenv files, review CSVs, model outputs,
  virtual environments and DVC caches. Blank credential templates remain tracked.
- The local annotation CSV includes customer text and is protected by `reports/`.
  Only verified aggregates are documented. No individual themes are published.
- Removed 49 saved notebook output objects and reset execution counters: historical
  data previews could contain customer fields even when full-text matching finds
  no exact review. No notebook cells or source analyses were removed.
- No tracked CSV, model binary or local database was found in the current tree.
  A bounded scan of tracked contents found no configured nonempty credential value
  or selected common access-token patterns. This is not a comprehensive security
  or full-history certification.
- Before ignore consolidation, `mlruns/` held 120 files totaling approximately
  6.80 GB, `reports/` 702 files totaling 30.63 MB and `metrics/` one 560-byte JSON.
  Files remain on disk; none were deleted or staged.
- Git reports approximately 982 MiB of packed storage and 17 temporary pack files
  totaling approximately 7.94 GiB. No garbage collection or history rewrite was
  attempted. Removing current notebook outputs does **not** remove older outputs
  from Git history. A separate history/privacy review is needed before treating
  historical publication as cleared; destructive remediation requires approval.

## Reproducibility and runtime limits

`uv.lock` resolves 338 packages; 328 installed packages passed compatibility checks.
The frozen offline sync succeeded against existing caches, not a clean machine.
Installed host versions were FastAPI 0.143.0, Uvicorn 0.54.0, MLflow 3.17.0,
DVC 3.67.1, Evidently 0.7.23, SentenceTransformers 6.1.0, scikit-learn 1.9.1,
NumPy 2.5.4, SciPy 1.18.1 and pytest 9.2.0. These describe the audited host
environment, not an independently rebuilt image.
Docker/Compose and the model selection reference were inspected without changes.
Credentials are supplied at runtime; the remote reference retains the approved
model identity/checksum. The read-only MLflow check succeeded without creating a run.

DVC status is **not clean**. MPNet embedding dependencies changed and its cached
output is absent; MiniLM, training, prediction and evaluation stages also report
changed source or lock dependencies. No DVC reproduction, pointer update or data
retrieval was performed. Serving artifact provenance is separate from this stale
workspace state; a fresh full training reconstruction is not claimed.

The existing four-service stack was healthy and retained its volumes. Runtime
`src/api/main.py` matches the host source. Runtime `src/predict.py` differs from
HEAD, which gained later optional offline embedding-return support. Live endpoint
checks validate the running image; they do not validate rebuilding the exact
current checkout. No rebuild was necessary for the documentation audit or attempted.

The annotation file was edited by its human annotator after sample generation.
Its whole-file checksum therefore differs from the initial sampling manifest;
no resampling or overwrite was performed. This does not alter frozen Phase 7A
metrics. The owner subsequently clarified that the ten populated coherence cells
are collective judgments for the ten cluster samples and cover all 100 sampled
reviews. Their physical CSV positions span nine row-level cluster IDs (two cells
on cluster 6 rows, none on cluster 7 rows); those positions are not interpreted as
per-review coverage. Documentation follows the clarified cluster-level annotation
protocol, without moving cells, assigning specific ratings to cluster IDs or
inventing 100 independent per-review ratings.

## Executed checks

| Check | Result |
|---|---|
| Frozen offline dependency sync / lock check | Passed |
| Installed dependency compatibility | Passed |
| Full offline pytest suite | 167 passed, 13 skipped, 12 warnings; 25.02 s |
| Existing-stack integration pytest suite | 13 passed; 19.69 s |
| Compose configuration / existing service health | Passed; four healthy services |
| NGINX syntax / Prometheus configuration | Passed |
| Read-only MLflow access | Passed; no run created |
| Experiment and simulator dry runs | Passed; no workload executed |
| DVC status | Inspected; stale/missing dependencies and output as above |
| Git whitespace, local Markdown links and notebook source preservation | Checked during final consolidation |

The 12 warnings concern upstream Litestar dependency deprecations exercised by
Evidently. Optional integration tests were skipped in the offline run, then all
executed successfully with explicit flags. No dependency upgrade was introduced.
Official technical reference pages were consulted; external link availability is
not a permanent guarantee. Local links and anchors are checked independently.
No dedicated Markdown linter was installed; a standard-library checker verified
11 local links/anchors, balanced fenced blocks and the absence of local absolute
paths in the new English documentation. Detailed test logs, DVC status and version
metadata remain ignored under `reports/repository_consolidation/20261117/`.

## Proposed Git actions — not executed

Review the diff, including notebook output removal, before staging this exact
allowlist. It deliberately excludes private artifacts and the two pre-existing
untracked configuration files:

```bash
git add .gitignore README.md docs/reproduction.md docs/final_validation.md docs/repository_audit.md notebooks/01_exploration.ipynb notebooks/02_experimentation.ipynb
git diff --cached --check
git diff --cached --stat
git commit -m "docs: consolidate final project documentation and privacy hygiene"
git push -u origin rebuild_step8_finalization
```

The integration destination has not been approved. After explicit approval and
switching to that approved destination, the proposed merge command is:

```bash
git merge --no-ff rebuild_step8_finalization
```

No destination is inferred from historical `main` or `feat/grafana`. Checkpoint 8B
stops before any of these Git mutations.
