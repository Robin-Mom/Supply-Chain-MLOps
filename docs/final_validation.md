# Final validation evidence

This aggregate record consolidates the completed Phase 6E, 7A and 7B evidence from
16 November 2026 and the repository audit of 17 November 2026. Raw reviews,
individual annotations and generated reports remain local and ignored by Git.
No training, final evaluation or full production replay was repeated in Phase 8.

## Frozen model

- Encoder: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
- Estimator: Spherical KMeans, 10 clusters, seed 42.
- DagsHub training run: `c650a6675bf5438c81d90b96ee4e154c`.
- Logical identity: `c93cb37fac6cbebb9f3cce20cb962722c0190a5e588afffbb80f6512eebafab0`.
- Artifact SHA-256: `0576bf251a3b4daad93598bc971669d32b18e71d9fd5a9b1b98034486ca3f4af`.

Selection compared 20 configurations across two encoders, two clustering methods
and five seeds, using held-out geometry and cross-seed stability. The selected
configuration's mean cosine silhouette was 0.117329. This aggregate selection
score is not a paired individual-run baseline for final-test inference. No refit
on the combined training/selection set was performed.

## Controlled operational validation

| Measure | Verified Phase 6E result |
|---|---:|
| Reference coverage | 4,000/4,000 |
| Production coverage | 8,000/8,000 |
| Production prediction batches | 500, sequential, 16 reviews each |
| Failed requests / retries | 0 / 0 |
| Replay elapsed time | 771.28 s |
| Throughput including configured pauses | 10.37 reviews/s |
| Mean / p95 client batch latency | 0.541 / 0.718 s |
| Observable prediction-request increment | 500 in Prometheus/Grafana |

This is one controlled localhost workload, not maximum capacity or an SLA. HTTP
batch latency includes more than model computation. A cold start took 1,045.67 s,
with encoder acquisition/reconstruction dominating; warm tests do not resolve it.
The private source is
`reports/integration_validation/phase6e_full_20261116/final-validation.json`.

## Distribution monitoring

| Domain | Measure | Effect | Holm-adjusted p |
|---|---|---:|---:|
| Input data | Character length KS D | 0.03650 | 0.0015 |
| Input data | Word count KS D | 0.03725 | 0.0015 |
| Predictions | Cluster total variation | 0.04938 | 0.0015 |

The protocol used 1,999 resamples, seed 42 and alpha 0.05. Each effect was below
its provisional practical threshold of 0.1. Statistical significance does not
establish model degradation. Dependence between reviews and Monte Carlo resolution
limit interpretation. HTML and JSON evidence is retained locally under
`reports/drift_monitoring/phase6e-full-validation/`.

## Final geometric evaluation

| Measure | Phase 7A result |
|---|---:|
| Unique reviews evaluated | 2,261/2,261 |
| Exact cosine silhouette | 0.111541 |
| Mean assigned-centroid similarity | 0.738337 |
| Mean dispersion | 0.261663 |
| Mean assignment margin | 0.089303 |
| Cluster coverage | 10/10; sizes 102–342 |

Evaluation reused the frozen estimator and original training centroids. All
observations were retained. No development IDs overlap final IDs, but 127 final
reviews match training text and 76 match model-selection text after stripping.
These sets need not be disjoint. Complete content-level independence cannot be
claimed. The nominal “3k” filename contains 2,261 rows; coverage uses the actual
file, not the nominal name. Geometry measures separation, not semantic accuracy.
Private evidence remains in `reports/final_evaluation/final_20261116/`.

## Exploratory semantic annotation

The deterministic sample contains 100 distinct review identifiers, ten per cluster:
four high similarities, three near-median similarities and three remaining low
margins. Seed 42 resolves ties; within-cluster duplicate stripped texts are skipped.
All quotas were fulfilled; four duplicate candidates were bypassed.

One annotator assessed thematic coherence across **all 100 sampled reviews**.
The owner clarified that each nonempty consistency cell applies collectively to
the corresponding ten-review cluster sample. The ten cluster-level judgments are
five good, four excellent and one average, covering 100 reviews through grouped
assessment. They are not 100 independent per-review ratings. All 100 theme fields
are populated. Ambiguity and comment cells are empty; blank cells do not establish
an absence of ambiguity. The owner's preliminary summary was “good to very good.”

No individual themes or customer text are reproduced here. Targeted equal-quota
sampling, one annotator and missing semantic ground truth prevent unbiased global
accuracy estimates or inter-rater agreement claims.

## Consolidation checks

Phase 8: 167 offline tests passed, 13 opt-in tests skipped, 12 upstream deprecation
warnings; subsequently all 13 live integration tests passed. Lock, installed
dependencies, Compose, NGINX and Prometheus configuration checks passed. MLflow
access was verified without creating a run. DVC stage state and the older running
image limit fresh-checkout reproducibility claims; see the
[repository audit](repository_audit.md) and [command status](reproduction.md).
