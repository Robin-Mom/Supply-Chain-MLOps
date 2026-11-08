"""Evaluate geometry, c-TF-IDF keywords and clustering stability.

Default evaluation uses saved training embeddings and labels. --input evaluates
new reviews with a frozen KMeans or Spherical KMeans model. Stability explicitly refits models and
is restricted to the training data; it is opt-in because it can be expensive.
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

from train import METHODS, fit_clustering, load_reviews, write_json
from tracking import configure_tracking


def evaluate_clustering(X, labels, sample_size=5000, seed=42, metric="cosine"):
    """Score non-noise observations; report undefined scores as JSON null.

    sample_size=0 computes an exact silhouette. The default uses at most 5000
    assigned observations; all other statistics use every assigned observation.
    """
    X, labels = np.asarray(X), np.asarray(labels)
    if X.ndim != 2 or len(X) == 0 or labels.shape != (len(X),):
        raise ValueError("Expected a non-empty matrix and one label per row.")
    if not np.isfinite(X).all() or sample_size < 0:
        raise ValueError("Embeddings must be finite and sample_size nonnegative.")
    assigned = labels >= 0
    valid_X, valid_y = X[assigned], labels[assigned]
    unique, counts = np.unique(valid_y, return_counts=True)
    results = {"n_samples": len(X), "n_clusters": len(unique),
               "coverage": float(assigned.mean()), "noise_ratio": float(1-assigned.mean()),
               "largest_cluster_ratio": float(counts.max()/len(valid_y)) if len(counts) else None,
               "davies_bouldin": None, f"silhouette_{metric}": None,
               "silhouette_n_samples": 0}
    if not 2 <= len(unique) < len(valid_y):
        return results
    results["davies_bouldin"] = float(davies_bouldin_score(valid_X, valid_y))
    idx = np.arange(len(valid_y))
    if sample_size and len(idx) > sample_size:
        idx = np.random.default_rng(seed).choice(idx, sample_size, replace=False)
    results["silhouette_n_samples"] = len(idx)
    if 2 <= len(np.unique(valid_y[idx])) < len(idx):
        results[f"silhouette_{metric}"] = float(silhouette_score(valid_X[idx], valid_y[idx], metric=metric))
    return results


def name_clustering(texts, labels, top_n=10):
    """Extract French unigram/bigram keywords using the notebook's c-TF-IDF."""
    from bertopic.vectorizers import ClassTfidfTransformer
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import CountVectorizer
    from spacy.lang.fr.stop_words import STOP_WORDS
    columns = ["cluster", "rank", "term", "score_ctfidf"]
    df = pd.DataFrame({"text": texts, "cluster": labels})
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[(df.cluster >= 0) & df.text.ne("")].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=columns)
    stop_words = sorted({token for word in STOP_WORDS
                         for token in re.findall(r"(?u)\b\w\w+\b", word.lower())})
    vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(1, 2), min_df=1)
    try:
        X_docs = vectorizer.fit_transform(df.text)
    except ValueError as error:
        if "empty vocabulary" in str(error):
            return pd.DataFrame(columns=columns)
        raise
    cluster_labels, cluster_ids = np.unique(df.cluster, return_inverse=True)
    membership = csr_matrix((np.ones(len(df)), (cluster_ids, np.arange(len(df)))),
                            shape=(len(cluster_labels), len(df)))
    X_ctfidf = ClassTfidfTransformer().fit_transform(membership @ X_docs).tocsr()
    vocabulary = vectorizer.get_feature_names_out()
    rows = []
    for i, cluster_id in enumerate(cluster_labels):
        row = X_ctfidf.getrow(i)
        for rank, pos in enumerate(np.argsort(row.data)[::-1][:top_n], start=1):
            rows.append({"cluster": int(cluster_id), "rank": rank,
                         "term": vocabulary[row.indices[pos]], "score_ctfidf": float(row.data[pos])})
    return pd.DataFrame(rows, columns=columns)


def compare_partitions(a, b):
    """Report ARI with noise as one label, plus ARI excluding noise in either run."""
    common_assigned = (a >= 0) & (b >= 0)
    return {"ari": float(adjusted_rand_score(a, b)) if len(a) >= 2 else None,
            "ari_assigned_only": float(adjusted_rand_score(a[common_assigned], b[common_assigned]))
            if common_assigned.sum() >= 2 else None,
            "n_common": len(a), "n_common_assigned": int(common_assigned.sum())}


def evaluate_stability(X, method, config, mode="statistical", n_repeats=10,
                       sample_fraction=0.8, seed=42):
    """Compare partitions with fixed seeds, varying seeds or both resampling and seeds.

    statistical: independent subsets, identical seed.
    algorithmic: identical full data, varying seed, all pairs compared.
    global: independent subsets and varying seed.
    Noise (-1) is treated as a cluster in `ari`, matching the notebook.
    """
    if mode not in {"statistical", "algorithmic", "global"}:
        raise ValueError("Unknown stability mode.")
    if n_repeats < 2 or not 0 < sample_fraction <= 1:
        raise ValueError("Use at least two repeats and 0 < sample_fraction <= 1.")
    def run(data, run_seed):
        print(f"Stability: {method}, {mode}, shape={data.shape}, seed={run_seed}", flush=True)
        return fit_clustering(data, method, config, run_seed)[2]
    rows = []
    if mode == "algorithmic":
        partitions = [run(X, seed+i) for i in range(n_repeats)]
        for i, j in combinations(range(n_repeats), 2):
            rows.append({"repeat_a": i, "repeat_b": j, **compare_partitions(partitions[i], partitions[j])})
    else:
        size = int(len(X)*sample_fraction)
        if size < 2:
            raise ValueError("Subsets must contain at least two observations.")
        rng = np.random.default_rng(seed)
        for i in range(n_repeats):
            idx_a = np.sort(rng.choice(len(X), size, replace=False))
            idx_b = np.sort(rng.choice(len(X), size, replace=False))
            _, pos_a, pos_b = np.intersect1d(idx_a, idx_b, return_indices=True)
            seed_a = seed if mode == "statistical" else seed+2*i
            seed_b = seed if mode == "statistical" else seed+2*i+1
            a, b = run(X[idx_a], seed_a), run(X[idx_b], seed_b)
            rows.append({"repeat_a": i, "repeat_b": i, **compare_partitions(a[pos_a], b[pos_b])})
    return pd.DataFrame(rows).assign(method=method, mode=mode)


def main(argv=None):
    """Write metric JSON, keyword CSV and optional stability CSV/boxplots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, help="Optional combined JSON metrics for DVC.")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--input", type=Path, help="Optional held-out CSV evaluated with the saved models.")
    parser.add_argument("--text-column", default="commentaire")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--silhouette-sample-size", type=int, default=5000, help="0 means exact silhouette.")
    parser.add_argument("--skip-keywords", action="store_true")
    parser.add_argument("--stability", nargs="+", choices=("statistical", "algorithmic", "global"), default=[])
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--sample-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true", help="Append evaluation to recorded training runs.")
    args = parser.parse_args(argv)
    metadata = json.loads((args.artifact_dir / "metadata.json").read_text())
    methods = list(dict.fromkeys(args.methods or metadata["methods"]))
    if any(m not in METHODS for m in methods):
        parser.error(f"Supported methods are: {METHODS}.")
    if any(m not in metadata["methods"] for m in methods):
        parser.error("One or more requested methods were not trained.")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new directory.")
    if args.metrics_output and args.metrics_output.exists():
        parser.error("Metrics file already exists; choose a new path.")
    if args.input and args.stability:
        parser.error("Stability refitting is restricted to training data.")
    if args.silhouette_sample_size < 0 or args.batch_size < 1:
        parser.error("Invalid sample size or batch size.")
    if args.stability and (args.n_repeats < 2 or not 0 < args.sample_fraction <= 1):
        parser.error("Stability requires n-repeats >= 2 and 0 < sample-fraction <= 1.")
    runs = {}
    if args.log_mlflow:
        for method in methods:
            info_path = args.artifact_dir / method / "mlflow_run.json"
            if not info_path.exists():
                parser.error("No recorded MLflow run. Enable tracking during training first.")
            runs[method] = json.loads(info_path.read_text())
            try:
                configure_tracking(runs[method]["tracking_uri"])
            except ValueError as error:
                parser.error(str(error))
    if args.input:
        from predict import predict_reviews, load_bundle, predict_embeddings
        _, texts = load_reviews(args.input, args.text_column)
        _, X = predict_reviews(args.artifact_dir, methods[0], texts, args.device, args.batch_size)
    else:
        X = np.load(args.artifact_dir / "embeddings.npy", allow_pickle=False)
        texts = pd.read_csv(args.artifact_dir / "texts.csv")["text"].to_numpy()
    args.output_dir.mkdir(parents=True)
    combined_metrics = {}
    for method in methods:
        directory = args.output_dir / method
        directory.mkdir()
        source = args.artifact_dir / method
        labels = predict_embeddings(load_bundle(args.artifact_dir, method), X) if args.input else np.load(source / "labels.npy", allow_pickle=False)
        metrics = evaluate_clustering(X, labels, args.silhouette_sample_size, args.seed)
        combined_metrics[method] = metrics
        write_json(directory / "metrics.json", metrics)
        write_json(directory / "evaluation_config.json", {"dataset": "held_out" if args.input else "training",
                   "input": str(args.input) if args.input else None, "seed": args.seed,
                   "silhouette_sample_size": args.silhouette_sample_size, "n_repeats": args.n_repeats,
                   "sample_fraction": args.sample_fraction, "stability": args.stability})
        pd.DataFrame({"text": texts, "cluster_id": labels}).to_csv(directory / "assignments.csv", index=False)
        if not args.skip_keywords:
            name_clustering(texts, labels).to_csv(directory / "keywords.csv", index=False)
        for mode in dict.fromkeys(args.stability):
            start = perf_counter()
            results = evaluate_stability(X, method, metadata["config"], mode, args.n_repeats, args.sample_fraction, args.seed)
            results.to_csv(directory / f"stability_{mode}.csv", index=False)
            write_json(directory / f"stability_{mode}_timing.json", {"seconds": perf_counter()-start})
            if args.plot:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                valid = results["ari"].dropna()
                if not valid.empty:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.boxplot(valid.to_numpy())
                    ax.set_title(f"{method}: {mode} stability")
                    ax.set_ylabel("Adjusted Rand Index (including noise)")
                    fig.tight_layout()
                    fig.savefig(directory / f"stability_{mode}.png", dpi=150)
                    plt.close(fig)
        if args.log_mlflow:
            import mlflow
            info = runs[method]
            configure_tracking(info["tracking_uri"])
            prefix = "held_out" if args.input else "training"
            with mlflow.start_run(run_id=info["run_id"]):
                mlflow.log_metrics({f"{prefix}_{k}": v for k, v in metrics.items() if v is not None})
                mlflow.log_artifacts(str(directory), artifact_path=f"evaluation/{prefix}/{args.output_dir.name}")
        print(f"{method}: {json.dumps(metrics)}", flush=True)
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.metrics_output, combined_metrics)


if __name__ == "__main__":
    main()
