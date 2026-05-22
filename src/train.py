#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging as log
import os
import json
import joblib
import dagshub
import mlflow
import numpy as np
import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from spacy.lang.fr.stop_words import STOP_WORDS


def extract_top_words_per_cluster(tfidf_matrix, feature_names, top_n=5):
    labels = {}

    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[-top_n:][::-1]
        labels[i] = [feature_names[j] for j in top_indices]

    return labels


def main(
    verbose,
    exp_name,
    run_name,
    artifact_path,
    filepath,
    sep,
    colname,
    sentenceTransformer,
    n_clusters,
    output_dir,
    log_output,
    gpu_accel
):

    dagshub.init(
        repo_owner='schmilblick-ai',
        repo_name='Supply-Chain-MLOps',
        mlflow=True
    )

    mlflow.set_experiment(exp_name)

    os.makedirs(output_dir, exist_ok=True)

    logdir = os.path.dirname(log_output)
    os.makedirs(logdir, exist_ok=True)

    def write_log(message):
        log.info(message)
        with open(log_output, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    write_log("Loading processed dataset")

    df_processed = pd.read_csv(filepath, sep=sep)

    documents = df_processed[colname].astype(str).tolist()

    device = "cuda" if gpu_accel else "cpu"

    write_log(f"Loading SentenceTransformer: {sentenceTransformer}")

    embedding_model = SentenceTransformer(
        sentenceTransformer,
        device=device
    )

    write_log("Computing embeddings")

    embeddings = embedding_model.encode(
        documents,
        batch_size=64,
        show_progress_bar=True
    )

    write_log("Training BERTopic")

    topic_model = BERTopic(verbose=False)

    topics, probs = topic_model.fit_transform(
        documents,
        embeddings
    )

    write_log("Computing centroids")

    df = pd.DataFrame({
        "doc": documents,
        "topic": topics
    })

    df["embedding"] = list(embeddings)

    centroids = (
        df[df.topic != -1]
        .groupby("topic")["embedding"]
        .apply(lambda x: np.mean(np.vstack(x), axis=0))
    )

    centroid_matrix = np.vstack(centroids.values)

    write_log("Training KMeans meta-clustering")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto"
    )

    meta_topics = kmeans.fit_predict(centroid_matrix)

    topic_to_meta = dict(zip(centroids.index, meta_topics))

    df["meta_topic"] = df["topic"].map(topic_to_meta)
    df["meta_topic"] = df["meta_topic"].fillna(-1)

    write_log("Generating c-TF-IDF labels")

    meta_docs = (
        df[df.meta_topic != -1]
        .groupby("meta_topic")["doc"]
        .apply(lambda docs: " ".join(docs))
    )

    if len(meta_docs) == 0:
        meta_labels = {}

    else:

        vectorizer = CountVectorizer(
            stop_words=list(STOP_WORDS),
            ngram_range=(1, 2),
            min_df=5
        )

        X = vectorizer.fit_transform(meta_docs)

        ctfidf_model = ClassTfidfTransformer()

        c_tf_idf = ctfidf_model.fit_transform(X)

        feature_names = vectorizer.get_feature_names_out()

        meta_labels = extract_top_words_per_cluster(
            c_tf_idf,
            feature_names,
            top_n=5
        )

    def format_label(meta_topic):

        if meta_topic == -1:
            return "outlier"

        if meta_topic not in meta_labels:
            return "unknown"

        return " | ".join(meta_labels[meta_topic])

    df["meta_label"] = df["meta_topic"].apply(format_label)

    write_log("Saving artifacts")

    artifacts_dir = os.path.join(output_dir, "artifacts")

    os.makedirs(artifacts_dir, exist_ok=True)

    df.to_csv(
        os.path.join(artifacts_dir, "clusterized.csv"),
        index=False
    )

    np.save(
        os.path.join(artifacts_dir, "embeddings.npy"),
        embeddings
    )

    topic_model.save(
        os.path.join(artifacts_dir, "bertopic_model"),
        serialization="safetensors"
    )

    joblib.dump(
        kmeans,
        os.path.join(artifacts_dir, "kmeans.pkl")
    )

    joblib.dump(
        meta_labels,
        os.path.join(artifacts_dir, "meta_labels.pkl")
    )

    joblib.dump(
        topic_to_meta,
        os.path.join(artifacts_dir, "topic_to_meta.pkl")
    )

    config = {
        "sentence_transformer": sentenceTransformer,
        "n_clusters": n_clusters
    }

    with open(
        os.path.join(artifacts_dir, "config.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=4)

    with mlflow.start_run(run_name=run_name) as run:

        run_id = run.info.run_id

        with open(
            os.path.join(output_dir, "last_run_id.txt"),
            "w",
            encoding="utf-8"
        ) as f:
            f.write(run_id)

        mlflow.log_params(config)

        mlflow.log_metric(
            "n_documents",
            len(df)
        )

        mlflow.log_metric(
            "n_topics",
            len(set(topics)) - (1 if -1 in topics else 0)
        )

        mlflow.log_artifacts(
            artifacts_dir,
            artifact_path=artifact_path
        )

    write_log("Training complete")


def _cli():

    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbose", action="store_true")

    parser.add_argument(
        "-en",
        "--exp_name",
        default="Bertopic_Trustpilot"
    )

    parser.add_argument(
        "-rn",
        "--run_name",
        default="first_run"
    )

    parser.add_argument(
        "-ap",
        "--artifact_path",
        default="model"
    )

    parser.add_argument(
        "-f",
        "--filepath",
        default="data/avis_bertopic.csv"
    )

    parser.add_argument(
        "-s",
        "--sep",
        default=","
    )

    parser.add_argument(
        "-cn",
        "--colname",
        default="commentaire"
    )

    parser.add_argument(
        "-st",
        "--sentenceTransformer",
        default="paraphrase-multilingual-MiniLM-L12-v2"
    )

    parser.add_argument(
        "-nc",
        "--n_clusters",
        default=10,
        type=int
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default="models"
    )

    parser.add_argument(
        "-lo",
        "--log_output",
        default="logs/train.log"
    )

    parser.add_argument(
        "-gpu",
        "--gpu_accel",
        action="store_true"
    )

    args = parser.parse_args()

    if args.verbose:
        log.basicConfig(level=log.INFO)

    return vars(args)


if __name__ == "__main__":
    main(**_cli())

