#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging as log
import os
import joblib
import dagshub
import mlflow

from sentence_transformers import SentenceTransformer
from mlflow.tracking import MlflowClient


def load_production_model(model_name):

    client = MlflowClient()

    versions = client.get_latest_versions(
        model_name,
        stages=["Production"]
    )

    if len(versions) == 0:
        raise ValueError(
            f"No Production model found for {model_name}"
        )

    model_uri = versions[0].source

    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri
    )

    return local_path


def predict(text, model_name):

    dagshub.init(
        repo_owner='schmilblick-ai',
        repo_name='Supply-Chain-MLOps',
        mlflow=True
    )

    model_path = load_production_model(model_name)

    with open(
        os.path.join(model_path, "config.json"),
        "r"
    ) as f:
        config = json.load(f)

    st_model_name = config["sentence_transformer"]

    kmeans = joblib.load(
        os.path.join(model_path, "kmeans.pkl")
    )

    meta_labels = joblib.load(
        os.path.join(model_path, "meta_labels.pkl")
    )

    embedding_model = SentenceTransformer(
        st_model_name, device='cpu'
    )

    embedding = embedding_model.encode([text])

    meta_topic = kmeans.predict(embedding)[0]

    if meta_topic in meta_labels:
        label = " | ".join(meta_labels[meta_topic])
    else:
        label = "unknown"

    return {
        "text": text,
        "meta_topic": int(meta_topic),
        "meta_label": label
    }


def _cli():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--text",
        required=True
    )

    parser.add_argument(
        "-mn",
        "--model_name",
        default="trustpilot_bertopic_v2"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true"
    )

    args = parser.parse_args()

    if args.verbose:
        log.basicConfig(level=log.INFO)

    return vars(args)


if __name__ == "__main__":

    args = _cli()

    result = predict(
        args["text"],
        args["model_name"]
    )

    print(result)

