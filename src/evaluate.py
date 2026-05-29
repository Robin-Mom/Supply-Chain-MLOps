#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging as log
import json
import os
import dagshub
import mlflow
import numpy as np
import pandas as pd

from mlflow.tracking import MlflowClient
from sklearn.metrics import silhouette_score
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

def push_metrics(score):
    """Envoie le score de silhouette au PushGateway Prometheus"""
    try:
        registry = CollectorRegistry()
        g = Gauge("model_silhouette_score", "Silhouette score du modèle", registry=registry)
        g.set(float(score))
        push_to_gateway("pushgateway:9091", job="evaluate_pipeline", registry=registry)
        log.info("Metrics pushed to Prometheus successfully")
    except Exception as e:
        log.warning(f"Could not push metrics to Prometheus: {e}")

def main(args):
    # Initialisation
    dagshub.init(repo_owner='schmilblick-ai', repo_name='Supply-Chain-MLOps', mlflow=True)
    
    with open(args['run_id_file'], "r") as f:
        run_id = f.read().strip()

    df = pd.read_csv(args['filepath'], sep=args['sep'])
    mask = df[args['col_topics']] != -1
    
    X = np.vstack(df.loc[mask, args['col_embs']].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ")).values)
    labels = df.loc[mask, args['col_topics']].values

    if len(set(labels)) < 2:
        raise ValueError("Need at least 2 clusters to compute silhouette score")

    score = silhouette_score(X, labels, metric="cosine")
    
    # Sauvegarde locale métrique
    os.makedirs(os.path.dirname(args['metrics_output']), exist_ok=True)
    with open(args['metrics_output'], "w") as f:
        json.dump({"silhouette_score": float(score)}, f, indent=4)

    # MLflow Tracking
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("silhouette_score", float(score))

    # Promotion Logique
    client = MlflowClient()
    runs = mlflow.search_runs(experiment_names=[args['experiment_name']])
    best_score = runs["metrics.silhouette_score"].max() if "metrics.silhouette_score" in runs.columns else -1

    if score >= best_score:
        artifact_uri = mlflow.get_artifact_uri(args['artifact_path'])
        
        try:
            client.create_registered_model(args['model_name'])
        except:
            pass
        
        # Création et Promotion
        result = client.create_model_version(name=args['model_name'], source=artifact_uri, run_id=run_id)
        client.transition_model_version_stage(name=args['model_name'], version=result.version, stage="Production")
        
        # Archive les anciennes versions Production
        versions = client.search_model_versions(f"name='{args['model_name']}'")
        for v in versions:
            if v.stage == "Production" and v.version != result.version:
                client.transition_model_version_stage(name=args['model_name'], version=v.version, stage="Archived")

        log.info(f"New best model registered: {score}")
        
        # Envoi métrique vers Prometheus
        push_metrics(score)
    else:
        log.info(f"Model not promoted. Score={score}, Best={best_score}")

def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", default="models/artifacts/clusterized.csv")
    parser.add_argument("-ct", "--col_topics", default="meta_topic")
    parser.add_argument("-ce", "--col_embs", default="embedding")
    parser.add_argument("-mo", "--metrics_output", default="metrics/silhouette.json")
    parser.add_argument("-ri", "--run_id_file", default="models/last_run_id.txt")
    parser.add_argument("-en", "--experiment_name", default="Bertopic_Trustpilot_v4")
    parser.add_argument("-mn", "--model_name", default="trustpilot_bertopic_v4")
    parser.add_argument("-ap", "--artifact_path", default="model")
    # Ajout de l'argument sep manquant
    parser.add_argument("-sep", "--sep", default=",", help="Separator for CSV") 
    return vars(parser.parse_args())

if __name__ == "__main__":
    main(_cli())