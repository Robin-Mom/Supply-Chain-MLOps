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


def main(
    verbose,
    filepath,
    sep,
    col_topics,
    col_embs,
    metrics_output,
    log_output,
    run_id_file,
    experiment_name,
    model_name,
    artifact_path
):
# initialisation repository dagshub
    dagshub.init(
        repo_owner='schmilblick-ai',
        repo_name='Supply-Chain-MLOps',
        mlflow=True
    )
# récupération du dernier run_id
    with open(run_id_file, "r") as f:
        run_id = f.read().strip()
# chargement des dernières données clusterizées
    df = pd.read_csv(filepath, sep=sep)
# exclusion des outliers
    mask = df[col_topics] != -1
# récupération des embeddings (sans les outliers)
    X = np.vstack(
        df.loc[mask, col_embs]
        .apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        .values
    )
# récupération des labels (sans les outliers)
    labels = df.loc[mask, col_topics].values
# filtre de sécurité: il faut au moins 2 clusters pour pouvoir calculer un silhouette score
    if len(set(labels)) < 2:
        raise ValueError("Need at least 2 clusters")
# calcul du silhouette score
    score = silhouette_score(
        X,
        labels,
        metric="cosine"
    )
# sécurité: si le répertoire d'output pour la métrique (silhouette score) n'éxiste pas encore, il est créé
    os.makedirs(
        os.path.dirname(metrics_output),
        exist_ok=True
    )
# écriture du silhouette score
    with open(metrics_output, "w") as f:
        json.dump(
            {"silhouette_score": float(score)},
            f,
            indent=4
        )
# mlflow tracking
    with mlflow.start_run(run_id=run_id): # le run_id généré par la dernière éxécution de train.py est utilisé pour traquer la métrique

        mlflow.log_metric(
            "silhouette_score",
            float(score)
        )
# utilisation de mlflowclient pour enregistrer le modèle
        client = MlflowClient()
# on load tous les runs correspondant à l'expérience en cours
        runs = mlflow.search_runs(
            experiment_names=[experiment_name]
        )
# récupération et mise à jour du meilleur score
        if "metrics.silhouette_score" in runs.columns:
            best_score = runs["metrics.silhouette_score"].max()
        else:
            best_score = -1

        if score >= best_score:
# Si le core du modèle courant est supèrieur au meilleur score enregistré alors le modèle courant est enregistré et tagué pour la mise en production
            artifact_uri = mlflow.get_artifact_uri(artifact_path)
            
            client = MlflowClient()
            
            # Création du registre si inexistant 
            try: 
                client.create_registered_model(model_name) 
            except Exception: 
                pass
            
            # Création version
            result = client.create_model_version( name=model_name, source=artifact_uri, run_id=run_id )
            
            # Promotion Production
            client.transition_model_version_stage( name=model_name, version=result.version, stage="Production" )

            log.info(
                f"New best model registered: {score}"
            )
# si non, le modèle n'est pas enregistré
        else:

            log.info(
                f"Model not promoted. Score={score}, Best={best_score}"
            )


def _cli():

    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbose", action="store_true")

    parser.add_argument(
        "-f",
        "--filepath",
        default="models/artifacts/clusterized.csv"
    )

    parser.add_argument(
        "-s",
        "--sep",
        default=","
    )

    parser.add_argument(
        "-ct",
        "--col_topics",
        default="meta_topic"
    )

    parser.add_argument(
        "-ce",
        "--col_embs",
        default="embedding"
    )

    parser.add_argument(
        "-mo",
        "--metrics_output",
        default="metrics/silhouette.json"
    )

    parser.add_argument(
        "-lo",
        "--log_output",
        default="logs/evaluate.log"
    )

    parser.add_argument(
        "-ri",
        "--run_id_file",
        default="models/last_run_id.txt"
    )

    parser.add_argument(
        "-en",
        "--experiment_name",
        default="Bertopic_Trustpilot"
    )

    parser.add_argument(
        "-mn",
        "--model_name",
        default="trustpilot_bertopic"
    )

    parser.add_argument(
        "-ap",
        "--artifact_path",
        default="model"
    )

    args = parser.parse_args()

    if args.verbose:
        log.basicConfig(level=log.INFO)

    return vars(args)


if __name__ == "__main__":
    main(**_cli())

