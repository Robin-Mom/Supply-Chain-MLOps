#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging as log
import joblib
from sentence_transformers import SentenceTransformer

def predict(text, model_path, st_model_name):
    # -------------------------
    # 1. Load modèles
    # -------------------------
    kmeans = joblib.load(model_path + "_kmeans.pkl")
    meta_labels = joblib.load(model_path + "_meta_labels.pkl")
    log.info(f"loaded kmeans and meta_labels from {model_path}")
    embedding_model = SentenceTransformer(st_model_name)
    log.info(f"loaded SentenceTransformer model: {st_model_name} for embeddings computation.")
    # -------------------------
    # 2. Embedding
    # -------------------------
    embedding = embedding_model.encode([text])

    # -------------------------
    # 3. Meta-clustering direct
    # -------------------------
    meta_topic = kmeans.predict(embedding)[0]

    # -------------------------
    # 4. Label
    # -------------------------
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
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            argument_default=argparse.SUPPRESS)
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Boolean: activate verbose mode. Default is no verbose.")
    parser.add_argument("-t", "--text", type=str, required=True, help="input text for wich you want to assign cluster.")
    parser.add_argument("-m", "--model", type=str, default="models/BERTopic", help="path to model you want to use for predict")
    parser.add_argument("--st", type=str, default="paraphrase-multilingual-mpnet-base-v2", help="name of sentence Transformer used to generate initial embeddings. Beware of using the sampe as for train.py")

    args = parser.parse_args()
    
    if args.verbose == True :
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
        log.info("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    return vars(args)


if __name__ == "__main__":
    args = _cli()
    result = predict(args["text"], args["model"], args["st"])
    print(result)
