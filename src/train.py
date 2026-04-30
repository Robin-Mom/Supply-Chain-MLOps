#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train.py:
Train the model, generate 10 final clusters, export data_clusterized.csv, model, metrics and logs.

AUTHOR
Robin Mom

VERSION
1.0

DATE
29/04/2026
"""
### Imports

import argparse
import logging as log # module standard pour la gestion des messages de diagnostic.
import pandas as pd
import numpy as np
import os

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from spacy.lang.fr.stop_words import STOP_WORDS
import joblib

### FUNCTIONS




### MAIN

def main(verbose, filepath, sep, colname, sentenceTransformer, n_clusters, model_output, data_output, log_output):

	# -------------------------
	# 1. Données
	# -------------------------
	log.info(f"MINIMAL LOGGING: Loading processed data: {filepath}")
	logdir = os.path.dirname(log_output)
	os.makedirs(logdir, exist_ok=True)
	with open(log_output, "w", encoding="utf-8") as f:
		f.write(f"MINIMAL LOGGING: Loading processed data: {filepath}\n")
	df_processed = pd.read_csv(filepath, sep=sep)
	documents = df_processed[colname].tolist()
	log.info(f"MINIMAL LOGGING: Data loaded")
	with open(log_output, "a", encoding="utf-8") as f:
		f.write(f"MINIMAL LOGGING: Data loaded\n")
	# -------------------------
	# 2. Embeddings
	# -------------------------
	log.info(f"MINIMAL LOGGING: Starting embeddings with {sentenceTransformer}")
	with open(log_output, "a", encoding="utf-8") as f:
		f.write(f"MINIMAL LOGGING: Starting embeddings with {sentenceTransformer}\n")
	embedding_model = SentenceTransformer(sentenceTransformer, device="cuda")
	embeddings = embedding_model.encode(documents, batch_size=64, show_progress_bar=True)
	log.info("MINIMAL LOGGING: embeddings complete")
	with open(log_output, "a", encoding="utf-8") as f:
		f.write("MINIMAL LOGGING: embeddings complete\n")
	# -------------------------
	# 3. BERTopic (clustering initial)
	# -------------------------
	log.info("MINIMAL LOGGING: Starting BERTopic clustering pipeline (embeddings + UMAP + HDBSCAN)")
	with open(log_output, "a", encoding="utf-8") as f:
		f.write("MINIMAL LOGGING: Starting BERTopic clustering pipeline (embeddings + UMAP + HDBSCAN)\n")
	topic_model = BERTopic(verbose=False)
	topics, probs = topic_model.fit_transform(documents, embeddings)
	log.info("MINIMAL LOGGING: Clustering complete")
	with open(log_output, "a", encoding="utf-8") as f:
		f.write("MINIMAL LOGGING: Clustering complete\n")
	# -------------------------
	# 4. Calcul des centroïdes par topic
	# -------------------------
	log.info("MINIMAL LOGGING: computing centroids from initial embeddings for Meta-clustering")
	with open(log_output, "a", encoding="utf-8") as f:
		f.write("MINIMAL LOGGING: computing centroids from initial embeddings for Meta-clustering\n")
	df = pd.DataFrame({
		"doc": documents,
		"topic": topics
	})

	df["embedding"] = list(embeddings)

	centroids = (
		df[df.topic != -1]  # enlever outliers
		.groupby("topic")["embedding"]
		.apply(lambda x: np.mean(np.vstack(x), axis=0))
	)

	centroid_matrix = np.vstack(centroids.values)
	log.info("MINIMAL LOGGING: centroids computed")
	with open(log_output, "a", encoding="utf-8") as f:
		f.write("MINIMAL LOGGING: centroids computed\n")
	# -------------------------
	# 5. Meta-clustering (KMeans → 10 clusters)
	# -------------------------
	log.info(f"MINIMAL LOGGING: starting KMeans Meta-clustering with n_clusters = {n_clusters}")
	with open(log_output, "a", encoding="utf-8") as f:
		f.write(f"MINIMAL LOGGING: starting KMeans Meta-clustering with n_clusters = {n_clusters}\n")
	kmeans = KMeans(n_clusters=n_clusters, random_state=42)
	meta_topics = kmeans.fit_predict(centroid_matrix)
	log.info("MINIMAL LOGGING: Meta-clustering complete")
	with open(log_output, "a", encoding="utf-8") as f:
		f.write("MINIMAL LOGGING: Meta-clustering complete\n")
	# mapping topic initial → meta-topic
	topic_to_meta = dict(zip(centroids.index, meta_topics))

	# -------------------------
	# 6. Attribution des meta-topics aux documents
	# -------------------------
	df["meta_topic"] = df["topic"].map(topic_to_meta)

	# outliers (-1) → optionnel
	df["meta_topic"] = df["meta_topic"].fillna(-1)
	
	log.info("MINIMAL LOGGING: Starting c-TF-IDF labeling for meta-clusters")
	with open(log_output, "a", encoding="utf-8") as f:
		f.write("MINIMAL LOGGING: Starting c-TF-IDF labeling for meta-clusters\n")

	# -------------------------
	# 7. Attribution des noms de clusters
	# -------------------------
	### functions
	def extract_top_words_per_cluster(tfidf_matrix, feature_names, top_n=5):
		labels = {}
		for i in range(tfidf_matrix.shape[0]):
			row = tfidf_matrix[i].toarray().flatten()
			top_indices = row.argsort()[-top_n:][::-1]
			labels[i] = [feature_names[j] for j in top_indices]
		return labels

	def format_label(meta_topic):
		if meta_topic == -1 or meta_topic not in meta_labels:
			return "outlier"
		return " | ".join(meta_labels[meta_topic])
	###
	
	# 7.1. Construction des "documents" par meta-cluster
	meta_docs = (
		df[df.meta_topic != -1]
		.groupby("meta_topic")["doc"]
		.apply(lambda docs: " ".join(docs))
	)

	# Sécurité : vérifier qu'on a des clusters
	if len(meta_docs) == 0:
		log.warning("No meta-clusters found for labeling")
		meta_labels = {}
	else:
		# 7.2. Vectorizer adapté au français
		french_stopwords = list(STOP_WORDS)
		vectorizer = CountVectorizer(
			stop_words=french_stopwords,
			ngram_range=(1, 2),
			min_df=5
		)

		X = vectorizer.fit_transform(meta_docs)

		# 7.3. c-TF-IDF (implémentation officielle BERTopic)
		ctfidf_model = ClassTfidfTransformer()
		c_tf_idf = ctfidf_model.fit_transform(X)

		feature_names = vectorizer.get_feature_names_out()

		# 7.4. Extraction des top mots par meta-cluster
		meta_labels = extract_top_words_per_cluster(
			c_tf_idf, feature_names, top_n=5
		)

	# 7.5. Mapping meta_topic -> label texte
	
	df["meta_label"] = df["meta_topic"].apply(format_label)

	log.info("MINIMAL LOGGING: Meta-cluster labeling complete")
	with open(log_output, "a", encoding="utf-8") as f:
		f.write("MINIMAL LOGGING: Meta-cluster labeling complete\n")
	# -------------------------
	# 8. Résultat final
	# -------------------------
	log.info(f"MINIMAL LOGGING: label and size of Meta-clusters:\n{df['meta_label'].value_counts()}")
	with open(log_output, "a", encoding="utf-8") as f:
		f.write(f"MINIMAL LOGGING: label and size of Meta-clusters:\n{df['meta_label'].value_counts()}\n")
	# -------------------------
	# 9. Sauvegarde du modèle et de données labellisées
	# -------------------------
	log.info("Saving outputs")
	with open(log_output, "a", encoding="utf-8") as f:
		f.write("Saving outputs\n")
	datadir = os.path.dirname(data_output)
	os.makedirs(datadir, exist_ok=True)
	modeldir = os.path.dirname(model_output)
	os.makedirs(modeldir, exist_ok=True)
	df.to_csv(data_output)
	topic_model.save(model_output)
	joblib.dump(kmeans, model_output+"_kmeans.pkl")
	joblib.dump(topic_to_meta, model_output+"_topic_to_meta.pkl")
	joblib.dump(embeddings, model_output+"_embeddings_"+sentenceTransformer+".pkl")
	
def _cli():
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            argument_default=argparse.SUPPRESS)
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Boolean: activate verbose mode. Default is no verbose.")
    parser.add_argument('-f', '--filepath', default="data/processed.csv", type=str, help="path to your input csv file of processed dataset")
    parser.add_argument('-s', '--sep', default=',', type=str, help="separator to parse input csv")
    parser.add_argument('-cn', '--colname', default="commentaire", type=str, help="Column name in processed data dataframe to use for embeddings")
    parser.add_argument('-st', '--sentenceTransformer', default="paraphrase-multilingual-mpnet-base-v2", type=str, help="Name of sentenceTransformer model to use in BERTopic")
    parser.add_argument('-nc', '--n_clusters', default=10, type=int, help="Number of final clusters")
    parser.add_argument('-mo', '--model_output', default="models/BERTopic", type=str, help="Name of output file for model.")
    parser.add_argument('-do', '--data_output', default="data/clusterized.csv", type=str, help="Name of output file for clusterized data.")
    parser.add_argument('-lo', '--log_output', default="models/minimal.log", type=str, help="Name of output file for minimal logs.")
        
    args = parser.parse_args()
    
    if args.verbose == True :
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
        log.info("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    return vars(args)

if __name__ == '__main__':
    main(**_cli())












