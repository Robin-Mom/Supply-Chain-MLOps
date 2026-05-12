import pandas as pd
import spacy
import os
import logging
import time

# Configuration du logging (toujours vers logs/pipeline.log)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/pipeline.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

def run_spacy_pipeline():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_FILE = os.path.join(BASE_DIR, 'data', 'avis_cleaned.csv')
    OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'avis_bertopic.csv')

    if not os.path.exists(INPUT_FILE):
        log.error("Fichier source introuvable. Lancez d'abord preprocess.py")
        return

    log.info("Chargement du modèle SpaCy français...")
    try:
        nlp = spacy.load("fr_core_news_sm")
    except OSError:
        log.error("Modèle SpaCy introuvable. Lancez: python -m spacy download fr_core_news_sm")
        return

    log.info("Lecture des données...")
    df = pd.read_csv(INPUT_FILE, sep=';', low_memory=False)
    
    # Limitation pour test (optionnel) ou traitement complet
    textes = df['commentaire'].astype(str).tolist()

    log.info(f"Début du traitement SpaCy (Lemmatisation) sur {len(df)} lignes...")
    start_time = time.time()

    # Nettoyage optimisé
    clean_texts = []
    # n_process=-1 utilise tous les CPU disponibles
    docs = nlp.pipe(
        [t.lower() for t in textes], 
        disable=["ner", "parser"], 
        n_process=-1, 
        batch_size=500
    )

    for doc in docs:
        # On garde le lemme si ce n'est pas un stopword, ni une ponctuation, et longueur > 2
        tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and len(t.text) > 2]
        clean_texts.append(" ".join(tokens))

    df['avis_spacy'] = clean_texts
    
    # Nettoyage final : suppression des lignes vides après lemmatisation
    df = df[df['avis_spacy'].str.strip() != ""]

    duration = (time.time() - start_time) / 60
    log.info(f"Traitement terminé en {duration:.2f} minutes.")

    log.info(f"Sauvegarde vers {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False, sep=',', quoting=1)
    log.info("Fichier prêt pour BERTopic !")

if __name__ == "__main__":
    run_spacy_pipeline()
