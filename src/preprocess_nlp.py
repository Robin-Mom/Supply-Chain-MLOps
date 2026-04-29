import pandas as pd
import os
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configuration du logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/pipeline.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# Téléchargement des ressources NLTK
log.info("Téléchargement des ressources NLTK...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

def clean_text(text):
    """Nettoyage complet NLP pour le français."""
    if pd.isna(text):
        return ""
    
    # 1. Minuscules et suppression caractères spéciaux/nombres
    text = str(text).lower()
    text = re.sub(r'[^a-zàâçéèêëîïôûù\s]', '', text)
    
    # 2. Tokenization
    tokens = word_tokenize(text, language='french')
    
    # 3. Stopwords
    stop_words = set(stopwords.words('french'))
    stop_words.update(['avis', 'client', 'commande', 'site', 'plus', 'oscaro', 'pieces', 'piece']) 
    
    # Filtrage (mots vides et mots trop courts)
    cleaned_tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    return " ".join(cleaned_tokens)

def run_nlp_pipeline():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_FILE = os.path.join(BASE_DIR, 'data', 'avis_cleaned.csv')
    OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'avis_nlp.csv')

    if not os.path.exists(INPUT_FILE):
        log.error("Fichier source introuvable. Lancez d'abord preprocess.py")
        return

    log.info("Lecture des données nettoyées...")
    df = pd.read_csv(INPUT_FILE, sep=';', low_memory=False)

    log.info(f"Début du traitement NLP sur {len(df)} lignes (cela peut prendre un moment)...")
    df['commentaire_nltk'] = df['commentaire'].apply(clean_text)

    # On supprime les lignes qui seraient devenues vides après nettoyage NLTK
    df = df[df['commentaire_nltk'] != ""]

    log.info(f"Sauvegarde du dataset NLP : {len(df)} lignes restantes.")
    df.to_csv(OUTPUT_FILE, index=False, sep=';')
    log.info(f"Fichier sauvegardé : {OUTPUT_FILE}")

if __name__ == "__main__":
    run_nlp_pipeline()
