import pandas as pd
import os
import logging

# 1. Création du dossier logs s'il n'existe pas
os.makedirs("logs", exist_ok=True)

# 2. Configuration du logging (Correction de la syntaxe ici)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"), # Sauvegarde dans ce fichier
        logging.StreamHandler()                  # Affiche aussi dans le terminal
    ]
)
# La variable log doit être définie APRES la config et DEHORS
log = logging.getLogger(__name__)

def clean_data(input_path, output_path):
    """
    Fonction de nettoyage des données (Preprocessing)
    """
    if not os.path.exists(input_path):
        log.error(f"Fichier introuvable : {input_path}")
        return

    log.info(f"Chargement des données depuis {input_path}...")
    # On utilise low_memory=False pour éviter les warnings sur les gros fichiers
    df = pd.read_csv(input_path, sep=';', low_memory=False)

    # 1. Gestion des doublons
    initial_count = len(df)
    df = df.drop_duplicates(keep='first')
    removed_doubles = initial_count - len(df)
    log.info(f"{removed_doubles} doublons supprimés.")

    # 2. Gestion des valeurs manquantes
    if 'commentaire' in df.columns and 'titre' in df.columns:
        missing_comments = df['commentaire'].isna().sum()
        df['commentaire'] = df['commentaire'].fillna(df['titre'])
        log.info(f"{missing_comments} commentaires vides remplacés par leur titre.")
    
    # Suppression des lignes où le commentaire est toujours vide après remplissage
    df = df.dropna(subset=['commentaire'])
    
    # 3. Sauvegarde des données propres
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, sep=';')
    log.info(f"Données nettoyées sauvegardées dans {output_path} (Taille finale : {len(df)} lignes).")

if __name__ == "__main__":
    # Définition des chemins
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_FILE = os.path.join(BASE_DIR, 'data', 'avis_40k.csv')
    OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'avis_cleaned.csv')

    clean_data(INPUT_FILE, OUTPUT_FILE)