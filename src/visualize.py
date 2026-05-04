import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os
import logging

# Configuration du logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

def generate_wordcloud(data, note_max, source_type):
    """
    Génère et sauvegarde le nuage de mots.
    """
    # 1. Filtrage par note
    subset = data[data['note'] <= note_max]

    if subset.empty:
        log.warning(f"Aucun avis trouvé pour une note <= {note_max}.")
        return

    # 2. Choix de la colonne de texte selon la source
    # Si c'est du NLP, on utilise 'commentaire_nltk', sinon 'commentaire'
    colonne_texte = 'commentaire_nltk' if source_type == 'nlp' else 'commentaire'
    
    # Préparation du texte (on ignore les valeurs vides)
    text = " ".join(str(review) for review in subset[colonne_texte] if pd.notna(review) and review != "")

    # 3. Configuration visuelle
    couleur = "Reds" if note_max <= 2 else "viridis"
    suffixe = "Brut" if source_type == "brut" else "NLP"
    titre = f"Nuage de mots ({suffixe}) - Notes <= {note_max}"

    # 4. Stopwords (on les garde au cas où pour les données brutes)
    mots_inutiles = set(STOPWORDS)
    mots_inutiles.update(["le", "la", "les", "de", "des", "un", "une", "du", "pour", "dans", 
                          "en", "ce", "ces", "est", "a", "au", "aux", "sur", "plus", "très",
                          "oscaro", "commande", "pièces", "pièce", "avis", "produit"])

    # 5. Création du WordCloud
    log.info(f"Génération du WordCloud {suffixe} pour notes <= {note_max}...")
    wc = WordCloud(
        width=1200, height=600, 
        background_color='white',
        stopwords=None if source_type == 'nlp' else mots_inutiles, # NLTK a déjà filtré
        max_words=100,
        colormap=couleur
    ).generate(text)

    # 6. Sauvegarde
    OUTPUT_DIR = "reports/figures"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, f"wordcloud_{source_type}_note_{note_max}.png")
    
    plt.figure(figsize=(15, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(titre, fontsize=20)
    plt.axis("off")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Succès ! Image sauvegardée : {filename}")

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    while True:
        print("\n--- Menu WordCloud Interactif ---")
        print("1. Traiter les données Nettoyées (avis_cleaned.csv)")
        print("2. Traiter les données NLP avec NLTK (avis_nlp.csv)")
        print("q. Quitter")
        
        choix_source = input("Choisissez la source de données : ").lower()
        
        if choix_source == 'q':
            break
        
        if choix_source == '1':
            path = os.path.join(BASE_DIR, 'data', 'avis_cleaned.csv')
            source_type = 'brut'
        elif choix_source == '2':
            path = os.path.join(BASE_DIR, 'data', 'avis_nlp.csv')
            source_type = 'nlp'
        else:
            print("⚠️ Choix invalide.")
            continue

        if not os.path.exists(path):
            log.error(f"Fichier introuvable : {path}. Lancez les scripts de preprocessing d'abord.")
            continue

        # Demande de la note
        try:
            note_input = input("Entrez la note maximale (1 à 5) : ")
            note_max = int(note_input)
            if not (1 <= note_max <= 5):
                print("⚠️ La note doit être entre 1 et 5.")
                continue
        except ValueError:
            print("⚠️ Entrée invalide.")
            continue

        # Chargement et exécution
        df = pd.read_csv(path, sep=';', low_memory=False)
        generate_wordcloud(df, note_max, source_type)

if __name__ == "__main__":
    main()