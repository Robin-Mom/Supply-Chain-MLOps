import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time
import random

def fetch_truspilot_reviews(site="www.oscaro.com",nb_pages=5):
    base_url = f"https://fr.trustpilot.com/review/{site}"
    all_data = []
    
    # Headers pour simuler un vrai navigateur
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7"
    }

    for page in range(1, nb_pages + 1):
        print(f"Extraction de la page {page}...")
        
        # Gestion de l'URL pour la pagination
        url = f"{base_url}?page={page}"
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # On cherche la balise magique qui contient toutes les infos
            script_tag = soup.find('script', id='__NEXT_DATA__')
            
            if script_tag:
                data = json.loads(script_tag.string)
                # Parcours du dictionnaire JSON pour atteindre les avis
                reviews_list = data['props']['pageProps']['reviews']
                
                for rev in reviews_list:
                    all_data.append({
                        'Date': rev.get('createdAt'),
                        'Auteur': rev.get('consumer', {}).get('displayName'),
                        'Note': rev.get('rating'),
                        'Titre': rev.get('title'),
                        'Commentaire': rev.get('text'),
                        'Réponse_Oscaro': rev.get('reply', {}).get('message') if rev.get('reply') else "Pas de réponse"
                    })
            
            # Pause aléatoire pour ne pas être banni (comportement humain)
            time.sleep(random.uniform(1.5, 3.0))
            
        except Exception as e:
            print(f"Erreur sur la page {page}: {e}")
            break

    # Sauvegarde des données
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv('avis_oscaro.csv', index=False, encoding='utf-16')
        print(f"\n✅ Succès ! {len(all_data)} avis enregistrés dans 'avis_oscaro.csv'.")
    else:
        print("❌ Aucune donnée n'a pu être extraite.")

# Lancement pour les 3 premières pages (tu peux changer le chiffre)
#fetch_truspilot_reviews(nb_pages=3)

def fetch_truspilot_reviews1(site="www.oscaro.com",nb_pages=5):
    return site,nb_pages