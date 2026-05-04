# Pour notre scrapping

## accés API json
Trust pilote travaille avec aws-waf (pas wtf) AWS WAF (Web Application Firewall)
nous avons une URL api json

https://fr.trustpilot.com/_next/data/businessunitprofile-consumersite-2.6642.0/review/oscaro.com.json?businessunit=oscaro.com

le buildid change régulièrement et à trouvé dans la première page

faire un search buildID dans la page coté dev-Tool

le builID "buildId":"businessunitprofile-consumersite-2.6642.0"

la formated string qui va bien

buildId="businessunitprofile-consumersite-2.6642.0"
f"https://fr.trustpilot.com/_next/data/{buildId}/review/oscaro.com.json?businessunit=oscaro.com"


## les cookies
Cookie                |Rôle
----------------------|--------------------------------------------
aws-waf-token         | 🔑 Le plus important — passe-droit AWS WAF
jwt                   | Token d'authentification (tu es connecté à un compte)
tp-consumer-id        | Ton ID Trustpilot
OptanonConsent        | Consentement cookies (RGPD)
_ga, _gcl_au, _fbp    | Analytics/tracking (inutiles pour scraper)
_hjSession_           | *Hotjar (inutiles)
amplitude_id_*        | Analytics (inutiles)

## aws-waf-token=dd14f0c7-7184-4ed9-a2b6-1cb6d151ff5c
⚠️ Durée de vie
Le aws-waf-token expire (quelques heures à quelques jours). Quand tu reçois à nouveau un 403, il faut retourner dans DevTools et en récupérer un nouveau. C'est la limite de cette approche manuelle — pour l'automatiser il faut revenir à Playwright qui résout le challenge tout seul.

## stabilité de la connection et partage avec un bot cp ou vm

a voir !




est-ce que playwright playwright-stealth est sur et est-ce que ca s'install sur un ubuntu docker ?

18:40
Claude a répondu : Fiabilité de playwright-stealth
Fiabilité de playwright-stealth
Moyen en 2026. La lib Python est mal maintenue — le repo original est quasi abandonné. AWS WAF et les WAF modernes la détectent souvent quand même.

Alternatives plus solides :

rebrowser-playwright — fork de Playwright spécialement anti-détection
nodriver / undetected-chromedriver — pour Selenium
Browserless.io / Apify — services cloud qui gèrent ça pour toi
Installation sur Ubuntu Docker
Oui, ça s'installe, mais il faut les dépendances système de Chromium :

dockerfile
FROM python:3.12-slim

# Dépendances système obligatoires pour Chromium
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    --no-install-recommends && rm -rf /var/lib/apt/lists/*

RUN pip install playwright playwright-stealth
RUN playwright install chromium --with-deps
Et dans ton code, obligatoire en Docker :

python
browser = p.chromium.launch(
    headless=True,
    args=["--no-sandbox", "--disable-dev-shm-usage"]  # ← sans ça, crash garanti
)


## rebrowser-playwright

dockerfileFROM python:3.12-slim

# Dépendances système pour Chromium
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libx11-6 \
    libxcb1 \
    libxext6 \
    wget \
    --no-install-recommends && rm -rf /var/lib/apt/lists/*

RUN pip install rebrowser-playwright requests beautifulsoup4

# Télécharge Chromium via rebrowser
RUN rebrowser-playwright install chromium

WORKDIR /app
COPY . .

CMD ["python", "scraper.py"]
Honnêtement je ne suis pas certain à 100% que rebrowser-playwright install chromium soit la bonne commande — le projet est récent et la doc change.
Vérifie d'abord
bash
pip install rebrowser-playwright

rebrowser-playwright install --help
Ou regarde directement leur repo :
https://github.com/rebrowser/rebrowser-playwright-python



## GESTION DU CHALLENGE DE CONNECTION

si je requete depuis un docker, comment ca se passe, est-ce qu'il peut prendre les cookies de ma session PC ?

18:50
Claude a répondu : Non, Docker ne peut pas accéder aux cookies de ton navigateur directement — c'est deux environnements isolés.
Non, Docker ne peut pas accéder aux cookies de ton navigateur directement — c'est deux environnements isolés. Mais voici comment contourner ça :

La solution simple — passer les cookies en variable d'environnement
bash
# Tu lances ton container avec les cookies en param
docker run -e COOKIE_STRING="aws-waf-token=...;jwt=...;" mon-scraper
python
# Dans ton script Python
import os

cookie_string = os.environ.get("COOKIE_STRING")

# Parser la string en dict
cookies = {}
for part in cookie_string.split(";"):
    if "=" in part:
        k, v = part.strip().split("=", 1)     #c'est là qu'on prend la key/val
        cookies[k] = v

response = requests.get(url, headers=headers, cookies=cookies)
Ou via un fichier .env
bash
# .env
COOKIE_STRING=aws-waf-token=dd14f0c7...;jwt=eyJhbG...
AWS_WAF_TOKEN=dd14f0c7-7184-4ed9-a2b6-1cb6d151ff5c:...
JWT=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
bash
docker run --env-file .env mon-scraper
Workflow concret
1. Tu ouvres trustpilot dans Chrome
2. DevTools → Console → document.cookie
3. Tu copies la string
4. Tu la colles dans ton .env
5. docker run --env-file .env mon-scraper
6. Quand ça re-403 → tu refresh le aws-waf-token
Le vrai problème
Le aws-waf-token est lié à une IP + fingerprint navigateur. Depuis Docker, l'IP est différente de ton PC — AWS WAF peut invalider le token immédiatement même s'il est frais.

Ton PC  →  IP: 90.x.x.x   →  token valide ✅
Docker  →  IP: 172.x.x.x  →  même token   ❌ potentiellement rejeté
Si c'est le cas, la seule option propre reste Playwright dans Docker qui génère son propre token 
-> en résolvant le challenge depuis l'IP du container.