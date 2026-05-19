# gpu cpu variant, taille du cache uv et ci pip installation

[(about markdows extentions)](https://code.visualstudio.com/docs/languages/markdown#_markdown-extensions)

[[_TOC_]]

## 1. question cache uv - pratique pour compresser le cache entre 2 ci/cd

Etude sur les pratique de cache uv compressé

pas de mention d'usage et de lecture à la volé, mais référence à des pratiques de compression entre ci/cd ? stockage du cache

le compropis étant la vitesse de restauration du cache en local via tar -xzf vs le réuse from scratch, sans cache sur la vm

ainsi,

```bash
# Compresser le cache uv
tar -czf uv-cache.tar.gz -C ~/.cache/uv .

# Restaurer le cache avant utilisation
mkdir -p ~/.cache/uv
tar -xzf uv-cache.tar.gz -C ~/.cache/uv

# uv utilisera automatiquement le cache restauré
uv pip install -r requirements.txt
```

uv vérifie l'intégrité et les métadonnées des fichiers du cache. Si les timestamps ou permissions sont altérés par la compression, il peut ignorer le cache et retélécharger quand même.

C'est une approche courante en CI/CD (GitHub Actions, GitLab CI) pour persister le cache entre les jobs.

## 2. QUESTION  gpu cpu avec pip
 

```bash
requirements/
├── requirements-cpu.txt
├── requirements-gpu.txt
└── requirements-base.txt   # dépendances communes

txt# requirements-cpu.txt
-r requirements-base.txt
--index-url https://download.pytorch.org/whl/cpu
torch
torchvision
torchaudio

txt# requirements-gpu.txt
-r requirements-base.txt
torch
torchvision
torchaudio
```


```bash
# Installation CPU
uv pip install -r requirements-cpu.txt

# Installation GPU (index PyPI par défaut)
uv pip install -r requirements-gpu.txt
```


Recommendation : dans un contexte Airflow + Docker, l'Option A est la plus lisible et la plus simple à intégrer dans un Dockerfile :
```dockerfile
ARG TORCH_VARIANT=cpu
COPY requirements-${TORCH_VARIANT}.txt .
RUN uv pip install -r requirements-${TORCH_VARIANT}.txt
``` 

```bash
docker build --build-arg TORCH_VARIANT=gpu -t mon-image .

# mise à jour pip freeze sans torch pour la base

bashpip freeze | grep -v torch > requirements.txt

Si tu veux exclure tout ce qui est lié à torch (torchvision, torchaudio, etc.) :
bashpip freeze | grep -Ev "torch|torchvision|torchaudio" > requirements.txt

Avec uv :
bashuv pip freeze | grep -Ev "torch|torchvision|torchaudio" > requirements.txt

Astuce — si tu as plusieurs packages à exclure régulièrement, crée un script :
bash#!/bin/bash
# torchfreefreeze.sh
EXCLUDE="torch|torchvision|torchaudio|triton|nvidia"
TGTrequirements=$1
pip freeze | grep -Ev "$EXCLUDE" > $TGTrequirements
echo "✅ requirements.txt généré sans les packages GPU/torch"

# call: cet instruction fabrique ou met à jour le requirements-base.txt avec les filtres
./torchfreefreeze.sh requirements-base.txt

#et le buid docker avec le TORCH_VARIANT
docker build --build-arg TORCH_VARIANT=gpu -t mon-image .


peut aller dans un make file

```


```Makefile
# def d'une variable et de sa valeur par defaut cpu avec override externe possible
TORCH_VARIANT ?= cpu

freeze:
	pip freeze | grep -Ev "torch|torchvision|torchaudio|triton|nvidia" > requirements-base.txt
	@echo "✅ requirements-base.txt généré sans torch"

install:
	uv pip install -r requirements-base.txt
	uv pip install -r requirements-$(TORCH_VARIANT).txt
	@echo "✅ Installation terminée avec variante $(TORCH_VARIANT)"

build:
	docker build --build-arg TORCH_VARIANT=$(TORCH_VARIANT) -t mon-image .
	@echo "✅ Image buildée avec variante $(TORCH_VARIANT)"

```

```bash
# Valeur par défaut (cpu)
make install
make build

# Avec GPU explicite
make install TORCH_VARIANT=gpu
make build TORCH_VARIANT=gpu

# Freeze (sans torch, sans variante)
make freeze
```



## 3. QUESTION uv et les pratiques de compression de cache
question pratique du cache uv compressé, est-ce que tu peux élaborer sur l'usage et la praticité. 
Pourquoi on le fait, qu'elle est l'intérêt, quelle est le problème qu'on souhaite résoudre ?

le cache, c'est rapide, mais c'est gros !

Alors ?


Claude a répondu : 

Le problème qu'on cherche à résoudre
uv est ultra-rapide si le cache est déjà là. Mais le cache peut peser plusieurs GB (surtout avec torch). Donc la question est :

Comment transporter ce cache d'un endroit à un autre sans retélécharger tout depuis PyPI ?

Les cas d'usage typiques
### 1. CI/CD (GitHub Actions, GitLab CI)
Chaque job repart de zéro. Sans cache → retélécharge torch à chaque pipeline.

```yaml
# GitHub Actions - persist le cache entre les runs
- uses: actions/cache@v3
  with:
    path: ~/.cache/uv
    key: uv-${{ hashFiles('requirements*.txt') }}
```

Ici le cache est géré nativement par la plateforme CI, pas besoin de tar.gz.

### 2. Environnement isolé / air-gap
Pas d'accès internet, ou accès limité. On prépare le cache sur une machine connectée, on le compresse, on le transfère.

```bash
# Machine avec internet
uv pip install -r requirements-gpu.txt
tar -czf uv-cache.tar.gz -C ~/.cache/uv .

# Transfert
scp uv-cache.tar.gz user@serveur-isolé:/tmp/

# Machine isolée
mkdir -p ~/.cache/uv
tar -xzf /tmp/uv-cache.tar.gz -C ~/.cache/uv
uv pip install -r requirements-gpu.txt  # aucun téléchargement
```

C'est le cas d'usage le plus légitime du tar.gz.

### 3. Docker build sans cache registry
```dockerfile
# Copier le cache uv dans l'image au build
COPY uv-cache.tar.gz /tmp/
RUN mkdir -p /root/.cache/uv && \
    tar -xzf /tmp/uv-cache.tar.gz -C /root/.cache/uv && \
    uv pip install -r requirements-gpu.txt
```

Utile si tu rebuildes souvent la même image sans Docker layer cache.

Mais alors, le tar.gz ça vaut le coup ?

```
Situation	                          tar.gz utile ?
CI/CD avec cache natif (GH Actions)	❌ inutile, la plateforme gère
Serveur isolé / air-gap	            ✅ indispensable
Partager un env entre collègues     ✅ pratique
Docker build classique              ❌ préférer un registry ou --mount=type=cache
Laptop → laptop                     ✅ si même OS/arch

```
Le vrai problème : la taille ;
Un cache uv avec torch GPU peut faire 3-5 GB. Compressé en tar.gz :

torch se compresse mal (binaires déjà optimisés) → gain ~10-20% seulement, Transférer 3 GB reste long

Alternatives plus intelligentes :

```bash
# 1. Cache uv avec liens symboliques (pas de copie)
uv pip install --link-mode=hardlink -r requirements.txt

# 2. Ne cacher que les wheels, pas les sources
tar -czf uv-wheels.tar.gz -C ~/.cache/uv/wheels .
# Beaucoup plus léger que le cache complet

# 3. Monter un cache partagé en réseau (NFS, S3)
# Tous les workers partagent le même cache → pas de transfert
UV_CACHE_DIR=/mnt/shared/uv-cache uv pip install -r requirements.txt
```

### Conclusion
Le tar.gz du cache uv est un outil de transport ponctuel, pas une stratégie de cache quotidienne.

Le vrai gain de uv c'est la vitesse de résolution et d'installation, pas la compression. Si tu veux optimiser le cache, la meilleure approche est un cache partagé monté en réseau ou le cache natif de ta plateforme CI.


# QUESTION uv taille et version multiple de python
s'il y a plusieurs version de python, est-ce que le cache uv augmente significativement ?
et si on nous donne un cache, comment savoir quelle version de python sont compatibles avec ?

Claude a répondu : 

Cache uv et versions Python multiples
Est-ce que le cache grossit significativement ?
Oui, mais moins qu'on ne le pense, car uv est intelligent :

```bash
~/.cache/uv/
├── wheels/         # ← C'est ici que ça grossit
│   ├── torch-2.3.0-cp310-cp310-linux_x86_64.whl   # Python 3.10
│   ├── torch-2.3.0-cp311-cp311-linux_x86_64.whl   # Python 3.11
│   └── torch-2.3.0-cp312-cp312-linux_x86_64.whl   # Python 3.12
├── sdists/         # ← Partagé entre toutes les versions Python
└── interpreter/    # ← Métadonnées par version
```

Les sdists et métadonnées sont partagés. Seules les wheels compilées sont dupliquées par version Python.

Pour torch spécifiquement :

Versions Python	Taille approximative
1 version	~2.5 GB
3 versions	~6-7 GB
5 versions	~11-12 GB


Comment savoir quelles versions Python sont dans un cache ?

```bash
# 1. idée : Inspecter les wheels directement
ls ~/.cache/uv/wheels/
# torch-2.3.0-cp310-cp310-linux_x86_64.whl
#              ^^^^
#              cp310 = CPython 3.10

# 2. Script plus lisible
find ~/.cache/uv/wheels -name "*.whl" | \
  grep -oP 'cp\d+' | \
  sort -u | \
  sed 's/cp\([0-9]\)\([0-9]*\)/Python 3.\2/'



Décoder le nom d'une wheel :

torch-2.3.0  -  cp310  -  cp310  -  linux_x86_64  .whl
    │               │                     │
  version      Python 3.10            plateforme
Les tags possibles :

Tag	Signification
cp310	CPython 3.10
cp311	CPython 3.11
cp312	CPython 3.12
py3	Compatible toutes versions Python 3
none-any	Pure Python, toutes versions
```


Script complet d'audit d'un cache reçu
Voir le script corriger [fichier audit-uv-cache.sh](audit-uv-cache.sh)


```bash
# Appel sur un cache extrait d'un tar.gz
tar -xzf uv-cache.tar.gz -C /tmp/uv-cache-audit
chmod +x audit-uv-cache.sh
#call
./audit-uv-cache.sh /tmp/uv-cache-audit

#sinon appel simple pour le cache en local et faire un diagnostique
./audit-uv-cache.sh

```

Conclusion

Si tu reçois un cache uv, l'info est encodée dans le nom des wheels — pas besoin de métadonnées externes. Le script d'audit te donne en 10 secondes les versions Python compatibles et la taille par version.

Autre point, sans rentrer dans le détail:

° **uv cache prune** va cleaner le cache si des projets ont été rm (pas mv sur la même partitition, car c'est un lookup sur les .venv locaux avec hardlink insensible au mv - sauf si mv entre différente partitition car il y a alors reconstruction d'inode)

° **uv cache clean** - netoyage complet, par exemple mensuel ou trimestriel, ou si pb de taille, le cache se reconstruit progressivement avec les projets suivants

Donc le cache n'est pas primordiale, accélère le temps sur plusieurs run, pas sur un seul
-> sur chaque plateform ou


# QUESTION FINALE
donc est-ce que uv est pratique dans le cas d'installation sur docker ? 
en dev (beaucoup de cycle de mise au point ? 
en prod (install des dépendances à l'allumage du container) ?

qu'est-ce qu'on peut conclure ?

Claude synthétise : Analysons les deux contextes :

En développement Docker — beaucoup de cycles
```dockerfile
# Pattern typique sans optimisation
RUN pip install -r requirements.txt   # ← retélécharge à chaque rebuild
```

uv apporte :

```dockerfile
RUN pip install uv
RUN uv pip install -r requirements.txt  # ← 10x plus rapide
```

Mais le vrai gain en dev c'est le mount cache Docker :

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt
``` 

Le cache uv est persisté entre les rebuilds Docker sur la même machine. Résultat : seules les dépendances qui changent sont retéléchargées.

Conclusion dev : uv + mount cache = excellent, les cycles de mise au point sont vraiment accélérés.

En production — install à l'allumage du container
C'est une mauvaise pratique en général, indépendamment de uv :

```bash
# Anti-pattern prod
docker run mon-image
```

→ installe les dépendances au démarrage  # ← fragile, lent, PyPI requis
En prod le conteneur doit être immuable et autosuffisant :

```dockerfile
# Bonne pratique : dépendances installées au BUILD, pas au RUN
FROM python:3.12-slim

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt
# L'image contient déjà tout → démarrage instantané
```

Conclusion prod : uv n'apporte rien si les dépendances sont installées au démarrage — et cette pratique est à éviter de toute façon.

Le vrai atout de uv en Docker : le multi-stage
```dockerfile
# Stage 1 : build
FROM python:3.12-slim AS builder
RUN pip install uv
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --target=/app/deps -r requirements.txt

# Stage 2 : image finale légère
FROM python:3.12-slim
COPY --from=builder /app/deps /app/deps
COPY . /app
ENV PYTHONPATH=/app/deps
``` 

L'image finale ne contient ni uv, ni le cache → légère et propre.

Tableau de synthèse
```BASH
Contexte	                uv utile ?	         Pourquoi
Devlocal                  ✅ très utile	     rapidité, isolation par projet
DevDocker + mount cache   ✅ excellent	       cycles rapides, cache persisté
BuildDockerCI/CD          ✅ bon	             build plus rapide
Prod install au démarrage ❌ anti-pattern	   à éviter, uv ne change rien  - ANTI PATTERNE TRES FREQUENTE EN FOURNITURE D'IMAGE EN ENTREPRISE
Prod image immuable       ✅ utile au build	 image construite plus vite
```

# Conclusion générale

uv est un outil de build et de résolution, pas un gestionnaire de runtime. Son terrain de jeu c'est le moment où on installe des dépendances. Plus ce moment est fréquent (dev, CI), plus uv est précieux. En prod, si l'image est bien construite, uv a déjà fait son travail avant le démarrage du container.