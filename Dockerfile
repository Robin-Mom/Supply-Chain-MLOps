# 4 idée pour être léger
# 1/ image de départ légère
# 2/ gestion du cache uv et cache externe le cache uv n'est pas embarqué dans l'image
# 3/ suppression des outils de build
# 4/ gestion du backend selectif gpu ou cpu à paramétrer ultérieurement

# Utiliser une image Python légère
FROM python:3.12-slim as builder

# Installer les dépendances système nécessaires - ici on est toujours root
#RUN sudo apt-get update && sudo apt-get install -y git build-essential gcc && sudo rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Installation de 'uv' pour une gestion ultra-rapide des packages
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN useradd -m ubuntu
USER ubuntu

# Définir le répertoire de travail
WORKDIR /app

# cpu gpu -> default cpu
ARG BACKEND_EXTRA=cpu

# Copier les fichiers de dépendances
COPY --chown=ubuntu:ubuntu pyproject.toml uv.lock ./

# Installer 'uv' pour gérer les dépendances rapidement - layer lourd
#RUN uv sync --frozen --no-install-project
RUN --mount=type=cache,target=/home/ubuntu/.cache/uv,uid=1000,gid=1000 \
    uv sync --extra ${BACKEND_EXTRA} --no-cache --frozen --no-install-project

# Copier le reste du code (dont le dossier src et models) - invalide le cache
COPY --chown=ubuntu:ubuntu . .

#Installe uniquement le package supply-chain-mlops par-dessus - Très rapide car les deps sont déjà là -Ce layer est léger et se rebuilde vite
RUN --mount=type=cache,target=/home/ubuntu/.cache/uv,uid=1000,gid=1000 \
    uv sync --extra ${BACKEND_EXTRA} --no-cache --frozen

# Étape 2 : Image finale (sans outils de build)
# multi-stage build pour supprimer les outils de build (gcc, build-essential) de l'image finale
FROM python:3.12-slim

# 1. Copier uv AVANT de changer d'utilisateur (nécessite root)
COPY --from=builder /bin/uv /bin/uv
COPY --from=builder /bin/uvx /bin/uvx

RUN useradd -m ubuntu
USER ubuntu
WORKDIR /app
COPY --chown=ubuntu:ubuntu --from=builder /app /app
USER ubuntu

# Exposer le port de l'API
EXPOSE 8000

# Commande pour lancer l'API - pas avec uv run qui refait l'install
CMD [".venv/bin/uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]