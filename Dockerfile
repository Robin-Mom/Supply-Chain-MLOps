# Utiliser une image Python légère
FROM python:3.12-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y git build-essential gcc&& rm -rf /var/lib/apt/lists/*

# Installation de 'uv' pour une gestion ultra-rapide des packages
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copier les fichiers de dépendances
COPY pyproject.toml uv.lock ./

# Installer 'uv' pour gérer les dépendances rapidement
ENV UV_LINK_MODE=copy

RUN uv sync --frozen --no-install-project

# Copier le reste du code (dont le dossier src et models)
COPY . .

# Exposer le port de l'API
EXPOSE 8000

# Commande pour lancer l'API
CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]