# 4 idée pour être léger -> de 10GB à 2.7GB
# 1/ image de départ légère
# 2/ gestion du cache uv et cache externe - le cache uv n'est pas embarqué dans l'image
# 3/ suppression des outils de build par couche secondaire - secondary layer après un builder layer
# 4/ gestion du backend selectif gpu ou cpu à paramétrer via BACKEND_EXTRA et en liens à la section uv/pyproject.tom

# ─── ARG global (avant tout FROM) ──────────────────────────────────────────
ARG BACKEND=cpu          # valeurs : cpu | gpu

# ════════════════════════════════════════════════════════════════════════════
# STAGE 1 – builder  (compile les wheels, n'ira pas en prod dans le runtime)
# ════════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim AS builder

ARG BACKEND
ARG DOCKFLD
RUN echo "Invoqué depuis slim Dockerfile: ${DOCKFLD}"
RUN echo "et donc truc $(pwd)!!"
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# On copie uniquement les fichiers de dépendances

COPY ./requirements-base.txt .
COPY ./requirements-${BACKEND}.txt .

# Installation dans un prefix isolé → facile à copier au stage suivant
RUN pip install --no-cache-dir --prefix=/install \
        -r requirements-${BACKEND}.txt

# reprise de 4. Téléchargement des assets NLP
#RUN python -m spacy download fr_core_news_sm
#RUN python -m nltk.downloader punkt
# Modèles spacy et nltk dans le prefix aussi
RUN PYTHONPATH=/install/lib/python3.12/site-packages \
    python -m spacy download fr_core_news_sm \
    --target /install/lib/python3.12/site-packages

RUN PYTHONPATH=/install/lib/python3.12/site-packages \
    python -m nltk.downloader -d /install/lib/python3.12/nltk_data punkt punkt_tab



# ════════════════════════════════════════════════════════════════════════════
# STAGE 2 – runtime  (image finale, minimale)
# ════════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim AS runtime

ARG BACKEND
# Label informatif embarqué dans l'image
LABEL torch.backend="${BACKEND}"

# Pour GPU : les libs CUDA runtime (si la machine hôte n'expose pas le driver)
# Pour CPU : rien de plus — on garde l'image slim
RUN if [ "$BACKEND" = "gpu" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            libcudnn9-cuda-12 \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# git reste nécessaire pour DVC
RUN apt-get update && apt-get install -y --no-install-recommends \
        git && rm -rf /var/lib/apt/lists/*

# Copie des packages installés depuis le builder
COPY --from=builder /install /usr/local

WORKDIR /app
COPY . .

# Exposer le port de l'API, peut être commun au trainer
EXPOSE 8000

# Commande pour lancer l'API - pas avec uv run qui refait l'install - à voir aussi dans le docker compose

#was CMD [".venv/bin/uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
# Du coup on ne fait pas de repro force par defaut CMD ["dvc", "repro", "--force"] à voir à l'usage
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]