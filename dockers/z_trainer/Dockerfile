FROM python:3.12-slim

# 1. On définit un argument de build avec "cpu" par défaut (pour toi et la CI)
ARG DEVICE_TYPE=cpu

WORKDIR /app

# Le trainer a obligatoirement besoin de git pour DVC
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 2. Installation conditionnelle de PyTorch AVANT le reste du requirements
# On force la version CPU légère si demandé, sinon pip installera la version CUDA par défaut
RUN if [ "$DEVICE_TYPE" = "cpu" ] ; then \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu ; \
    else \
        pip install --no-cache-dir torch torchvision ; \
    fi

# 3. Copier et installer le reste des dépendances de DVC/Train via pip standard
COPY requirements-trainer.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Téléchargement des assets NLP
RUN python -m spacy download fr_core_news_sm
RUN python -m nltk.downloader punkt

COPY . .

# Par défaut, ce conteneur lance le pipeline DVC
CMD ["dvc", "repro", "--force"]