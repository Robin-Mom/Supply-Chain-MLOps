FROM python:3.12-slim

WORKDIR /app

# Le trainer a obligatoirement besoin de git pour DVC
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer uniquement les dépendances de DVC/Train via pip standard
COPY requirements-trainer.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download fr_core_news_sm
RUN python -m nltk.downloader punkt

COPY . .

# Par défaut, ce conteneur lance le pipeline DVC
CMD ["dvc", "repro", "--force"]