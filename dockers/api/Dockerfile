FROM python:3.12-slim

WORKDIR /app

# L'API a juste besoin de gcc pour compiler certaines extensions si nécessaire
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer uniquement les dépendances de l'API via pip standard
COPY requirements-api.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]