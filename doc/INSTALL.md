
# Problématique d'access des notebooks au code dans src
cela semble trivial, mais est un point d'attention

On fait deux choses en parallèle : mettre un __init__.py dans src et déclarer le projet en mode éditable

## __init__.py
```bash
# 1. créer les __init__.py
touch src/__init__.py
```

Il peut y avaoir des instructions maline dedans, mais en général c'est juste vide


## 2. installer le projet en mode éditable (⚠️ version pip)
pip install -e .
Et colle ça en cellule 1 de tous tes notebooks :

```python
python%load_ext autoreload
%autoreload 2

from src.models import load_bertopic_model
from src.data_loader import load_vectors
```

Pas de sys.path bricolé, imports propres, et les modifs dans src/ sont visibles immédiatement sans redémarrer le kernel. C'est exactement ce setup qu'on retrouve dans les projets ML sérieux (Kedro, cookiecutter-data-science, etc.).



# plus propre avec uv — c'est le workflow recommandé aujourd'hui.
```bash
# avec uv (moderne)
uv pip install -e .
```
Même effet, mais uv est 10-100x plus rapide et gère le virtual env proprement.

# Le workflow complet avec uv depuis zéro

# 1. créer le projet et son venv
```bash
uv init my-project       # crée pyproject.toml, .python-version, etc.
cd my-project
```

# 2. créer la structure

```bash
mkdir src notebooks
touch src/__init__.py
```

# 3. installer en mode éditable (lit pyproject.toml)
```bash
uv pip install -e .
```

# 4. ajouter des dépendances
uv add  ... (streamlit bertopic sentence-transformers notebook)

La différence importante : uv run

Avec uv, le venv est isolé. Pour que Jupyter utilise le bon kernel (celui du venv), deux options :
## Option A — lancer Jupyter via uv run (le plus simple) :
```bash
uv run jupyter notebook
# ou
uv run jupyter lab
```
Le notebook tourne dans le venv du projet → src/ est importable directement, sans sys.path.

## Option B — enregistrer le kernel (si tu utilises un Jupyter global) :
```bash
uv add --dev ipykernel
uv run python -m ipykernel install --user --name my-project
```

Puis sélectionner le kernel my-project dans Jupyter.

## pyproject.toml minimal qui marche avec uv
uv init génère déjà un pyproject.toml, tu l'adaptes juste :
```toml
toml[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "streamlit",
    ... bla bla ...
    "pandas",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

Puis :
```bash
uv pip install -e .
```

Et dans tes notebooks, sans aucun sys.path :
```python
python%load_ext autoreload
%autoreload 2

from src.models import load_bertopic_model   # ça marche directement
```

En résumé
 
|  | `pip` | `uv` |
| --- | --- | --- |
| Installer en éditable | `pip install -e .` | `uv pip install -e .` |
| Ajouter une dépendance | `pip install X` + maj manuel | `uv add X` (maj pyproject.toml auto) |
| Lancer Jupyter | `jupyter notebook` | `uv run jupyter notebook` |
| Vitesse | lente | 10-100x plus rapide |
| Lock file | `requirements.txt` manuel | `uv.lock` automatique |
 
Le point clé : uv run jupyter notebook est la commande à retenir — elle garantit que le notebook voit exactement le même environnement que ton app.py.