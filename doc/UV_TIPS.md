# uv - reminder ?

    So in our progress to use uv and manage multiple environments let's consider the following scenario 
    - let say we need to test an alternate env, while keeping something working and willing to try a litle alternative.

# summary of combinations
    `uv env .bla`
    `source .bla/Scripts/activate`
    `uv sync`            ## may populate .venv (creating it in the background)
    `uv sync --active`   ## might be dues if missconf bellow
    `uv add foo`
    `uv add foo --active`

## Explanations
if you run *uv venv .bla*, you inherit a *.bla/Scripts/activate* that is hardcoded toward *.bla*, but not only,
also toward its parent folder, preventing move and renaming of the parent folder with ease !!
So what do you need to know if you wish to relocate you projet, case of a redeployment, this is what
we explore here.

if you rename your projet (for any conflict reason) or move it
the activate script becomes obsolete, pointing to the old name and location (bouh)

So one suggestion is to go and update the scripts - but what a hell stupide to modify generated script by hand - at any time you loose your configs

And so yep, you can `rm -rf .venv` because to retrieve the promise is that you simply need a `uv venv .bla` and a `uv sync`
Well, not exaclty.

litle problem as uv is not taking into accout ut UV_VIRTUALENV defined, but goes to kind of hardcoded .venv - over and over

/!\ Hence, despite a proper .bla activation, a first uv sync will sync to .venv silently !!!

All next uv sync will complain that the active .bla is not matching the project environment path

    **warning**: `VIRTUAL_ENV=.bla` does not match the project environment path `.venv` and will be ignored; use `--active` to target the active environment instead

So when running `uv venv .bla`, this will create the .bla env with the new parent folder location
"uv sync" this will rename

IN ORDER TO KEEP uv sync TO the active env, there is two configurations to address

The first one in .env file with 
## .env
`export UV_PROJECT_ENVIRONMENT=.bla`

As a consequence, in a step you have to switch betweed env, it can be that the pyproject.toml might be impacted, the uv.lock as well - so it really depends on the attempt, it might be good to backup the toml

add this .env also to the .gitignore

second file to configure is the .vscode/settings.json, to allow vscode to use the .env injection
## vscode/settings.json
{
    "python.terminal.useEnvFile": true,
    "python.envFile": "${workspaceFolder}/.env"
}


as most possible use

So it is, please take care !


#Next topic is about mlops now and how it articulates with git

## 1. Stratégie Git : Le "GitHub Flow"
Oubliez les branches par personne (ex: branche-marvin). On travaille par fonctionnalité. Si l'un de vous travaille sur la collecte et l'autre sur le dashboard, ils ne doivent pas se marcher sur les pieds.
• main (ou master) : C'est votre "Saint Graal". Le code ici doit toujours être fonctionnel et prêt à être déployé. On ne push jamais directement dessus.
• Branches de fonctionnalités (feature branches) : Chaque fois qu'une tâche commence, créez une branche dédiée :
feat/collecte-data
feat/training-pipeline
feat/api-fastapi
• Pull Requests (PR) : Quand une branche est finie, on fait une PR vers la main. Un des deux autres collègues doit valider le code avant la fusion. C'est le meilleur moyen d'apprendre les uns des autres.

## 2. Division des tâches (Le "Qui fait quoi ?")
Pour votre roadmap, je vous conseille une répartition par "pôles de responsabilité" plutôt que par petits tickets. Voici une proposition pour un binôme/trinôme :
• Membre A (Le Data Engineer / Architecte) :
Mise en place de la structure des dossiers.
Scripts de collecte (collect.py) et de processing (process.py).
Conteneurisation (Docker / Docker-compose).
• Membre B (Le Data Scientist / ML Engineer) :
Pipeline d'entraînement (train.py) et évaluation (evaluate.py).
Tracking avec MLflow (Phase 2).
Optimisation de BERTopic et réduction des thèmes.
• Membre C (Le DevOps / Backend) :
Création de l'API (FastAPI).
Mise en place de la CI/CD (GitHub Actions).
Monitoring (Phase 4).

## 3. Le problème des "Gros Fichiers" (Crucial)
GitHub déteste les fichiers de plus de 50 Mo. Vos modèles BERTopic et vos datasets d'avis vont bloquer vos git push.

`.gitignore` est votre meilleur ami : Ajoutez-y immédiatement les dossiers venv/, __pycache__/, les dossiers de modèles .safetensors et vos fichiers .csv.
Alternative pour les modèles : Pour la Phase 2, vous utiliserez MLflow pour versionner les modèles. En attendant, partagez vos fichiers modèles via un Cloud (Drive, S3) ou utilisez Git LFS (Large File Storage).


## 4. Structure de dossier standard (Cookiecutter ML)
Mettez-vous d'accord dès le premier jour sur cette structure :
Plaintext

PROJET_TRUSTPILOT/
│
├── data/               # Dossier ignoré par Git (contient les CSV)
├── models/             # Dossier ignoré par Git (contient les .safetensors)
├── src/                # Le code source
│   ├── collect.py
│   ├── process.py
│   ├── train.py
│   └── inference.py
├── notebooks/          # Pour vos tests brouillons
├── app/                # Code Streamlit ou FastAPI
├── tests/              # Tests unitaires (Phase 3)
├── Dockerfile
├── requirements.txt
├── README.md 
└──.gitignore


Mon conseil "Wit" :

`I 'm a poor Lonesome Coder, and a long way from /home`

Ne devenez pas des "codeurs solitaires". Le MLOps est une discipline de collaboration. Si le membre A change le format du CSV dans process.py sans prévenir le membre B qui entraîne le modèle, tout explose. Communiquez sur Slack/Discord à chaque fusion de branche !

Les agents IA ne procèdent pas autrement !

## Mise en oeuvre

Sur un dossier local, ou remote sur la vm, peut import

- créer le dossier 

- y préparer un README.md

créer l'aroborescence de dossier et les fichiers vide

`mkdir data models src notebooks app tests Dockerfile`

`touch src/collect.py src/process.py src/train.py src/inference.py requirement.txt .gitignore`

## on ajoute des .gitkeep pour garder la structure des dossiers et assurer un clonage facile
`for f in data models notebooks app tests Dockerfile; do touch $f/.gitkeep; done`

Note Dockerfile

    The docker build command expects the file name to exactly be Dockerfile. So, in that case, you can simply do

    `docker build .`

    In other cases, you have to specify the full name as

    `docker build -f Dockerfile.build .`

remplir le .gitignore dés le début

```
# Python-generated files
__pycache__/
*.py[oc]
build/
dist/
wheels/
data/

*.egg-info

# Virtual environments
.venv
.sl

# Modèles lourds on commente si on passe en git LFS
# *.h5
# *.keras
# *.pkl
# *.csv

# IDE
*.vscode
!.vscode/settings.json
.env
.streamlit/secrets.toml
```

## initialiser git

`git init`

## Ajoutez votre URL remote (récupérez-la sur votre page GitHub)
au passage example de Yohan https://github.com/DataScientest/exam_Bash_MLOps# 
et rappel sur cookiecutter https://github.com/cookiecutter/cookiecutter

## créer le repo sur github avec `new`
https://github.com/new

## brancher le repo local au repo remote via http
    `git remote add origin https://github.com/schmilblick-ai/Supply-Chain-MLOps.git`

ou via ssh (avec ssh key defini dans votre home directory machine / user)

    `git remote add origin git@github.com:schmilblick-ai/Supply-Chain-MLOps.git`

et si on change d'avis ?? on peut refaire mais il faut d'abord un remove

    `git remote remove origin`


## La taille du Repo à regarder de temps en temps
    `curl -s https://api.github.com/repos/USER/REPO | grep size`
    `curl -s https://api.github.com/repos/schmilblick-ai/Supply-Chain-MLOps | grep size`

## vérification du git remote
    `git remote -v`

## premier commit propre, staging (litéralement exposition sur la scène ou sortie de la coulisse) et commit
    `git add .`
    `git commit -m "Initial commit: Tabula Rasa (Clean start)"`

## un premier push
    `git push`

et non pas assez

## To push the current branch and set the remote as upstream, use we also need to force due to remote creation and to avoid an initial pull
    `git push --force --set-upstream origin main`


Merci à Stéphane Robert pour ses innombrables tuyaux

https://blog.stephane-robert.info/docs/conteneurs/images-conteneurs/optimiser-taille-image/
