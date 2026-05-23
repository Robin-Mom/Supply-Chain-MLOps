
# Variables possible
BACKEND   ?= cpu
IMAGE     := supply-chain-mlops-slim
TAG       := $(IMAGE):$(BACKEND)
DOCKFLD   :=
DOCK      := Dockerfile
# a project variable that will server to prefix the images and containers and identify the ressources for cleaning
# mainly used in -p option of docker compose -d or idn docker-compose
PROJECT   := pr001

#definition du docker_compose disponible
DOCKER_COMPOSE := $(shell which docker-compose 2>/dev/null || echo "docker compose")

# definition de AIRFLOW_UID ou bien dans .env (mais avec restriction de config vscode)
export AIRFLOW_UID := $(shell id -u)
export AIRFLOW_GID := 0

.DEFAULT_GOAL := help

# ── Phony ─────────────────────────────────────────────────────
#target not to take for files but command lines are liste in the .PHONY statement
.PHONY: help secrets airflow_init airflow_prepdocker_sock airflow_up airflow_force_recreate airflow_down airflow_reset airflow_ps up down slmbuild slmbuild-cpu slmbuild-gpu slmbuild-test

# ── Help ──────────────────────────────────────────────────────
help: ## this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Build dev Certificate Authority and Certificates for 443 ──
secrets:
	./dockers/nginx/certs/secretgen.sh localhost ./dockers/nginx/certs

# ── Airflow ────────────────────────────────────────────────────
# AIRFLOW INIT SECTION TO ALWAYS RUN FIRST TIME DEPLOYED ON A NEW DEVICE
airflow_init: ## Initialiser airflow
	$(DOCKER_COMPOSE) -f dockers/airflow/docker-compose.yaml up airflow-init

#La commande ci-dessus est à exécuter à chaque redémarrage de votre machine virtuelle sinon l'erreur suivante sera levée 
# : docker.errors.DockerException: Error while fetching server API version: ('Connection aborted.', PermissionError(13, 'Permission denied'))
# prereq juste pour utiliser le DockerOperator
airflow_prepdocker_sock:
	sudo chmod a+rw /var/run/docker.sock

airflow_up: ## alumer airflow
	$(DOCKER_COMPOSE) -f dockers/airflow/docker-compose.yaml up -d

# AIRFLOW SECTION
airflow_force_recreate:  ## force la recréation de airflow
	$(DOCKER_COMPOSE) -f dockers/airflow/docker-compose.yaml up --force-recreate

airflow_down: ## éteindre airflow
	$(DOCKER_COMPOSE) -f dockers/airflow/docker-compose.yaml down 

airflow_reset: 
	echo "$(DOCKER_COMPOSE) down -v"

airflow_ps:  ## list process runnings $(DOCKER_COMPOSE) -f airflow/docker-compose.yaml ps
	$(DOCKER_COMPOSE) -f dockers/airflow/docker-compose.yaml ps

# ── Global ────────────────────────────────────────────────────
# GLOBAL SECTION mixte airflow et le docker-compose standard et a venir nginx
up: ## lance tout les dockers
	$(DOCKER_COMPOSE) -f dockers/airflow/docker-compose.yaml up -d
	$(DOCKER_COMPOSE) -f ./docker-compose.yaml up -d

down: ## ferme tout les dockers
	$(DOCKER_COMPOSE) -f ./docker-compose.yaml down
	$(DOCKER_COMPOSE) -f dockers/airflow/docker-compose.yaml down

#exemple
# Image CPU (~2.5 GB au lieu de 12 GB)
# docker build --build-arg BACKEND=cpu -t supply-chain-mlops-trainer:cpu .

# Image GPU (CUDA 12.4)
# docker build --build-arg BACKEND=gpu -t supply-chain-mlops-trainer:gpu .

# ── slimBuild ─────────────────────────────────────────────────
# construction avec le dockerfile spécifié et la target
slmbuild: ## Build l'image trainer (BACKEND=cpu|gpu)  #--no-cache --progress=plain 
	docker build \
	--build-arg BACKEND=$(BACKEND) \
	--build-arg DOCKFLD=$(DOCKFLD) \
	-f $(DOCKFLD)$(DOCK) -t $(TAG) .

slmbuild-cpu: ## Build CPU explicitement
	$(MAKE) slmbuild BACKEND=cpu DOCKFLD=dockers/slim/

slmbuild-gpu: ## Build GPU (CUDA 12.4) explicitement
	$(MAKE) slmbuild BACKEND=gpu DOCKFLD=dockers/slim/

#  pour appeler une target Make depuis une autre target, 
#  c'est toujours $(MAKE) <target> VAR=valeur
#  et TAG est une fonction f(IMAGE,BACKEND)
slmbuild-test: ## Build image de test CPU :: make slimbuild-test
	$(MAKE) slmbuild-cpu IMAGE=supply-test


#docker image for nginx proxy server & all
start-project:
	# was  docker-compose up --build api
	docker compose -p $(PROJECT) -f docker_compose1.yaml up -d --build

log-project:
	docker compose -p $(PROJECT) logs
	
stop-project:
	docker compose -p $(PROJECT) down -v

