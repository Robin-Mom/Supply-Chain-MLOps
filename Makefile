#import .env
include .env
export

# Variables possible
BACKEND   ?= cpu
IMAGE     := supply-chain-mlops-slim
TAG       := $(IMAGE):$(BACKEND)
DOCKFLD   := dockers/slim/
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
.PHONY: help secrets airflow_init airflow_prepdocker_sock airflow_up airflow_force_recreate airflow_down airflow_reset airflow_ps dvc-repro dvc-push up down slmbuild slmbuild-cpu slmbuild-gpu slmbuild-test start-api start-airflow start-ip-conditional clean cleanDandling

# ── Help ──────────────────────────────────────────────────────
help: ## this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Build dev Certificate Authority and Certificates for 443 ──
secrets:
	#note j'ai mis une IP variable 108.130.252.7 mais on en a pas besoin - juste pour que le dossier certs soit en 4ème param
	./dockers/nginx/certs/secretgen.sh localhost LIORA-VM-77Gi 108.130.252.7 ./dockers/nginx/certs

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

# ── dvc ────────────────────────────────────────────────────
# La règle à afficher dans le README ou le Makefile
# UN fichier ne peut être tracké QUE par l'un des deux :
# Fichier SOURCE    →  .dvc file    (dvc add)
# Fichier GÉNÉRÉ   →  dvc.lock     (dvc repro)
# Si les deux existent pour le même fichier → PROBLÈME

FORCE ?= 0
dvc-mergeInc:  ## merge a set with incremental overlapping 40K ith 28k
	@echo "--- dvc merge incremental force=$(FORCE)" #" $(if $(filter 1,$(FORCE)),--force,)"
	dvc repro --downstream mergInc $(if $(filter 1,$(FORCE)),--force,)

dvc-repro:
	dvc repro $(if $(filter 1,$(FORCE)),--force,)

dvc-check:  ## pour faire la validation dvc
	@echo "--- Git status ---"
	@git status
	@echo ""
	@echo "--- DVC status ---"
	@dvc status
	@echo ""
	@echo "--- Fichiers dans dvc.lock ---"
	@grep "path:" dvc.lock
	@echo ""
	@echo "--- Fichiers .dvc statiques (tracking résiduel) ---"
	@find . -name "*.dvc" -not -path "./.dvc/*"
	@echo ""
	@echo "--- Doublons OUTS + statique (problématique) ---"
	@for f in $$(find . -name "*.dvc" -not -path "./.dvc"); do \
		fname=$$(grep "path:" $$f | awk '{print $$2}'); \
		if grep -A 5 "outs:" dvc.lock 2>/dev/null | grep -q "$$fname"; then \
			echo "⚠️  DOUBLON OUTS : $$fname"; \
		fi \
	done
	@echo ""
	@echo "--- Fichiers statiques en DEPS (normal) ---"
	@for f in $$(find . -name "*.dvc" -not -path "./.dvc"); do \
		fname=$$(grep "path:" $$f | awk '{print $$2}'); \
		if grep -A 5 "deps:" dvc.lock 2>/dev/null | grep -q "$$fname"; then \
			echo "✅ DEPS OK : $$fname"; \
		fi \
	done


dvc-push: ## on push les fichiers généré par une repro ou les fichiers add externe
	dvc push
	git add dvc.lock
	git commit -m "update pipeline"
	git push

dvc-utest: ## Unit testing dvc on local host before going on prod - à customizer selon besoin
	#15 mn pour 4000 avis
	export TRAIN_ARGS="--exp_name Bertopic_Trustpilot_v2" && dvc repro --downstream train --force


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

slmtestconfig: ## Testing the docker compose config
	docker compose -f docker-compose1.yaml config |less #| grep -A 10 "args"
	# On voit les bon transferts d'arguments

#docker image for nginx proxy server & all
start-project:
	# was  docker-compose up --build api
	docker network create airflow_default 2>/dev/null || true
	$(DOCKER_COMPOSE) -p $(PROJECT) -f docker-compose1.yaml up -d --build 
	#--dry-run

log-project:
	$(DOCKER_COMPOSE) -p $(PROJECT) -f docker-compose1.yaml logs
	
stop-project:
	$(DOCKER_COMPOSE) -p $(PROJECT) -f docker-compose1.yaml down -v

diag-project:
	$(DOCKER_COMPOSE) -p $(PROJECT) -f docker-compose1.yaml config


# Détecte la machine automatiquement - I car on travaillera en IP pas en dns pour la config nginx
CURRENT_IP := $(shell hostname -I | awk '{print $$1}')

start-api:  ## starting grafana < prometheus < api < nginx
	@echo "Lancement API sur $(CURRENT_IP)"
	docker network create airflow_default 2>/dev/null || true
	$(DOCKER_COMPOSE) -p $(PROJECT) -f docker-compose1.yaml up -d --build


start-airflow: ## airflow start separation to consider on localhost mono approad
	@echo "Lancement Airflow sur $(CURRENT_IP)"
	$(DOCKER_COMPOSE) -p $(PROJECT) -f docker-compose1.yaml up -d

start-ip-conditional: ## 👍 main starter that separate for starting on different hosts
	@if [ "$(CURRENT_IP)" = "${API_HOST}" ]; then \
		make start-api; \
	elif [ "$(CURRENT_IP)" = "${AIRFLOW_HOST}" ]; then \
		make start-airflow; \
	fi

clean: ## Disk Space Recycling
	@ echo "=== Initial State ==="
	@ docker system df
	# Supprimer les dangling images
	@ docker image prune -f
	@ docker container prune -f
	@ docker volume prune -f
	@ docker builder prune -af
	@ echo "=== Post Cleaning State ==="
	@ docker system df

cleanDandling: ## Disk Space Recycling
	# Supprimer les dangling images
	@ docker image prune -f


nginxConfReload: ## reload a chaud pour tester un changement de config nginx
	docker exec nginx_revproxy nginx -s reload

nginxLogs: ## view the logs
	docker compose -p pr001 -f docker-compose1.yaml logs -f	