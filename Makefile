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
PROJECT   := pr007

#definition du docker_compose disponible
DOCKER_COMPOSE := $(shell which docker-compose 2>/dev/null || echo "docker compose")

# definition de AIRFLOW_UID ou bien dans .env (mais avec restriction de config vscode)
export AIRFLOW_UID := $(shell id -u)
export AIRFLOW_GID := 0

.DEFAULT_GOAL := help

# ── Phony ─────────────────────────────────────────────────────
#target not to take for files but command lines are liste in the .PHONY statement
.PHONY: help secrets airflow_init airflow_prepdocker_sock airflow_up airflow_force_recreate airflow_down airflow_reset airflow_ps dvc-repro dvc-push up down slmbuild slmbuild-cpu slmbuild-gpu slmbuild-test start-api start-airflow start-ip-conditional clean cleanDandling graf_ps graf_inspect graf_logs

# ── Help ──────────────────────────────────────────────────────
help: ## 👍 this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Build dev Certificate Authority and Certificates for 443 ──
secrets:  ## 👍 Generate CA.crt and nginx.crt in the dockers/nginx/certs area for dev
	#note j'ai mis une IP variable 108.130.252.7 mais on en a pas besoin - juste pour que le dossier certs soit en 4ème param
	./dockers/nginx/certs/secretgen.sh localhost LIORA-VM-77Gi 108.130.252.7 ./dockers/nginx/certs

# ── Airflow ────────────────────────────────────────────────────
# AIRFLOW pull ? to be confirmed
airflow_pull:  ## To be validated on the prereqs for airflow
	docker compose -f dockers/airflow/docker-compose.yaml pull

# AIRFLOW INIT SECTION TO ALWAYS RUN FIRST TIME DEPLOYED ON A NEW DEVICE
airflow_init: ## Initialiser airflow - ⚠️ airflow_init(evaluated)
	$(DOCKER_COMPOSE) -f dockers/airflow/docker-compose.yaml up airflow-init

#other triplet suggestion
#airflow-build:  ## airflow build
#	$(DOCKER_COMPOSE)  -f dockers/airflow/docker-compose.yaml build

#airflow-init-1:  ## runing initialisation ⚠️ airflow-init(non evaluated) <> airflow_init(evaluated)
#	$(DOCKER_COMPOSE)  -f dockers/airflow/docker-compose.yaml run --rm airflow-webserver airflow db migrate

#airflow-up: ## allumage de airflow
#	$(DOCKER_COMPOSE)  -f dockers/airflow/docker-compose.yaml up -d


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

airflow_diag:  ## list process runnings with -a
	$(DOCKER_COMPOSE) -f dockers/airflow/docker-compose.yaml ps -a

airflow_log:  ## view logs optional with -f
	$(DOCKER_COMPOSE) -f dockers/airflow/docker-compose.yaml logs airflow-webserver

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
slmbuild0: ## Build l'image trainer (BACKEND=cpu|gpu)  #--no-cache --progress=plain 
	docker build \
	--build-arg BACKEND=$(BACKEND) \
	--build-arg DOCKFLD=$(DOCKFLD) \
	-f $(DOCKFLD)$(DOCK) -t $(TAG) .

slimbuild: ## Build l'image trainer (BACKEND=cpu|gpu)  #--no-cache --progress=plain 
	docker compose -f $(DOCKFLD)docker-compose.yaml --profile building build common 
	
slimbuild-cpu: ## Build CPU explicitement
	$(MAKE) slimbuild BACKEND=cpu DOCKFLD=dockers/slim/

slimbuild-gpu: ## Build GPU (CUDA 12.4) explicitement
	$(MAKE) slimbuild BACKEND=gpu DOCKFLD=dockers/slim/

#  pour appeler une target Make depuis une autre target, 
#  c'est toujours $(MAKE) <target> VAR=valeur
#  et TAG est une fonction f(IMAGE,BACKEND)
slimbuild-test: ## Build image de test CPU :: make slimbuild-test
	$(MAKE) slimbuild-cpu IMAGE=supply-test

slmtestconfig: ## Testing the docker compose config
	docker compose -f docker-compose1.yaml config |less #| grep -A 10 "args"
	# On voit les bon transferts d'arguments

#docker image for nginx proxy server & all
start-project:
	# was  docker-compose up --build api
	docker network create airflow_default 2>/dev/null || true
	$(DOCKER_COMPOSE) -p $(PROJECT) -f docker-compose1.yaml up -d --build 
	#--dry-run

proj-log:
	$(DOCKER_COMPOSE) -p $(PROJECT) -f docker-compose1.yaml logs
	
proj-stop: ## 👍 Stop current running containers for the project $(PROJECT)
	$(DOCKER_COMPOSE) -p $(PROJECT) -f docker-compose1.yaml down -v

proj-diag:
	$(DOCKER_COMPOSE) -p $(PROJECT) -f docker-compose1.yaml config


# Détecte la machine automatiquement - I car on travaillera en IP pas en dns pour la config nginx
CURRENT_IP := $(shell hostname -I | awk '{print $$1}')

proj-start:  ## 👍 starting grafana < prometheus < api < nginx + airflow if not commented
	@echo "Lancement API sur $(CURRENT_IP)"
	docker network create airflow_default 2>/dev/null || true
	$(DOCKER_COMPOSE) -p $(PROJECT) -f docker-compose1.yaml up -d --build

airf-start: ## airflow start separation to consider on localhost mono approad
	@echo "Lancement Airflow sur $(CURRENT_IP)"
	$(DOCKER_COMPOSE) -p $(PROJECT) -f docker-compose2.yaml up -d

airf-stop: ## 👍 Stop current running containers for the project $(PROJECT)
	$(DOCKER_COMPOSE) -p $(PROJECT) -f docker-compose2.yaml down -v

start-ip-conditional: ## main starter that separate for starting on different hosts
	@if [ "$(CURRENT_IP)" = "${API_HOST}" ]; then \
		make proj-start; \
	elif [ "$(CURRENT_IP)" = "${AIRFLOW_HOST}" ]; then \
		make airf-start; \
	fi

clean: ## 👍 Disk Space Recycling
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

uvPruning: ## one of the hardest hit, with --ci also ref https://docs.astral.sh/uv/concepts/cache/#clearing-the-cache
	@ uv cache prune --ci

nginxConfReload: ## reload a chaud pour tester un changement de config nginx
	docker exec nginx_revproxy nginx -s reload

nginxLogs: ## view the logs
	docker compose -p $(PROJECT) -f docker-compose1.yaml logs -f

ngxtest: ## test syntax of nginx cong before reloading
	@ docker exec nginx_revproxy nginx -t  && echo " "	

ngxreload: ## hot reload of nginx config
	@ $(MAKE) ngxtest && docker exec nginx_revproxy nginx -s reload && echo " "	

updateFreeze: ## Update the slim requirement-freeze.txt, make it used in the resolution if it exists
	# once docker is up we can capture the pip resolution and reapply
	docker exec $(PROJECT)-api-1 pip freeze > dockers/slim/requirements-freeze.txt

graf_ps: ## Statut du container Grafana
	docker compose ps grafana

graf_inspect: ## Inspection détaillée de l'état Grafana
	docker inspect grafana_dashboard | grep -A10 "State"

graf_logs: ## Logs en direct de Grafana
	docker compose logs -f grafana