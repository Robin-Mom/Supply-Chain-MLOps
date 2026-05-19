#definition du docker_compose disponible
DOCKER_COMPOSE := $(shell which docker-compose 2>/dev/null || echo "docker compose")

# definition de AIRFLOW_UID ou bien dans .env (mais avec restriction de config vscode)
export AIRFLOW_UID := $(shell id -u)
export AIRFLOW_GID := 0

# AIRFLOW INIT SECTION TO ALWAYS RUN FIRST TIME DEPLOYED ON A NEW DEVICE
airflow_init:
	$(DOCKER_COMPOSE) -f airflow/docker-compose.yaml up airflow-init

airflow_up:
	$(DOCKER_COMPOSE) -f airflow/docker-compose.yaml up -d

# AIRFLOW SECTION
airflow_force_recreate:
	$(DOCKER_COMPOSE) -f airflow/docker-compose.yaml up --force-recreate

airflow_down:
	$(DOCKER_COMPOSE) -f airflow/docker-compose.yaml down 

airflow_reset:
	echo "docker-compose down -v"

airflow_ps:
	$(DOCKER_COMPOSE) -f airflow/docker-compose.yaml ps

airflow_prepdocker_sock:
	sudo chmod a+rw /var/run/docker.sock

# GLOBAL SECTION
up:
	$(DOCKER_COMPOSE) -f airflow/docker-compose.yaml up -d
	$(DOCKER_COMPOSE) -f ./docker-compose.yaml up -d

down:
	$(DOCKER_COMPOSE) -f ./docker-compose.yaml down
	$(DOCKER_COMPOSE) -f airflow/docker-compose.yaml down