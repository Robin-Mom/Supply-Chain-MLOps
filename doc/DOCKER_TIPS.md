# Docker after PR STARTUP

TL;TR
docker-compose build --no-cache api
docker-compose down
docker-compose up api


# le vrai clean pour repartir de zéro

Ici un peu d'inventaire et de listing pour controler et mesurer, puis des commandes d'actions

```bash
docker system list
docker system prune -af --volumes - récup de 9G
```

un peu du journal
```bash
sudo journalctl --disk-usage
sudo journalctl --vacuum-size=100M
```

curieusement .venv était root tout d'un coup ???

```bash
sudo rm -rf .venv/
rm uv.lock
```

à checker
```bash 
uv sync --extra cpu
```

ok .venv à 2GB - stable


```bash 
(supply-chain-mlops) ubuntu@ip-172-31-44-53:~/Supply-Chain-MLOps$ du -sk .venv/lib/python3.12/site-packages/torch/lib/*|sort -n
…
30324   .venv/lib/python3.12/site-packages/torch/lib/libtorch_python.so
431828  .venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so
```

pas de gpu

au cas ou pip serait réemployé
```bash 
uv pip freeze > requirements.txt 
```

DIAGNOSTIQUE et facture
	- le disque est rempli à 95% après le build
	- le cache du uv
	- du -sh /home/ubuntu/.cache/uv/ 2.9GB  -> uv cache clean


```bash 
	docker image list
	REPOSITORY               TAG       IMAGE ID       CREATED          SIZE
	supply-chain-mlops_api   latest    78e137a52baf   22 minutes ago   9.7GB
```
```bash 
	docker system prune -af --volumes
```

récup de 9G
	
ne pas oublier 
```bash 
uv sync --extra cpu
```

RESET SITUATION
following prune has no effect, because container list will show one record
```bash 
docker container list -a

CONTAINER ID   IMAGE                    COMMAND                  CREATED       STATUS                   PORTS     NAMES
8fc03cebeef5   supply-chain-mlops_api   "uv run uvicorn src.…"   5 hours ago   Exited (2) 5 hours ago             supply-chain mlops_api_1

docker image prune -aq 

required
docker image rm -f supply-chain-mlops_api
docker image prune -aq

docker system prune -af --volumes
```

#0 les diagnostiques de place
```bash 
docker system df
```

```bash 
#0 ⚠️ One Liner pour un makefile (attention si rmi plante xargs -r termine positive quand meme)
df -Ph . && docker system df && docker images -aq | xargs -r docker rmi supply-chain-mlops_api -f && docker builder prune -af && docker system prune -af && docker system df  && df -Ph .
```  

```bash 
	# 1. Arrêtez tous les conteneurs Docker en cours et Supprimez tous les conteneurs (y compris ceux arrêtés)
	docker stop $(docker ps -aq)
	docker rm $(docker ps -aq)
	flt=supply-chain-mlops_api;docker rm $(docker container ls -a|grep $flt|cut -d" " -f1)
	
	# 2. Supprimer les images Docker
	docker rmi $(docker images -aq) -f
	
	# 3. Nettoyer Docker (cache, volumes, réseaux)
	docker system prune -af --volumes
	
	# 4. Nettoyez le cache BuildKit (libère ~ 10.3 GB)
	docker builder prune -af
	
	# 5. Nettoyez les images Docker inutilisées (libère ~9.76 GB)
	docker system prune -af

	# 6. Supprimez le cache uv local (libère ~2.9 Go)
	du -sh ~/.cache/uv
	rm -rf ~/.cache/uv

	# 7. Supprimez le cache uv partagé (si vous avez utilisé UV_CACHE_DIR -> non pas encore)
	rm -rf ~/uv_cache
	
	# 8. Supprimez les fichiers temporaires Docker (logs, etc.) -> pas besoin encore
	du -sh /var/lib/docker/*
	sudo rm -rf /var/lib/docker/tmp/*
	
	#9. si besoin de redémarra de docker
	sudo systemctl restart docker
```


état de départ:
AVAIL 11GB

## Sanity & validation docker disk  usage
```bash 
#script généré par chat.mistral.ai
./ddu.sh
```

```bash 
#collection et comparaison
(cd ddu; ../ddu.sh before_docker)

docker-compose build --no-cache api
docker-compose build --no-cache --build-arg BACKEND_EXTRA=gpu api

docker-compose up api
```

Si pas d'itération sur le ou les builds en cours - USAGE ddu.sh

```bash 
# clean
docker builder prune
# measure before
(cd ddu; ../ddu.sh before_docker)
# run or build
# ...
# measure before
(cd ddu; ../ddu.sh after_docker)
# compare
(cd ddu; ../ddu.sh compare before_uv after_uv)

```

few key handy command

Utilisez uv cache clean pour supprimer les packages non utilisés dans le cache: 

```bash 

uv cache clean

vérif de la position de uv
docker build --target builder -t debug-builder . 
docker run --rm debug-builder ls -la /bin/uv /bin/uvx /usr/local/bin/ 2>&1
docker inspect supply-chain-mlops_api | grep -A 5 '"Cmd"'
docker run --rm -it supply-chain-mlops_api /bin/bash
docker container ls
docker logs 1c4ae80ea8d1

# UNITEST docker
docker run --rm -p 8000:8000 supply-chain-mlops_api .venv/bin/python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```