#dag pipeline for extract transform load train evaluate
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor
from airflow.operators.docker_operator import DockerOperator
from docker.types import Mount
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import datetime

# definition of the function to execute
#si besoin
def ltte():
    print("Load Train Transform Evaluate repro")


def mflowPromote():
    # automated promotion
    # analyse et choix auto de la promotion
    # la registration permet de ne comparer que les métrics du run en cours et les métrics du dernier promu
    # attention au drift qui consiste à ce que l'ancien modèle qui marchait bien au début
    # devient moins bon sur les nouvelles tendances de données.
    run_id=1
    print(f"Promote Runid:{run_id}")

with DAG(
    dag_id='e_t_l_t_e',
    description='Ingest Extract push et promote model',
    tags=['Supply','Liora'],
    schedule_interval=None,
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0),
    }
) as dag:
    
    #https://airflow.apache.org/docs/apache-airflow-providers-standard/stable/operators/index.html
    #https://airflow.apache.org/docs/apache-airflow-providers-standard/stable/operators/bash.html
    ingest = BashOperator(
        task_id="ingest",
        bash_command="python ingest.py && dvc add data/raw && dvc push",
    )
    ltte = BashOperator(
    task_id="repro",
    bash_command="cd ./ltteloc && dvc repro",
    )

    if False:
        #autre appel possible
        #vérifier aussi ce que
        from airflow.providers.docker.operators.docker import DockerOperator

        #dans le train, trace output des metrics ou score
        # dans ton script Python :
        import json
        theSilScore=0.72
        print(json.dumps({"silhouette_score": theSilScore}))  # ← stdout capturé par Airflow

        train_task = DockerOperator(
            task_id="train_model",
            image="mlopsv-app",          # ← la même image ✅
            command="dvc repro --force",
            volumes=["/app:/app"],
            auto_remove=True,            # ← conteneur supprimé après ✅
            do_xcom_push=True,          # ← capture stdout dans XCom ✅
            dag=dag,
        )

        # 3. Tâche suivante lit le score
        def log_score(**context):
            output = context["ti"].xcom_pull(task_ids="train_model")
            score = json.loads(output)["silhouette_score"]
            print(f"Silhouette score : {score}")

        log_task = PythonOperator(
            task_id="log_score",
            python_callable=log_score,
            dag=dag,
        )        
        from airflow.operators.bash import BashOperator

        train_task = BashOperator(
            task_id="train_model",
            bash_command="docker compose -p mlopsv run --rm trainer",
            dag=dag,
        )


    push = BashOperator(
    task_id="push",
    bash_command="dvc push", #on est à la racine du projet ou il y a le dossier .dvc
    )

    #https://airflow.apache.org/docs/apache-airflow-providers-standard/stable/operators/python.html
    promotion = PythonOperator(
        task_id="promotion", 
        python_callable=mflowPromote
    )


    ingest >> ltte >> push >> promotion