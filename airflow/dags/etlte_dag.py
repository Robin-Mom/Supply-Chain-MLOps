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
    dag_id='my_first_dag',
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