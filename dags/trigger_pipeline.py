from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess

# Define the path to your master_pipeline.py
SCRIPT_PATH = "C:/Users/HAADI/Documents/clean_project/master_pipeline.py"

def run_pipeline():
    subprocess.run(["python", SCRIPT_PATH], check=True)

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1
}

with DAG(
    dag_id='run_master_pipeline',
    default_args=default_args,
    schedule_interval=None,  # Trigger manually or later change to daily/hourly
    catchup=False,
    tags=['etl', 'local', 'brfss']
) as dag:

    run_etl = PythonOperator(
        task_id='run_master_pipeline_task',
        python_callable=run_pipeline
    )
