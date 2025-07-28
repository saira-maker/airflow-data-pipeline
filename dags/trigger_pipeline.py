from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1
}

def run_script(script_name):
    subprocess.run(['python', f'/usr/local/airflow/dags/{script_name}'], check=True)

with DAG(
    dag_id='run_etl_pipeline_steps',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['etl', 'brfss', 'supabase']
) as dag:

    run_raw = PythonOperator(
        task_id='run_raw_data',
        python_callable=lambda: run_script('raw_data.py')
    )

    run_staging = PythonOperator(
        task_id='run_staging_data',
        python_callable=lambda: run_script('staging.py')
    )

    run_presentation = PythonOperator(
        task_id='run_presentation_data',
        python_callable=lambda: run_script('presentation.py')
    )

    run_raw >> run_staging >> run_presentation
