from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from docker.types import Mount

from args import default_args, HOST_DIR_PATH

dag = DAG(
    dag_id="dag_train_data",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(1)
)

with dag:
    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="preprocessing-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=HOST_DIR_PATH,
                      target="/data",
                      type='bind')]
    )

    wait_data_sensor = FileSensor(
        task_id="wait_processed_data",
        poke_interval=10,
        retries=100,
        filepath="data/raw/{{ ds }}/data.csv"
    )

    wait_target_sensor = FileSensor(
        task_id="wait_processed_target",
        poke_interval=10,
        retries=100,
        filepath="data/raw/{{ ds }}/target.csv"
    )

    split = DockerOperator(
        image="airflow-split",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="splitting-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=HOST_DIR_PATH,
                      target="/data",
                      type='bind')]
    )

    train = DockerOperator(
        image="airflow-train",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/model/{{ ds }}",
        task_id="training-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=HOST_DIR_PATH,
                      target="/data",
                      type='bind')]
    )

    validate = DockerOperator(
        image="airflow-validate",
        command="--input-dir /data/processed/{{ ds }} --model-dir /data/model/{{ ds }} --output-dir /data/metrics/{{ ds }}",
        task_id="validating-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=HOST_DIR_PATH, target="/data", type='bind')]
    )

    preprocess >> [wait_data_sensor, wait_target_sensor] >>split >> train >> validate

