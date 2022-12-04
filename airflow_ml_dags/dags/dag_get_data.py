from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

from args import default_args, HOST_DIR_PATH


dag = DAG(
    dag_id="dag_get_data",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(1)
)

with dag:
    download = DockerOperator(
        image="airflow-download",
        command="/data/raw/{{ ds }}",
        task_id="downloading-data",
        network_mode="bridge",
        mount_tmp_dir=False,
        do_xcom_push=False,
        mounts=[Mount(source=HOST_DIR_PATH,
                      target="/data",
                      type='bind')]
    )

    download
