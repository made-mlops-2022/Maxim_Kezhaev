from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from docker.types import Mount
from airflow.models import Variable

from args import default_args, HOST_DIR_PATH

model_path = Variable.get("model_path")

dag = DAG(
    dag_id="dag_predict_data",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(1)
)

with dag:
    wait_data_sensor = FileSensor(
        task_id="wait_pred_data",
        poke_interval=10,
        retries=100,
        filepath="data/raw/{{ ds }}/data.csv"
    )

    predict = DockerOperator(
        image="airflow-predict",
        command="--input-dir /data/raw/{{ ds }} --model-path {{ var.value.model_path }} --output-dir /data/predictions/{{ ds }}",
        environment={"MODEL_PATH": "{{ var.value.model_path }}"},
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=HOST_DIR_PATH, target="/data", type='bind')]
    )

    wait_data_sensor >> predict
