from datetime import timedelta

default_args = {
    "owner": "makezh",
    "email_on_failure": True,
    "email": ["max.kezhaev@gmail.com"],
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

HOST_DIR_PATH = "/Users/max/Downloads/programming/techpark/MLOps/Maxim_Kezhaev/airflow_ml_dags/data"
