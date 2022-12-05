## Environment variables 
~~~
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
export MODEL_PATH="<path_to_model>/model.pkl"
export MAIL_USER="<your email>"
export MAIL_PASS="<16-digit app password>"
~~~

## Run
~~~
docker compose up --build
~~~
