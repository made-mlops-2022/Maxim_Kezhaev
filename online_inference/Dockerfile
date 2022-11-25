FROM python:3.8-slim

WORKDIR /home/hw2
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache -r requirements.txt

RUN apt update && apt install -y wget
RUN wget -O "GaussianNB.pkl" "https://drive.google.com/uc?export=download&id=1PIUpjMheDYLKR4vQ2ITwR0xHg5N3e_yw"
ENV MODEL_NAME="GaussianNB.pkl"

COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]