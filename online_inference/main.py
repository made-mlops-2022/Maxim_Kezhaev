import os
import pickle
from typing import Literal

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = None


class ClevelandData(BaseModel):
    age: float
    sex: Literal[0, 1]
    cp: Literal[0, 1, 2, 3]
    trestbps: float
    chol: float
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: float
    exang: Literal[0, 1]
    oldpeak: float
    slope: Literal[0, 1, 2]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[0, 1, 2]


@app.on_event('startup')
def load_model():
    model_path = "../models/GaussianNB.pkl"
    # choose this because it has the best metrics

    with open(model_path, 'rb') as f:
        global model
        model = pickle.load(f)


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": "GaussianNB"}


@app.post("/predict")
def predict(data: ClevelandData):
    data = data.dict()
    df = pd.DataFrame([data])
    pred = model.predict(df)
    print(pred)
    if pred[0] == 1:
        answer = "disease"
    else:
        answer = "no disease"

    return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
