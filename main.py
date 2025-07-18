import uvicorn
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import pickle

class PredictRequest(BaseModel):
    features: List[float]

with open("models/iris_rf_optuna_scaled.pkl", "rb") as f:
    _model = pickle.load(f)
    print("✅ Modelo cargado en memoria")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Página de inicio"}


# 3) Endpoint de inferencia
@app.post("/predict")
def predict(item: PredictRequest):
    # Aquí ‘model’ ya está en memoria
    features = [item.features]
    pred = _model.predict(features)
    return {"prediction": int(pred[0])}
