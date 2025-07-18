from fastapi import FastAPI
from typing import List
from pydantic import BaseModel

class PredictRequest(BaseModel):
    features: List[float]  # aquí esperas una lista de floats


app = FastAPI()

@app.get("/")
def root():
    return {"message": "Página de inicio"}

@app.get("/")
def root():
    return {"message": "Página de inicio"}