from backend.predict import predict
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, request

app = FastAPI()

class LandmarksRequest(BaseModel):
    landmarks: List[float]


@app.post("/predict")
def get_prediction(request: LandmarksRequest):
    return predict(request.landmarks)