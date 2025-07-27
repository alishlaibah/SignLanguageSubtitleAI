from tensorflow.keras.models import load_model
import joblib
import numpy as np
import os
import cv2


model_path = os.path.join("backend", "models", "asl_model.h5")
scaler_path = os.path.join("backend", "models", "scaler.save")

model = load_model(model_path)
scaler = joblib.load(scaler_path)


def predict(landmarks):
    if len(landmarks) == 63:
        landmarks = np.array(landmarks).reshape(1, -1)
        landmarks_scaled = scaler.transform(landmarks)
        prediction = model.predict(landmarks_scaled)[0][0]
        if prediction > 0.5:
            prediction_text = "A"
        else: 
            prediction_text = "Not A"

            return prediction_text