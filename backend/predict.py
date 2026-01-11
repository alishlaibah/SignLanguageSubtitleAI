from tensorflow.keras.models import load_model
import joblib
import numpy as np
import os
import cv2


model_path = os.path.join("backend", "saved_models", "asl_model.h5")
scaler_path = os.path.join("backend", "saved_models", "scaler.pkl")

model = load_model(model_path)
scaler = joblib.load(scaler_path)


def predict(landmarks):
    if len(landmarks) != 63:
        return "Invalid input: expected 63 features."
    
    np_landmarks = np.array(landmarks).reshape(1, -1)
    landmarks_scaled = scaler.transform(np_landmarks)
    prediction = model.predict(landmarks_scaled)[0][0]

    if prediction > 0.5:
        prediction_text = "A"
    else: 
        prediction_text = "Not A"

    confidence = float(prediction)

    return {"letter": prediction_text, "confidence": confidence}