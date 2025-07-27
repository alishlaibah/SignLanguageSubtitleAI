import os
import logging
import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("models/mlp_model.h5")
scaler = joblib.load("models/scaler.pkl")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Initialize MediaPipe Hand tracking and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create the hand tracking model with high confidence for better accuracy
hands = mp_hands.Hands (
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.9,
    min_tracking_confidence = 0.9
)

# Capture video from the webcam (change 0 or 1 depending on the device)
webcam = cv2.VideoCapture(1)

# Optimize webcam settings to reduce jitter
webcam.set(cv2.CAP_PROP_FPS, 30)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



while True:
    success, frame = webcam.read() # Success tells us if the webcam is working, frame is webcams image

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    prediction_text = "No Hand Detected"

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand, connections=mp_hands.HAND_CONNECTIONS)
        
        my_array = []
        for lm in hand.landmark:
            my_array.extend([lm.x, lm.y, lm.z])

        # Feeding the array to the model
        if len(my_array) == 63:
            my_array = np.array(my_array).reshape(1, -1)
            my_array_scaled = scaler.transform(my_array)
            prediction = model.predict(my_array_scaled)[0][0]

            if prediction > 0.5:
                prediction_text = "A"
            else: 
                prediction_text = "Not A"

    cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Cam", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

webcam.release()

cv2.destroyAllWindows()