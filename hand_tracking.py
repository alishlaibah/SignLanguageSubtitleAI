# the base of the project is hand tracking

import os
import logging
import cv2
import mediapipe as mp

# remove TensorFlow and MediaPipe logs 
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
    success, frame = webcam.read() # success tells us if the webcam is working, frame is webcams image

    # Applying hand tracking model
    # Convert fram from BGR (OpenCV default) to RGB which is required for MediaPipe
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    result = hands.process(frame)

    # Draw on frame
    # Convert it back to BGR for OpenCV visualization
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Ff hands detected, draw landmarks for each hand
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand, connections=mp_hands.HAND_CONNECTIONS)
    
    # Display the frame
    cv2.imshow("Cam", frame)
    
    key = cv2.waitKey(1)
    if key == ord("q"): 
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()