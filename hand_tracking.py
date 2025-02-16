# the base of the project is hand tracking

import cv2
import mediapipe as mp

# Initialize MediaPipe Hand tracking and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Capture video from the webcam (change 0 or 1 depending on the device)
webcam = cv2.VideoCapture(1)

while True:
    success, frame = webcam.read() # success tells us if the webcam is working, frame is webcams image

    # Applying hand tracking model
    # Convert fram from BGR (OpenCV default) to RGB which is required for MediaPipe
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    handsTracking = mp_hands.Hands().process(frame)

    # Draw on frame
    # Convert it back to BGR for OpenCV visualization
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Ff hands detected, draw landmarks for each hand
    if handsTracking.multi_hand_landmarks:
        for hand in handsTracking.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand, connections=mp_hands.HAND_CONNECTIONS)
    
    # Display the fram
    if success == True:
        cv2.imshow("Cam", frame)
        key = cv2.waitKey(1)

        # Exit if the user presses "q"
        if key == ord("q"): 
            break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()