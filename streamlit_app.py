import os
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import pandas as pd
import streamlit as st
import time

# Load the model
model = load_model('Model/model.h5')

# Mediapipe setup
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Letters (J and Z not included due to gesture motion)
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Streamlit app title
st.title("American Sign Language Detection")
frame_placeholder = st.empty()

# Add a placeholder for the predicted letter and confidence
predicted_text_placeholder = st.empty()

# Streamlit session state to control the video feed and accumulated sentence
if "running" not in st.session_state:
    st.session_state.running = False
if "detected_sentence" not in st.session_state:
    st.session_state.detected_sentence = ""  # Store the detected sentence
if "last_predicted" not in st.session_state:
    st.session_state.last_predicted = None  # Store last detected letter
if "last_prediction_time" not in st.session_state:
    st.session_state.last_prediction_time = time.time()  # Time of last prediction

# Start and Stop buttons
if st.button("Start"):
    st.session_state.running = True
if st.button("Stop"):
    st.session_state.running = False
if st.button("Reset Sentence"):
    st.session_state.detected_sentence = ""  # Reset the accumulated sentence

# Video capture logic
cap = cv2.VideoCapture(0)

while st.session_state.running:
    ret, frame = cap.read()
    if not ret:
        st.error("Cannot access the camera. Please check your camera connection.")
        break

    h, w, _ = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    # Process detected hands
    if result.multi_hand_landmarks:
        for handLMs in result.multi_hand_landmarks:
            x_min, x_max = w, 0
            y_min, y_max = h, 0

            # Calculate bounding box
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, x_max = min(x, x_min), max(x, x_max)
                y_min, y_max = min(y, y_min), max(y, y_max)

            x_min, x_max = max(0, x_min - 20), min(w, x_max + 20)
            y_min, y_max = max(0, y_min - 20), min(h, y_max + 20)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)

            # Prediction
            try:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hand_crop = gray_frame[y_min:y_max, x_min:x_max]
                hand_resized = cv2.resize(hand_crop, (28, 28))
                pixel_data = hand_resized.flatten() / 255.0
                pixel_data = pixel_data.reshape(-1, 28, 28, 1)

                prediction = model.predict(pixel_data)
                pred_array = prediction[0]
                predicted_letter = letterpred[np.argmax(pred_array)]
                confidence = np.max(pred_array)

                # Logic to accumulate letters with a delay between predictions
                current_time = time.time()

                # Only accumulate if enough time has passed and the letter is not the same as the previous one
                if confidence > 0.8 and predicted_letter != st.session_state.last_predicted and (current_time - st.session_state.last_prediction_time) > 1.0:
                    st.session_state.detected_sentence += predicted_letter  # Add letter to sentence
                    st.session_state.last_predicted = predicted_letter  # Update last predicted letter
                    st.session_state.last_prediction_time = current_time  # Update the last prediction time

                # Display predicted letter on the frame
                text = f"{predicted_letter} ({confidence:.2f})"
                cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Update the placeholder with the accumulated sentence
                predicted_text_placeholder.text(f"Detected Sentence: {st.session_state.detected_sentence}")

            except Exception as e:
                print(f"Prediction error: {e}")

    frame_placeholder.image(frame, channels="BGR")

    # Optionally add a delay to prevent the sentence from growing too fast
    time.sleep(0.1)

cap.release()
hands.close()
cv2.destroyAllWindows()
