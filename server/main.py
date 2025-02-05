from flask import Flask, request, jsonify
import cv2
import numpy as np
import math
import os
import base64
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from flask_cors import CORS
import mlflow
import mlflow.tensorflow

app = Flask(__name__)
CORS(app)

detector = HandDetector(maxHands=1, detectionCon=0.8, minTrackCon=0.5)
classifier = Classifier("keras_model.h5", "labels.txt")

imgSize = 500
offset = 20
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O"]
string = ""

# MLflow setup
mlflow.set_experiment("Hand_Gesture_Classification")
import traceback
import tempfile
# Helper function to append to string only if not repeated
def append_to_string(alpha, beta):
    if alpha.find(beta, len(alpha) - 2) == -1:
        return alpha + beta
    else:
        return alpha

@app.route('/predict', methods=['POST'])
def predict():
    global string
    img = None
    try:  
        # Check if the request contains base64 image data
        if 'image' in request.json:
            try:
                image_data = request.json['image']
                # Remove the data:image/jpeg;base64, prefix if present
                if 'base64,' in image_data:
                    image_data = image_data.split('base64,')[1]
                
                # Decode base64 to bytes
                img_bytes = base64.b64decode(image_data)
                
                # Convert bytes to numpy array
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                return jsonify({"error": f"Error decoding base64 image: {str(e)}"})
        
        # Handle file upload (existing code)
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No selected file"})
            
            # Read image as numpy array
            nparr = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        else:
            return jsonify({"error": "No image data provided"})
        
        if img is None:
            return jsonify({"error": "Invalid image format"})

        # Hand detection
        hands, img = detector.findHands(img)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[max(0, y - offset): y + h + offset, max(0, x - offset): x + w + offset]
            
            if imgCrop.size == 0:
                return jsonify({"error": "Invalid cropped image"})

            # Calculate aspect ratio and resize image
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            detected_letter = labels[index]
            string = append_to_string(string, detected_letter)

            # Log metrics to MLflow
            with mlflow.start_run():
                mlflow.log_param("model", "keras_model.h5")
                mlflow.log_param("input_image_size", imgSize)
                mlflow.log_metric("prediction_index", index)
                mlflow.log_metric("prediction_accuracy", max(prediction))
                # mlflow.log_metric("Detected Leter",detected_letter)  # Assuming prediction[1] is the accuracy/confidence

            # Save to file
            with open("file.txt", "w") as text_file:
                text_file.write(string)
        
            return jsonify({"predicted_letter": detected_letter, "full_string": string})
        
        else:
                raise Exception("No hands detected")

    except Exception as e:
        # Capture the traceback and log it
        error_message = str(e)
        error_traceback = traceback.format_exc()  # Capture full traceback

        # Write the traceback to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
            temp_file.write(error_traceback)
            temp_file_path = temp_file.name

        # Log the error message and the traceback as an artifact
        with mlflow.start_run():
            mlflow.log_param("error_message", error_message)  # Log the error message
            mlflow.log_artifact(temp_file_path, "error_traceback.txt")  # Save the error traceback file as an artifact
            print(f"Error occurred: {error_message}\n{error_traceback}")  # Print error for normal logging

        # Clean up the temporary file after logging
        os.remove(temp_file_path)

        return jsonify({"error": error_message, "traceback": error_traceback})

@app.route('/reset', methods=['POST'])
def reset_string():
    global string
    string = ""
    with open("file.txt", "w") as text_file:
        text_file.write(string)
    return jsonify({"message": "String reset successfully"})

@app.route("/", methods=['GET'])
def health_check():
    return jsonify({"message": "API is running"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
