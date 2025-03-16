from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load Deepfake Detection Model (Using TensorFlow & PyTorch models)
tf_model = tf.keras.models.load_model("models/image_forgery_cnn.h5")
torch_model = torch.jit.load("models/deepfake_detector.pt")
torch_model.eval()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/detect_forgery", methods=["POST"])
def detect_forgery():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    
    # Process Image using CNN
    image = preprocess_image(file_path)
    prediction = tf_model.predict(image)
    forgery_probability = float(prediction[0][0])
    
    result = "Forgery Detected" if forgery_probability > 0.5 else "Authentic Image"
    
    return jsonify({
        "file": file.filename,
        "prediction": result,
        "confidence": forgery_probability
    })

@app.route("/detect_deepfake", methods=["POST"])
def detect_deepfake():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    
    # Process Image using PyTorch Model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(file_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = torch_model(image)
        deepfake_probability = torch.nn.functional.softmax(output, dim=1)[0][1].item()
    
    result = "Deepfake Detected" if deepfake_probability > 0.5 else "Authentic Image"
    
    return jsonify({
        "file": file.filename,
        "prediction": result,
        "confidence": deepfake_probability
    })

if __name__ == "__main__":
    app.run(debug=True)
