import os
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'static', 'assets', 'css'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'static', 'assets', 'js'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'static', 'assets', 'img'), exist_ok=True)

# Load the Model
MODEL_PATH = os.path.join(BASE_DIR, 'healthy_vs_rotten.h5')
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"WARNING: Model file {MODEL_PATH} not found. Running in dummy mode.")
except Exception as e:
    print(f"Error loading model: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    if model is None:
        # Dummy prediction for UI testing when model is absent
        import random
        import time
        time.sleep(1) # Simulate processing delay
        classes = ['Healthy', 'Rotten']
        pred = random.choice(classes)
        confidence = random.uniform(0.75, 0.99)
        return pred, confidence

    # Preprocess the image for MobileNetV2
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        predictions = model.predict(img_array)
        
        # Determine class based on model output shape
        if predictions.shape[-1] == 1:
            # Binary classification (sigmoid)
            prob = predictions[0][0]
            if prob > 0.5:
                # Assuming 1 is Rotten, 0 is Healthy. Wait, usually 1 is the positive class.
                # Let's say based on typical alphabetical sorting: 'healthy' (0), 'rotten' (1)
                return "Rotten", float(prob)
            else:
                return "Healthy", float(1 - prob)
        else:
            # Categorical classification
            class_idx = np.argmax(predictions[0])
            prob = predictions[0][class_idx]
            # Will need class labels mappings. Assuming simple for now
            # In a real scenario, we save a class_indices mapping.
            return f"Class {class_idx}", float(prob)
            
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/blog.html', methods=['GET'])
def blog():
    return render_template('blog.html')

@app.route('/blog-single.html', methods=['GET'])
def blog_single():
    return render_template('blog-single.html')

@app.route('/portfolio-details.html', methods=['GET'])
def portfolio_details():
    return render_template('portfolio-details.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        # Generate unique filename
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        
        # Ensure we have the web-accessible path
        image_url = url_for('static', filename=f'uploads/{filename}')
        
        # Get prediction
        label, confidence = predict_image(filepath)
        
        return jsonify({
            'success': True,
            'label': label,
            'confidence': f"{confidence * 100:.2f}%",
            'image_url': image_url
        })
        
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
