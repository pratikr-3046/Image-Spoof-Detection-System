# app.py
import os
import numpy as np
import cv2
from skimage.restoration import denoise_tv_chambolle
from skimage.feature import local_binary_pattern
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the models with error handling
try:
    logger.info("Loading MobileNet model...")
    mobilenet_model = load_model('mobilenet_model.h5')
    logger.info("MobileNet model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading MobileNet model: {str(e)}")
    mobilenet_model = None

try:
    logger.info("Loading EfficientNet model...")
    efficientnet_model = load_model('efficientnet_model.h5')
    logger.info("EfficientNet model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading EfficientNet model: {str(e)}")
    efficientnet_model = None

try:
    logger.info("Loading SVM model...")
    svm_model = joblib.load('svm_model.joblib')
    logger.info("SVM model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading SVM model: {str(e)}")
    svm_model = None

# Check if at least one model was loaded
if mobilenet_model is None and efficientnet_model is None and svm_model is None:
    logger.error("No models could be loaded. Application will not function correctly.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0,1]
    return img_array

def preprocess_image_for_svm(img_path, img_size=(128, 128)):
    # Read and convert to grayscale
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Denoising
    ltv_denoised = denoise_tv_chambolle(gray, weight=0.1)
    ltv_denoised = (ltv_denoised * 255).astype(np.uint8)
    
    # Resizing
    resized = cv2.resize(ltv_denoised, img_size)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(resized)
    
    # Return as uint8 to avoid LBP warning
    return enhanced  # No normalization

def extract_lbp_features(image_array):
    # Ensure image is uint8 to avoid LBP warning
    if image_array.dtype.kind == 'f':
        image_array = (image_array * 255).astype(np.uint8)
    
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image_array, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(),
                          bins=np.arange(0, n_points + 3),
                          density=True)
    return hist.reshape(1, -1)  # Reshape for model input

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Initialize predictions
            mobilenet_pred = 0.5  # Default value if model not available
            efficientnet_pred = 0.5
            svm_pred = 0.5
            model_count = 0
            
            # Get MobileNet prediction if model is available
            if mobilenet_model is not None:
                processed_image = preprocess_image(filepath)
                mobilenet_pred = mobilenet_model.predict(processed_image)[0][0]
                model_count += 1
                logger.info(f"MobileNet prediction: {mobilenet_pred}")
            
            # Get EfficientNet prediction if model is available
            if efficientnet_model is not None:
                if 'processed_image' not in locals():
                    processed_image = preprocess_image(filepath)
                efficientnet_pred = efficientnet_model.predict(processed_image)[0][0]
                model_count += 1
                logger.info(f"EfficientNet prediction: {efficientnet_pred}")
            
            # Get SVM prediction if model is available
            if svm_model is not None:
                svm_preprocessed = preprocess_image_for_svm(filepath)
                lbp_features = extract_lbp_features(svm_preprocessed)
                
                # Handle SVM prediction based on model type
                if hasattr(svm_model, 'predict_proba'):
                    svm_pred = svm_model.predict_proba(lbp_features)[0][1]
                else:
                    svm_pred_raw = svm_model.predict(lbp_features)[0]
                    svm_pred = float(svm_pred_raw)
                model_count += 1
                logger.info(f"SVM prediction: {svm_pred}")
            
            # Calculate combined score - average of available models
            if model_count > 0:
                combined_score = (mobilenet_pred + efficientnet_pred + svm_pred) / model_count
            else:
                combined_score = 0.5  # Default if no models available
            
            # Threshold for final decision
            is_real = combined_score > 0.5
            
            result = {
                'image_path': '/static/uploads/' + filename,
                'mobilenet_confidence': float(mobilenet_pred),
                'efficientnet_confidence': float(efficientnet_pred),
                'svm_confidence': float(svm_pred),
                'combined_confidence': float(combined_score),
                'is_real': bool(is_real),
                'prediction': 'Real Image' if is_real else 'Fake Image',
                'models_used': model_count
            }
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    # Check if models were loaded
    models_loaded = []
    if mobilenet_model is not None:
        models_loaded.append("MobileNet")
    if efficientnet_model is not None:
        models_loaded.append("EfficientNet")
    if svm_model is not None:
        models_loaded.append("SVM")
    
    logger.info(f"Models loaded: {', '.join(models_loaded) if models_loaded else 'None'}")
    app.run(debug=True)