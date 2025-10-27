"""
Flask Backend API for Pneumonia Detection System
Provides REST API endpoints for model predictions and Grad-CAM visualization
"""
import os
import io
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from config import *
from data_preprocessing import preprocess_single_image
from gradcam import GradCAM, create_gradcam_visualization
from model import PneumoniaDetectionModel

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['UPLOAD_FOLDER'] = UPLOADS_DIR

# Global model variable
model = None


def allowed_file(filename):
    """
    Check if file extension is allowed
    
    Args:
        filename: Name of the file
        
    Returns:
        Boolean indicating if file is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """
    Load the trained model
    """
    global model
    
    if model is None:
        try:
            if os.path.exists(BEST_MODEL_PATH):
                model = keras.models.load_model(BEST_MODEL_PATH)
                print(f"Model loaded from {BEST_MODEL_PATH}")
            elif os.path.exists(MODEL_PATH):
                model = keras.models.load_model(MODEL_PATH)
                print(f"Model loaded from {MODEL_PATH}")
            else:
                print("Warning: No trained model found. Please train the model first.")
                return None
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    
    return model


def image_to_base64(image):
    """
    Convert image array to base64 string
    
    Args:
        image: Image array
        
    Returns:
        Base64 encoded string
    """
    # Convert to PIL Image
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    pil_image = Image.fromarray(image)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


@app.route('/')
def home():
    """
    Home endpoint
    """
    return jsonify({
        'message': 'Pneumonia Detection API',
        'version': '1.0.0',
        'endpoints': {
            '/api/health': 'Health check',
            '/api/predict': 'Make prediction (POST)',
            '/api/predict-with-gradcam': 'Prediction with Grad-CAM (POST)',
            '/api/model-info': 'Get model information'
        }
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    model_status = 'loaded' if model is not None else 'not loaded'
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """
    Get model information
    """
    current_model = load_model()
    
    if current_model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'model_name': 'Pneumonia Detection Model',
        'input_shape': list(INPUT_SHAPE),
        'classes': CLASS_NAMES,
        'model_path': MODEL_PATH
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make prediction on uploaded image
    """
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}), 400
    
    try:
        # Load model
        current_model = load_model()
        if current_model is None:
            return jsonify({'error': 'Model not available'}), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        preprocessed_image = preprocess_single_image(filepath)
        
        # Make prediction
        prediction = current_model.predict(preprocessed_image, verbose=0)
        probability = float(prediction[0][0])
        
        predicted_class = CLASS_NAMES[1] if probability > 0.5 else CLASS_NAMES[0]
        confidence = probability if probability > 0.5 else 1 - probability
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'prediction': {
                'class': predicted_class,
                'confidence': float(confidence),
                'probability': probability,
                'is_pneumonia': predicted_class == 'Pneumonia'
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict-with-gradcam', methods=['POST'])
def predict_with_gradcam():
    """
    Make prediction with Grad-CAM visualization
    """
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}), 400
    
    try:
        # Load model
        current_model = load_model()
        if current_model is None:
            return jsonify({'error': 'Model not available'}), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        preprocessed_image = preprocess_single_image(filepath)
        
        # Load original image
        original_image = cv2.imread(filepath)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Make prediction
        prediction = current_model.predict(preprocessed_image, verbose=0)
        probability = float(prediction[0][0])
        
        predicted_class = CLASS_NAMES[1] if probability > 0.5 else CLASS_NAMES[0]
        confidence = probability if probability > 0.5 else 1 - probability
        
        # Generate Grad-CAM
        gradcam = GradCAM(current_model)
        gradcam_results = gradcam.generate_gradcam(preprocessed_image, original_image)
        
        # Convert images to base64
        original_base64 = image_to_base64(gradcam_results['original'])
        heatmap_base64 = image_to_base64(
            cv2.applyColorMap(
                np.uint8(255 * gradcam_results['heatmap']),
                cv2.COLORMAP_JET
            )
        )
        superimposed_base64 = image_to_base64(gradcam_results['superimposed'])
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'prediction': {
                'class': predicted_class,
                'confidence': float(confidence),
                'probability': probability,
                'is_pneumonia': predicted_class == 'Pneumonia'
            },
            'gradcam': {
                'original': original_base64,
                'heatmap': heatmap_base64,
                'overlay': superimposed_base64
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Make predictions on multiple images
    """
    # Check if files are present
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        # Load model
        current_model = load_model()
        if current_model is None:
            return jsonify({'error': 'Model not available'}), 500
        
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                # Save uploaded file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Preprocess image
                preprocessed_image = preprocess_single_image(filepath)
                
                # Make prediction
                prediction = current_model.predict(preprocessed_image, verbose=0)
                probability = float(prediction[0][0])
                
                predicted_class = CLASS_NAMES[1] if probability > 0.5 else CLASS_NAMES[0]
                confidence = probability if probability > 0.5 else 1 - probability
                
                results.append({
                    'filename': filename,
                    'prediction': {
                        'class': predicted_class,
                        'confidence': float(confidence),
                        'probability': probability,
                        'is_pneumonia': predicted_class == 'Pneumonia'
                    }
                })
                
                # Clean up
                os.remove(filepath)
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """
    Handle file too large error
    """
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


@app.errorhandler(500)
def internal_server_error(error):
    """
    Handle internal server error
    """
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Load model on startup
    print("Starting Pneumonia Detection API...")
    load_model()
    
    # Run app
    app.run(
        host=API_HOST,
        port=API_PORT,
        debug=True
    )
