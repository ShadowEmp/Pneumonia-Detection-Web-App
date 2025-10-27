"""
Configuration file for Pneumonia Detection System
"""
import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR, UPLOADS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Training Configuration
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2

# Model Paths
MODEL_PATH = os.path.join(MODEL_DIR, 'pneumonia_model.h5')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_pneumonia_model.h5')

# Class Labels
CLASS_NAMES = ['Normal', 'Pneumonia']
NUM_CLASSES = len(CLASS_NAMES)

# Data Augmentation Parameters
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# API Configuration
API_HOST = '0.0.0.0'
API_PORT = 5000
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Visualization Settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_DPI = 100
HEATMAP_ALPHA = 0.4
HEATMAP_COLORMAP = 'jet'
