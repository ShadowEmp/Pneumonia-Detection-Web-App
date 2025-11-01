"""
Configuration for RSNA Pneumonia Detection Training
Dataset: RSNA Pneumonia Detection Challenge - 26,684 images
Perfectly labeled for pneumonia detection, optimized for Google Colab
"""
import os

# ============================================================================
# DATASET CONFIGURATION - RSNA (Perfect for Pneumonia Detection)
# ============================================================================

# RSNA Pneumonia Detection Challenge Dataset
DATASET_NAME = "rsna-pneumonia-detection"
KAGGLE_DATASET = "rsna-pneumonia-detection-challenge"

# Dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'rsna')
TRAIN_CSV = os.path.join(DATA_DIR, 'stage_2_train_labels.csv')
IMAGES_DIR = os.path.join(DATA_DIR, 'stage_2_train_images')

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

# Create directories
for directory in [OUTPUT_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# IMAGE CONFIGURATION
# ============================================================================

# Image dimensions (optimized for Colab)
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

# Class configuration
CLASS_NAMES = ['Normal', 'Pneumonia']
NUM_CLASSES = len(CLASS_NAMES)

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# Training parameters (optimized for 26K images)
EPOCHS = 50  # Increased for better training
BATCH_SIZE = 32
LEARNING_RATE = 3e-4  # Increased for better convergence
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Early stopping
EARLY_STOPPING_PATIENCE = 10  # More patience
REDUCE_LR_PATIENCE = 4
MIN_LEARNING_RATE = 1e-7

# Model architecture
BASE_MODEL = 'EfficientNetB1'
FREEZE_BASE_LAYERS = True
UNFREEZE_LAYERS = 50

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

AUGMENTATION_CONFIG = {
    'rotation_range': 15,
    'width_shift_range': 0.15,
    'height_shift_range': 0.15,
    'shear_range': 0.1,
    'zoom_range': [0.9, 1.1],
    'brightness_range': [0.85, 1.15],
    'horizontal_flip': True,
    'vertical_flip': False,
    'fill_mode': 'nearest'
}

USE_ADVANCED_AUGMENTATION = True
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

DENSE_LAYERS = [512, 256]
DROPOUT_RATE = 0.5
ACTIVATION = 'relu'
OUTPUT_ACTIVATION = 'sigmoid'

# ============================================================================
# OPTIMIZATION
# ============================================================================

OPTIMIZER = 'adam'
LOSS_FUNCTION = 'binary_crossentropy'
METRICS = ['accuracy', 'AUC', 'Precision', 'Recall']

USE_CLASS_WEIGHTS = True
AUTO_CLASS_WEIGHTS = True
FOCAL_LOSS = False  # Use focal loss for hard examples

# ============================================================================
# CALLBACKS
# ============================================================================

CHECKPOINT_MONITOR = 'val_auc'
CHECKPOINT_MODE = 'max'
SAVE_BEST_ONLY = True

USE_TENSORBOARD = True
TENSORBOARD_UPDATE_FREQ = 'epoch'
USE_CSV_LOGGER = True
USE_LR_SCHEDULE = True
LR_SCHEDULE_TYPE = 'cosine'

# ============================================================================
# FINE-TUNING
# ============================================================================

FINE_TUNE = True
FINE_TUNE_EPOCHS = 30  # Increased for better fine-tuning
FINE_TUNE_LEARNING_RATE = 5e-6
FINE_TUNE_UNFREEZE_LAYERS = 80

# ============================================================================
# EVALUATION
# ============================================================================

USE_TTA = True
TTA_STEPS = 5
PREDICTION_THRESHOLD = 0.3

COMPUTE_CONFUSION_MATRIX = True
COMPUTE_ROC_CURVE = True
COMPUTE_PR_CURVE = True
SAVE_PREDICTIONS = True

# ============================================================================
# VISUALIZATION
# ============================================================================

FIGURE_DPI = 300
FIGURE_SIZE = (12, 8)
SAVE_SAMPLE_IMAGES = True
NUM_SAMPLE_IMAGES = 16

# ============================================================================
# GOOGLE COLAB SPECIFIC
# ============================================================================

USE_GOOGLE_DRIVE = True
DRIVE_MOUNT_PATH = '/content/drive'
DRIVE_OUTPUT_PATH = '/content/drive/MyDrive/pneumonia_detection_rsna'

USE_MIXED_PRECISION = True
GPU_MEMORY_GROWTH = True

# ============================================================================
# DATA LOADING
# ============================================================================

NUM_WORKERS = 4
USE_MULTIPROCESSING = True
CACHE_DATASET = False
PREFETCH_BUFFER = 3

# ============================================================================
# RANDOM SEED
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# MODEL PATHS
# ============================================================================

MODEL_NAME = f'pneumonia_detection_{BASE_MODEL.lower()}_256x256_rsna'
MODEL_PATH = os.path.join(MODEL_DIR, f'{MODEL_NAME}.h5')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, f'{MODEL_NAME}_best.h5')
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, f'{MODEL_NAME}_final.h5')

TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, 'training_history.png')
FINE_TUNING_HISTORY_PATH = os.path.join(RESULTS_DIR, 'fine_tuning_history.png')
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
ROC_CURVE_PATH = os.path.join(RESULTS_DIR, 'roc_curve.png')
PR_CURVE_PATH = os.path.join(RESULTS_DIR, 'precision_recall_curve.png')
METRICS_PATH = os.path.join(RESULTS_DIR, 'metrics.json')
SAMPLE_IMAGES_PATH = os.path.join(RESULTS_DIR, 'sample_images.png')

# ============================================================================
# DATASET STATISTICS
# ============================================================================

TOTAL_IMAGES = 26684
PNEUMONIA_IMAGES_APPROX = 9555
NORMAL_IMAGES_APPROX = 17129

print("=" * 80)
print("RSNA PNEUMONIA DETECTION TRAINING CONFIGURATION")
print("=" * 80)
print(f"Dataset: RSNA Pneumonia Detection Challenge")
print(f"Total Images: {TOTAL_IMAGES:,}")
print(f"  - Pneumonia: ~{PNEUMONIA_IMAGES_APPROX:,} ({PNEUMONIA_IMAGES_APPROX/TOTAL_IMAGES*100:.1f}%)")
print(f"  - Normal: ~{NORMAL_IMAGES_APPROX:,} ({NORMAL_IMAGES_APPROX/TOTAL_IMAGES*100:.1f}%)")
print(f"Image Size: {IMG_HEIGHT}x{IMG_WIDTH}")
print(f"Base Model: {BASE_MODEL}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS} + {FINE_TUNE_EPOCHS} fine-tuning")
print(f"Optimized for: Google Colab")
print("=" * 80)
