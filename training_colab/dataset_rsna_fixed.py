"""
RSNA Pneumonia Detection Dataset Handler - Fixed for Colab
26,684 chest X-rays properly labeled for pneumonia detection
"""
import os
import pandas as pd
import numpy as np
import cv2
import pydicom
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from config_rsna import *


class RSNAPneumoniaDataset:
    """Handler for RSNA Pneumonia Detection dataset"""
    
    def __init__(self, data_dir=DATA_DIR, train_csv=TRAIN_CSV, images_dir=IMAGES_DIR):
        self.data_dir = data_dir
        self.train_csv = train_csv
        self.images_dir = images_dir
        self.df = None
        self.class_weights = None
        
    def load_and_prepare_data(self):
        """Load RSNA data and prepare for binary classification"""
        print("Loading RSNA Pneumonia Detection dataset...")
        print("=" * 80)
        
        # Load CSV
        df = pd.read_csv(self.train_csv)
        
        print(f"Original samples: {len(df):,}")
        
        # Group by patient and take max Target
        df_grouped = df.groupby('patientId').agg({
            'Target': 'max'
        }).reset_index()
        
        # Create labels
        df_grouped['label'] = df_grouped['Target']
        df_grouped['image_path'] = df_grouped['patientId'].apply(
            lambda x: os.path.join(self.images_dir, f'{x}.dcm')
        )
        
        # Filter only existing files
        print("\nFiltering existing images...")
        df_grouped['exists'] = df_grouped['image_path'].apply(os.path.exists)
        df_grouped = df_grouped[df_grouped['exists']].copy()
        df_grouped = df_grouped.drop('exists', axis=1)
        
        self.df = df_grouped
        
        print(f"\nDataset prepared:")
        print(f"  Total patients: {len(self.df):,}")
        print(f"  Pneumonia: {(self.df['label'] == 1).sum():,} ({(self.df['label'] == 1).sum()/len(self.df)*100:.1f}%)")
        print(f"  Normal: {(self.df['label'] == 0).sum():,} ({(self.df['label'] == 0).sum()/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def convert_dicom_to_array(self, dicom_path):
        """Convert DICOM to numpy array"""
        try:
            # Convert to string if it's bytes
            if isinstance(dicom_path, bytes):
                dicom_path = dicom_path.decode('utf-8')
            else:
                dicom_path = str(dicom_path)
            
            dcm = pydicom.dcmread(dicom_path)
            img = dcm.pixel_array
            
            # Handle different photometric interpretations
            if hasattr(dcm, 'PhotometricInterpretation'):
                if dcm.PhotometricInterpretation == "MONOCHROME1":
                    img = np.max(img) - img
            
            # Normalize to 0-255
            if img.max() > img.min():
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            
            # Resize
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            return img_rgb
        except Exception as e:
            # Return gray image on error (not black to avoid bias)
            return np.full((IMG_HEIGHT, IMG_WIDTH, 3), 128, dtype=np.uint8)
    
    def split_dataset(self, test_size=TEST_SPLIT, val_size=VALIDATION_SPLIT):
        """Split dataset"""
        print("\nSplitting dataset...")
        
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=test_size,
            stratify=self.df['label'],
            random_state=RANDOM_SEED
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            stratify=train_val_df['label'],
            random_state=RANDOM_SEED
        )
        
        print(f"\nDataset split:")
        print(f"  Training: {len(train_df):,}")
        print(f"  Validation: {len(val_df):,}")
        print(f"  Test: {len(test_df):,}")
        
        return train_df, val_df, test_df
    
    def compute_class_weights(self, train_df):
        """Compute class weights"""
        if not AUTO_CLASS_WEIGHTS:
            return None
        
        print("\nComputing class weights...")
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_df['label']),
            y=train_df['label']
        )
        
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        print(f"Class weights: {class_weight_dict}")
        
        self.class_weights = class_weight_dict
        return class_weight_dict
    
    def create_tf_dataset(self, df, batch_size=BATCH_SIZE, shuffle=True, augment=False):
        """Create TensorFlow dataset"""
        
        def load_image(path, label):
            """Load and preprocess image"""
            # Read DICOM - path is already a string tensor
            def _load_dicom(path_bytes):
                path_str = path_bytes.decode('utf-8') if isinstance(path_bytes, bytes) else str(path_bytes)
                return self.convert_dicom_to_array(path_str)
            
            img = tf.py_function(
                func=_load_dicom,
                inp=[path],
                Tout=tf.uint8
            )
            img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
            
            # Normalize to 0-1
            img = tf.cast(img, tf.float32) / 255.0
            
            # ImageNet normalization (EfficientNet expects this)
            mean = tf.constant([0.485, 0.456, 0.406])
            std = tf.constant([0.229, 0.224, 0.225])
            img = (img - mean) / std
            
            return img, label
        
        def augment_image(img, label):
            """Data augmentation"""
            # Random flip (horizontal only for chest X-rays)
            img = tf.image.random_flip_left_right(img)
            
            # Random rotation (small angles)
            angle = tf.random.uniform([], -10, 10) * (3.14159 / 180)
            img = tf.contrib.image.rotate(img, angle) if hasattr(tf.contrib, 'image') else img
            
            # Random brightness
            img = tf.image.random_brightness(img, 0.15)
            
            # Random contrast
            img = tf.image.random_contrast(img, 0.85, 1.15)
            
            # Random zoom (crop and resize)
            if tf.random.uniform([]) > 0.5:
                crop_size = tf.random.uniform([], 0.85, 1.0)
                h = tf.cast(IMG_HEIGHT * crop_size, tf.int32)
                w = tf.cast(IMG_WIDTH * crop_size, tf.int32)
                img = tf.image.random_crop(img, [h, w, 3])
                img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
            
            return img, label
        
        # Create dataset
        paths = df['image_path'].values
        labels = df['label'].values
        
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, seed=RANDOM_SEED)
        
        # Load images
        dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Augmentation
        if augment:
            dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


def download_rsna_dataset():
    """Instructions for downloading RSNA dataset"""
    print("=" * 80)
    print("RSNA PNEUMONIA DETECTION DATASET DOWNLOAD")
    print("=" * 80)
    print("\nRSNA Pneumonia Detection Challenge dataset")
    print("Total: 26,684 images (~3GB)")
    print("\nDownload from Kaggle:")
    print("1. Join competition: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge")
    print("2. Run: kaggle competitions download -c rsna-pneumonia-detection-challenge")
    print("3. Extract to data/rsna/")
    print("=" * 80)


if __name__ == "__main__":
    download_rsna_dataset()
