"""
RSNA Pneumonia Detection Dataset Handler
26,684 chest X-rays properly labeled for pneumonia detection
"""
import os
import pandas as pd
import numpy as np
import cv2
import pydicom
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tqdm import tqdm
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
        """
        Load RSNA data and prepare for binary classification
        """
        print("Loading RSNA Pneumonia Detection dataset...")
        print("=" * 80)
        
        # Load CSV
        df = pd.read_csv(self.train_csv)
        
        print(f"Original samples: {len(df):,}")
        print(f"\nColumns: {df.columns.tolist()}")
        
        # RSNA format: patientId, x, y, width, height, Target
        # Target: 1 = pneumonia, 0 = normal
        # Multiple rows per patient if multiple pneumonia regions
        
        # Group by patient and take max Target (if any region has pneumonia, patient has pneumonia)
        df_grouped = df.groupby('patientId').agg({
            'Target': 'max'  # 1 if any pneumonia region, 0 if all normal
        }).reset_index()
        
        # Convert labels to strings for Keras ImageDataGenerator
        df_grouped['label'] = df_grouped['Target'].apply(lambda x: 'Pneumonia' if x == 1 else 'Normal')
        df_grouped['label_numeric'] = df_grouped['Target']  # Keep numeric for class weights
        df_grouped['image_path'] = df_grouped['patientId'].apply(
            lambda x: os.path.join(self.images_dir, f'{x}.dcm')
        )
        
        self.df = df_grouped
        
        print(f"\nDataset prepared:")
        print(f"  Total patients: {len(self.df):,}")
        print(f"  Pneumonia: {(self.df['label'] == 'Pneumonia').sum():,} ({(self.df['label'] == 'Pneumonia').sum()/len(self.df)*100:.1f}%)")
        print(f"  Normal: {(self.df['label'] == 'Normal').sum():,} ({(self.df['label'] == 'Normal').sum()/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def convert_dicom_to_png(self, dicom_path):
        """Convert DICOM image to PNG format"""
        try:
            # Read DICOM
            dcm = pydicom.dcmread(dicom_path)
            img = dcm.pixel_array
            
            # Normalize to 0-255
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            return img_rgb
        except Exception as e:
            print(f"Error converting {dicom_path}: {e}")
            return None
    
    def verify_images(self, sample_size=100):
        """Verify that DICOM images exist and are readable"""
        print(f"\nVerifying {sample_size} random images...")
        
        sample_df = self.df.sample(n=min(sample_size, len(self.df)))
        valid_count = 0
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
            img_path = row['image_path']
            if os.path.exists(img_path):
                try:
                    img = self.convert_dicom_to_png(img_path)
                    if img is not None:
                        valid_count += 1
                except:
                    pass
        
        print(f"Valid images: {valid_count}/{len(sample_df)} ({valid_count/len(sample_df)*100:.1f}%)")
        
        return valid_count / len(sample_df)
    
    def split_dataset(self, test_size=TEST_SPLIT, val_size=VALIDATION_SPLIT):
        """Split dataset into train, validation, and test sets"""
        print("\nSplitting dataset...")
        
        # First split: train+val vs test (use label_numeric for stratification)
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=test_size,
            stratify=self.df['label_numeric'],
            random_state=RANDOM_SEED
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            stratify=train_val_df['label_numeric'],
            random_state=RANDOM_SEED
        )
        
        print(f"\nDataset split:")
        print(f"  Training: {len(train_df):,} images")
        print(f"  Validation: {len(val_df):,} images")
        print(f"  Test: {len(test_df):,} images")
        
        # Print class distribution
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            pneumonia_count = (split_df['label'] == 'Pneumonia').sum()
            normal_count = (split_df['label'] == 'Normal').sum()
            print(f"\n{split_name} distribution:")
            print(f"  Pneumonia: {pneumonia_count:,} ({pneumonia_count/len(split_df)*100:.1f}%)")
            print(f"  Normal: {normal_count:,} ({normal_count/len(split_df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def compute_class_weights(self, train_df):
        """Compute class weights for imbalanced dataset"""
        if not AUTO_CLASS_WEIGHTS:
            return None
        
        print("\nComputing class weights...")
        
        # Use numeric labels for class weight computation
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_df['label_numeric']),
            y=train_df['label_numeric']
        )
        
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        print(f"Class weights: {class_weight_dict}")
        
        self.class_weights = class_weight_dict
        return class_weight_dict
    
    def preprocess_function(self, img_path):
        """Preprocessing function for DICOM images"""
        img = self.convert_dicom_to_png(img_path)
        if img is None:
            # Return black image if conversion fails
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        return img / 255.0
    
    def create_data_generators(self, train_df, val_df, test_df):
        """Create data generators for training"""
        print("\nCreating data generators...")
        
        # Training generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=lambda x: self.convert_dicom_to_png(x) if isinstance(x, str) else x,
            **AUGMENTATION_CONFIG
        )
        
        # Validation and test generators (no augmentation)
        val_test_datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=lambda x: self.convert_dicom_to_png(x) if isinstance(x, str) else x
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col='image_path',
            y_col='label',
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=True,
            seed=RANDOM_SEED
        )
        
        val_generator = val_test_datagen.flow_from_dataframe(
            dataframe=val_df,
            x_col='image_path',
            y_col='label',
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_dataframe(
            dataframe=test_df,
            x_col='image_path',
            y_col='label',
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )
        
        print(f"Train generator: {len(train_generator)} batches")
        print(f"Validation generator: {len(val_generator)} batches")
        print(f"Test generator: {len(test_generator)} batches")
        
        return train_generator, val_generator, test_generator
    
    def visualize_samples(self, train_df, num_samples=16, save_path=None):
        """Visualize sample images from dataset"""
        print(f"\nVisualizing {num_samples} sample images...")
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()
        
        # Get balanced samples (use string labels)
        pneumonia_samples = train_df[train_df['label'] == 'Pneumonia'].sample(n=num_samples//2)
        normal_samples = train_df[train_df['label'] == 'Normal'].sample(n=num_samples//2)
        samples = pd.concat([pneumonia_samples, normal_samples]).sample(frac=1)
        
        for idx, (_, row) in enumerate(samples.iterrows()):
            if idx >= num_samples:
                break
                
            img_path = row['image_path']
            
            if os.path.exists(img_path):
                img = self.convert_dicom_to_png(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
                    
                    axes[idx].imshow(img)
                    label = row['label']  # Already a string
                    axes[idx].set_title(f'{label}', fontsize=12, fontweight='bold')
                    axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Sample images saved to {save_path}")
        
        plt.close()


def download_rsna_dataset():
    """Instructions for downloading RSNA dataset"""
    print("=" * 80)
    print("RSNA PNEUMONIA DETECTION DATASET DOWNLOAD")
    print("=" * 80)
    print("\nRSNA Pneumonia Detection Challenge dataset")
    print("Total: 26,684 images (~3GB)")
    print("Perfectly labeled for pneumonia detection!")
    print("\nDownload from Kaggle:")
    print("-" * 60)
    print("Dataset: rsna-pneumonia-detection-challenge")
    print("Size: ~3GB")
    print("\nSteps:")
    print("1. Setup Kaggle API credentials")
    print("2. Run: kaggle competitions download -c rsna-pneumonia-detection-challenge")
    print("3. Extract to data/rsna/")
    print("\nAfter download, your directory structure should be:")
    print("  data/rsna/")
    print("    ├── stage_2_train_images/  (DICOM files)")
    print("    └── stage_2_train_labels.csv")
    print("=" * 80)


if __name__ == "__main__":
    print("RSNA Pneumonia Detection Dataset Handler")
    print("=" * 80)
    
    download_rsna_dataset()
