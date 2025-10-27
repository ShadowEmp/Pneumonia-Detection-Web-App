"""
Data Preprocessing and Augmentation Module
Handles loading, preprocessing, and augmenting chest X-ray images
"""
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from config import *


class DataPreprocessor:
    """
    Handles all data preprocessing operations including loading,
    augmentation, and splitting datasets
    """
    
    def __init__(self, data_path):
        """
        Initialize the data preprocessor
        
        Args:
            data_path: Path to the dataset directory
        """
        self.data_path = data_path
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        
    def load_and_preprocess_image(self, image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)):
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to the image file
            target_size: Target size for resizing (height, width)
            
        Returns:
            Preprocessed image array
        """
        # Read image
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, target_size)
        
        # Normalize pixel values to [0, 1]
        img = img.astype('float32') / 255.0
        
        return img
    
    def load_dataset_from_directory(self, dataset_path):
        """
        Load dataset from directory structure:
        dataset_path/
            Normal/
                image1.jpg
                image2.jpg
            Pneumonia/
                image1.jpg
                image2.jpg
                
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            X: Image array
            y: Label array
        """
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_path = os.path.join(dataset_path, class_name)
            
            if not os.path.exists(class_path):
                print(f"Warning: Directory not found: {class_path}")
                continue
            
            print(f"Loading {class_name} images...")
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                try:
                    img = self.load_and_preprocess_image(img_path)
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {str(e)}")
                    continue
            
            print(f"Loaded {len([l for l in labels if l == class_idx])} {class_name} images")
        
        X = np.array(images)
        y = np.array(labels)
        
        return X, y
    
    def create_data_generators(self, train_path, val_path=None, test_path=None):
        """
        Create data generators with augmentation for training
        
        Args:
            train_path: Path to training data
            val_path: Path to validation data (optional)
            test_path: Path to test data (optional)
            
        Returns:
            train_generator, val_generator, test_generator
        """
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=AUGMENTATION_CONFIG['rotation_range'],
            width_shift_range=AUGMENTATION_CONFIG['width_shift_range'],
            height_shift_range=AUGMENTATION_CONFIG['height_shift_range'],
            shear_range=AUGMENTATION_CONFIG['shear_range'],
            zoom_range=AUGMENTATION_CONFIG['zoom_range'],
            horizontal_flip=AUGMENTATION_CONFIG['horizontal_flip'],
            fill_mode=AUGMENTATION_CONFIG['fill_mode'],
            validation_split=VALIDATION_SPLIT if val_path is None else 0
        )
        
        # Validation and test data generators (no augmentation)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create training generator
        self.train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='training' if val_path is None else None,
            shuffle=True
        )
        
        # Create validation generator
        if val_path:
            self.val_generator = val_test_datagen.flow_from_directory(
                val_path,
                target_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH_SIZE,
                class_mode='binary',
                shuffle=False
            )
        else:
            self.val_generator = train_datagen.flow_from_directory(
                train_path,
                target_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH_SIZE,
                class_mode='binary',
                subset='validation',
                shuffle=False
            )
        
        # Create test generator if test path provided
        if test_path:
            self.test_generator = val_test_datagen.flow_from_directory(
                test_path,
                target_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH_SIZE,
                class_mode='binary',
                shuffle=False
            )
        
        return self.train_generator, self.val_generator, self.test_generator
    
    def split_dataset(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split dataset into train, validation, and test sets
        
        Args:
            X: Image data
            y: Labels
            test_size: Proportion of test set
            val_size: Proportion of validation set from training data
            random_state: Random seed
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: separate test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate validation set from training
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size/(1-test_size),
            random_state=random_state, stratify=y_train_val
        )
        
        print(f"\nDataset Split:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def visualize_samples(self, X, y, num_samples=9, save_path=None):
        """
        Visualize sample images from the dataset
        
        Args:
            X: Image data
            y: Labels
            num_samples: Number of samples to display
            save_path: Path to save the visualization
        """
        plt.figure(figsize=(12, 12))
        
        indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
        
        for i, idx in enumerate(indices):
            plt.subplot(3, 3, i + 1)
            plt.imshow(X[idx])
            plt.title(f"{CLASS_NAMES[y[idx]]}")
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Sample visualization saved to {save_path}")
        
        plt.show()
    
    def get_class_distribution(self, y):
        """
        Get class distribution statistics
        
        Args:
            y: Label array
            
        Returns:
            Dictionary with class counts
        """
        unique, counts = np.unique(y, return_counts=True)
        distribution = dict(zip([CLASS_NAMES[i] for i in unique], counts))
        
        print("\nClass Distribution:")
        for class_name, count in distribution.items():
            percentage = (count / len(y)) * 100
            print(f"{class_name}: {count} ({percentage:.2f}%)")
        
        return distribution


def preprocess_single_image(image_path):
    """
    Preprocess a single image for prediction
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image ready for model input
    """
    preprocessor = DataPreprocessor(None)
    img = preprocessor.load_and_preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Module")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(DATA_DIR)
    
    print("\nThis module provides:")
    print("- Image loading and preprocessing")
    print("- Data augmentation")
    print("- Dataset splitting")
    print("- Visualization utilities")
    print("\nUse this module in your training script.")
