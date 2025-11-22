"""
Deep Learning Model Architecture
Implements CNN with Transfer Learning (ResNet50) for Pneumonia Detection
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
import os
from config import *


class PneumoniaDetectionModel:
    """
    Pneumonia Detection Model using Transfer Learning
    """
    
    def __init__(self, model_type='resnet50', use_pretrained=True):
        """
        Initialize the model
        
        Args:
            model_type: Type of model ('resnet50', 'vgg16', or 'custom_cnn')
            use_pretrained: Whether to use pretrained weights
        """
        self.model_type = model_type
        self.use_pretrained = use_pretrained
        self.model = None
        self.history = None
        
    def build_resnet50_model(self):
        """
        Build model using ResNet50 as base
        
        Returns:
            Compiled Keras model
        """
        # Load pre-trained ResNet50 without top layers
        base_model = ResNet50(
            weights='imagenet' if self.use_pretrained else None,
            include_top=False,
            input_shape=INPUT_SHAPE
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Build the model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ], name='ResNet50_Pneumonia_Detector')
        
        return model
    
    def build_vgg16_model(self):
        """
        Build model using VGG16 as base
        
        Returns:
            Compiled Keras model
        """
        # Load pre-trained VGG16 without top layers
        base_model = VGG16(
            weights='imagenet' if self.use_pretrained else None,
            include_top=False,
            input_shape=INPUT_SHAPE
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Build the model
        model = models.Sequential([
            base_model,
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ], name='VGG16_Pneumonia_Detector')
        
        return model
    
    def build_custom_cnn_model(self):
        """
        Build custom CNN model from scratch
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ], name='Custom_CNN_Pneumonia_Detector')
        
        return model
    
    def build_model(self):
        """
        Build the model based on specified type
        
        Returns:
            Compiled Keras model
        """
        if self.model_type == 'resnet50':
            self.model = self.build_resnet50_model()
        elif self.model_type == 'vgg16':
            self.model = self.build_vgg16_model()
        elif self.model_type == 'custom_cnn':
            self.model = self.build_custom_cnn_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def compile_model(self, learning_rate=LEARNING_RATE):
        """
        Compile the model with optimizer and loss function
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        print(f"\n{self.model_type.upper()} Model compiled successfully!")
        return self.model
    
    def get_callbacks(self, monitor='val_loss', patience=5):
        """
        Get training callbacks
        
        Args:
            monitor: Metric to monitor
            patience: Patience for early stopping
            
        Returns:
            List of callbacks
        """
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                BEST_MODEL_PATH,
                monitor=monitor,
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, train_generator, val_generator, epochs=EPOCHS):
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of epochs
            
        Returns:
            Training history
        """
        if self.model is None:
            self.compile_model()
        
        print("\nStarting model training...")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {BATCH_SIZE}")
        
        callbacks = self.get_callbacks()
        
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        return self.history
    
    def fine_tune(self, train_generator, val_generator, 
                  unfreeze_layers=50, epochs=10, learning_rate=1e-5):
        """
        Fine-tune the model by unfreezing some layers
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            unfreeze_layers: Number of layers to unfreeze from the end
            epochs: Number of fine-tuning epochs
            learning_rate: Learning rate for fine-tuning
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before fine-tuning")
        
        # Robustly find the base model (feature extractor)
        base_model = None
        
        # 1. Check if the first layer is the base model (standard case)
        if len(self.model.layers) > 0 and hasattr(self.model.layers[0], 'layers'):
            base_model = self.model.layers[0]
        
        # 2. If not, search for it (in case InputLayer is first)
        if base_model is None:
            for layer in self.model.layers:
                # Look for a layer that is a Model/Functional (has layers) and is not the main model itself
                if hasattr(layer, 'layers') and len(layer.layers) > 0:
                    base_model = layer
                    break
        
        # 3. If still not found, assume the model itself is flat (e.g. VGG16 loaded directly)
        if base_model is None:
            print("Note: No nested base model found. Fine-tuning the main model layers directly.")
            base_model = self.model

        print(f"Base model for fine-tuning: {base_model.name}")
        
        # Unfreeze the base model
        base_model.trainable = True
        
        # Freeze all layers except the last 'unfreeze_layers'
        # We filter out InputLayer objects which don't have weights
        layers_to_freeze = [l for l in base_model.layers if not isinstance(l, keras.layers.InputLayer)]
        
        # Safety check
        if len(layers_to_freeze) > unfreeze_layers:
            for layer in layers_to_freeze[:-unfreeze_layers]:
                layer.trainable = False
        else:
            print(f"Warning: Requested to unfreeze {unfreeze_layers} layers, but model only has {len(layers_to_freeze)}. All layers will be trainable.")
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        print(f"\nFine-tuning model (unfreezing last {unfreeze_layers} layers)...")
        
        callbacks = self.get_callbacks()
        
        fine_tune_history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return fine_tune_history
    
    def save_model(self, filepath=MODEL_PATH):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath=MODEL_PATH):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"\nModel loaded from {filepath}")
        return self.model
    
    def get_model_summary(self):
        """
        Print model summary
        """
        if self.model is None:
            self.build_model()
        
        return self.model.summary()
    
    def predict(self, image):
        """
        Make prediction on a single image
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Prediction probability and class
        """
        if self.model is None:
            raise ValueError("Model must be loaded or trained before prediction")
        
        prediction = self.model.predict(image, verbose=0)
        probability = prediction[0][0]
        predicted_class = CLASS_NAMES[1] if probability > 0.5 else CLASS_NAMES[0]
        confidence = probability if probability > 0.5 else 1 - probability
        
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'probability': float(probability)
        }


if __name__ == "__main__":
    # Example usage
    print("Pneumonia Detection Model")
    print("=" * 50)
    
    # Create model
    model_builder = PneumoniaDetectionModel(model_type='resnet50')
    model_builder.build_model()
    model_builder.compile_model()
    
    print("\nModel Summary:")
    model_builder.get_model_summary()
    
    print("\nModel built successfully!")
    print("Use this model in your training script.")
