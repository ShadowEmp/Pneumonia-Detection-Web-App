"""
Training Script for Pneumonia Detection Model
Handles complete training pipeline with visualization
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from config import *
from data_preprocessing import DataPreprocessor
from model import PneumoniaDetectionModel
from evaluation import ModelEvaluator


class ModelTrainer:
    """
    Handles the complete training pipeline
    """
    
    def __init__(self, data_path, model_type='resnet50'):
        """
        Initialize trainer
        
        Args:
            data_path: Path to dataset directory
            model_type: Type of model to train
        """
        self.data_path = data_path
        self.model_type = model_type
        self.preprocessor = DataPreprocessor(data_path)
        self.model_builder = PneumoniaDetectionModel(model_type=model_type)
        self.history = None
        
    def prepare_data(self, use_generators=True):
        """
        Prepare training, validation, and test data
        
        Args:
            use_generators: Whether to use data generators or load all data
            
        Returns:
            Data generators or arrays
        """
        print("\n" + "="*60)
        print("PREPARING DATA")
        print("="*60)
        
        if use_generators:
            # Assuming directory structure: data_path/train, data_path/val, data_path/test
            train_path = os.path.join(self.data_path, 'train')
            val_path = os.path.join(self.data_path, 'val')
            test_path = os.path.join(self.data_path, 'test')
            
            # Check if directories exist
            if not os.path.exists(train_path):
                print(f"Warning: {train_path} not found.")
                print("Please organize your data in the following structure:")
                print("data/")
                print("  train/")
                print("    Normal/")
                print("    Pneumonia/")
                print("  val/")
                print("    Normal/")
                print("    Pneumonia/")
                print("  test/")
                print("    Normal/")
                print("    Pneumonia/")
                return None, None, None
            
            train_gen, val_gen, test_gen = self.preprocessor.create_data_generators(
                train_path, val_path if os.path.exists(val_path) else None,
                test_path if os.path.exists(test_path) else None
            )
            
            print(f"\nTraining samples: {train_gen.samples}")
            print(f"Validation samples: {val_gen.samples}")
            if test_gen:
                print(f"Test samples: {test_gen.samples}")
            
            return train_gen, val_gen, test_gen
        else:
            # Load all data into memory
            X, y = self.preprocessor.load_dataset_from_directory(self.data_path)
            X_train, X_val, X_test, y_train, y_val, y_test = \
                self.preprocessor.split_dataset(X, y)
            
            return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train_model(self, train_data, val_data, epochs=EPOCHS, 
                   fine_tune=False, fine_tune_epochs=10):
        """
        Train the model
        
        Args:
            train_data: Training data (generator or array)
            val_data: Validation data (generator or array)
            epochs: Number of training epochs
            fine_tune: Whether to fine-tune after initial training
            fine_tune_epochs: Number of fine-tuning epochs
            
        Returns:
            Training history
        """
        print("\n" + "="*60)
        print("BUILDING AND COMPILING MODEL")
        print("="*60)
        
        # Build and compile model
        self.model_builder.build_model()
        self.model_builder.compile_model()
        
        # Print model summary
        print("\nModel Architecture:")
        self.model_builder.get_model_summary()
        
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        # Train model
        self.history = self.model_builder.train(
            train_data, val_data, epochs=epochs
        )
        
        # Fine-tuning (optional)
        if fine_tune and self.model_type in ['resnet50', 'vgg16']:
            print("\n" + "="*60)
            print("FINE-TUNING MODEL")
            print("="*60)
            
            fine_tune_history = self.model_builder.fine_tune(
                train_data, val_data,
                unfreeze_layers=50,
                epochs=fine_tune_epochs,
                learning_rate=1e-5
            )
            
            # Combine histories
            for key in self.history.history.keys():
                self.history.history[key].extend(fine_tune_history.history[key])
        
        # Save model
        self.model_builder.save_model()
        
        return self.history
    
    def plot_training_history(self, save_path=None):
        """
        Plot training and validation metrics
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            print("No training history available")
            return
        
        history = self.history.history
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot accuracy
        axes[0, 0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[0, 1].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[0, 1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot precision
        if 'precision' in history:
            axes[1, 0].plot(history['precision'], label='Training Precision', linewidth=2)
            axes[1, 0].plot(history['val_precision'], label='Validation Precision', linewidth=2)
            axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot recall
        if 'recall' in history:
            axes[1, 1].plot(history['recall'], label='Training Recall', linewidth=2)
            axes[1, 1].plot(history['val_recall'], label='Validation Recall', linewidth=2)
            axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(RESULTS_DIR, 'training_history.png')
        
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"\nTraining history plot saved to {save_path}")
        plt.show()
    
    def evaluate_model(self, test_data):
        """
        Evaluate the trained model
        
        Args:
            test_data: Test data (generator or array)
        """
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        evaluator = ModelEvaluator(self.model_builder.model)
        
        # Evaluate
        results = evaluator.evaluate(test_data)
        
        # Generate visualizations
        evaluator.plot_confusion_matrix(
            save_path=os.path.join(RESULTS_DIR, 'confusion_matrix.png')
        )
        
        evaluator.plot_roc_curve(
            save_path=os.path.join(RESULTS_DIR, 'roc_curve.png')
        )
        
        evaluator.plot_precision_recall_curve(
            save_path=os.path.join(RESULTS_DIR, 'precision_recall_curve.png')
        )
        
        # Save classification report
        report_path = os.path.join(RESULTS_DIR, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("PNEUMONIA DETECTION MODEL - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model Type: {self.model_type}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(evaluator.get_classification_report())
        
        print(f"\nEvaluation report saved to {report_path}")
        
        return results


def main():
    """
    Main training function
    """
    print("\n" + "="*60)
    print("PNEUMONIA DETECTION MODEL - TRAINING PIPELINE")
    print("="*60)
    
    # Configuration
    DATA_PATH = DATA_DIR  # Update this to your dataset path
    MODEL_TYPE = 'resnet50'  # Options: 'resnet50', 'vgg16', 'custom_cnn'
    USE_GENERATORS = True  # Set to False if you want to load all data into memory
    EPOCHS = 30
    FINE_TUNE = True
    FINE_TUNE_EPOCHS = 10
    
    # Initialize trainer
    trainer = ModelTrainer(DATA_PATH, model_type=MODEL_TYPE)
    
    # Prepare data
    if USE_GENERATORS:
        train_gen, val_gen, test_gen = trainer.prepare_data(use_generators=True)
        
        if train_gen is None:
            print("\nPlease set up your dataset and try again.")
            return
        
        # Train model
        trainer.train_model(
            train_gen, val_gen,
            epochs=EPOCHS,
            fine_tune=FINE_TUNE,
            fine_tune_epochs=FINE_TUNE_EPOCHS
        )
        
        # Plot training history
        trainer.plot_training_history()
        
        # Evaluate model
        if test_gen:
            trainer.evaluate_model(test_gen)
    else:
        train_data, val_data, test_data = trainer.prepare_data(use_generators=False)
        
        # Train model
        trainer.train_model(
            train_data, val_data,
            epochs=EPOCHS,
            fine_tune=FINE_TUNE,
            fine_tune_epochs=FINE_TUNE_EPOCHS
        )
        
        # Plot training history
        trainer.plot_training_history()
        
        # Evaluate model
        trainer.evaluate_model(test_data)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
