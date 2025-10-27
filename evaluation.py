"""
Model Evaluation Module
Provides comprehensive evaluation metrics and visualizations
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score
)
from config import *


class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations
    """
    
    def __init__(self, model):
        """
        Initialize evaluator
        
        Args:
            model: Trained Keras model
        """
        self.model = model
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
        
    def evaluate(self, test_data):
        """
        Evaluate model on test data
        
        Args:
            test_data: Test data (generator or tuple of (X_test, y_test))
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\nEvaluating model...")
        
        # Check if test_data is a generator or array
        if hasattr(test_data, 'next') or hasattr(test_data, '__next__'):
            # It's a generator
            test_generator = test_data
            
            # Get predictions
            self.y_pred_proba = self.model.predict(test_generator, verbose=1)
            self.y_pred = (self.y_pred_proba > 0.5).astype(int).flatten()
            self.y_true = test_generator.classes
            
            # Evaluate metrics
            results = self.model.evaluate(test_generator, verbose=1)
            metric_names = self.model.metrics_names
            
        else:
            # It's an array
            X_test, y_test = test_data
            self.y_true = y_test
            
            # Get predictions
            self.y_pred_proba = self.model.predict(X_test, verbose=1)
            self.y_pred = (self.y_pred_proba > 0.5).astype(int).flatten()
            
            # Evaluate metrics
            results = self.model.evaluate(X_test, y_test, verbose=1)
            metric_names = self.model.metrics_names
        
        # Create results dictionary
        results_dict = dict(zip(metric_names, results))
        
        # Calculate additional metrics
        results_dict['f1_score'] = f1_score(self.y_true, self.y_pred)
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        for metric, value in results_dict.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        print("="*60)
        
        return results_dict
    
    def get_confusion_matrix(self):
        """
        Calculate confusion matrix
        
        Returns:
            Confusion matrix array
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Model must be evaluated first")
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        return cm
    
    def plot_confusion_matrix(self, save_path=None, show=True):
        """
        Plot confusion matrix
        
        Args:
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        cm = self.get_confusion_matrix()
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add accuracy information
        accuracy = np.trace(cm) / np.sum(cm)
        plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.2%}',
                ha='center', transform=plt.gca().transAxes, fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_roc_curve(self, save_path=None, show=True):
        """
        Plot ROC curve
        
        Args:
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if self.y_true is None or self.y_pred_proba is None:
            raise ValueError("Model must be evaluated first")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve',
                 fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return roc_auc
    
    def plot_precision_recall_curve(self, save_path=None, show=True):
        """
        Plot Precision-Recall curve
        
        Args:
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if self.y_true is None or self.y_pred_proba is None:
            raise ValueError("Model must be evaluated first")
        
        # Calculate Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(
            self.y_true, self.y_pred_proba
        )
        avg_precision = average_precision_score(self.y_true, self.y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        
        # Plot Precision-Recall curve
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return avg_precision
    
    def get_classification_report(self):
        """
        Get detailed classification report
        
        Returns:
            Classification report string
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Model must be evaluated first")
        
        report = classification_report(
            self.y_true, self.y_pred,
            target_names=CLASS_NAMES,
            digits=4
        )
        
        return report
    
    def plot_prediction_samples(self, test_data, num_samples=16, save_path=None, show=True):
        """
        Plot sample predictions with true and predicted labels
        
        Args:
            test_data: Test data (X_test, y_test)
            num_samples: Number of samples to display
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if isinstance(test_data, tuple):
            X_test, y_test = test_data
        else:
            # For generators, get a batch
            X_test, y_test = next(iter(test_data))
        
        # Get predictions
        predictions = self.model.predict(X_test[:num_samples], verbose=0)
        pred_classes = (predictions > 0.5).astype(int).flatten()
        
        # Create figure
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten()
        
        for i in range(min(num_samples, len(X_test))):
            axes[i].imshow(X_test[i])
            
            true_label = CLASS_NAMES[int(y_test[i])]
            pred_label = CLASS_NAMES[pred_classes[i]]
            confidence = predictions[i][0] if pred_classes[i] == 1 else 1 - predictions[i][0]
            
            # Color code: green for correct, red for incorrect
            color = 'green' if true_label == pred_label else 'red'
            
            axes[i].set_title(
                f'True: {true_label}\nPred: {pred_label} ({confidence:.2%})',
                color=color, fontsize=10, fontweight='bold'
            )
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Sample predictions saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def generate_full_report(self, test_data, output_dir=RESULTS_DIR):
        """
        Generate complete evaluation report with all visualizations
        
        Args:
            test_data: Test data
            output_dir: Directory to save results
        """
        print("\nGenerating comprehensive evaluation report...")
        
        # Evaluate model
        results = self.evaluate(test_data)
        
        # Generate all visualizations
        self.plot_confusion_matrix(
            save_path=os.path.join(output_dir, 'confusion_matrix.png'),
            show=False
        )
        
        self.plot_roc_curve(
            save_path=os.path.join(output_dir, 'roc_curve.png'),
            show=False
        )
        
        self.plot_precision_recall_curve(
            save_path=os.path.join(output_dir, 'precision_recall_curve.png'),
            show=False
        )
        
        self.plot_prediction_samples(
            test_data,
            save_path=os.path.join(output_dir, 'sample_predictions.png'),
            show=False
        )
        
        # Save text report
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("PNEUMONIA DETECTION MODEL - EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write("-" * 70 + "\n")
            for metric, value in results.items():
                f.write(f"{metric.capitalize()}: {value:.4f}\n")
            
            f.write("\n\nCLASSIFICATION REPORT:\n")
            f.write("-" * 70 + "\n")
            f.write(self.get_classification_report())
            
            f.write("\n\nCONFUSION MATRIX:\n")
            f.write("-" * 70 + "\n")
            cm = self.get_confusion_matrix()
            f.write(f"\n{cm}\n")
        
        print(f"\nComplete evaluation report saved to {output_dir}")


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("=" * 50)
    print("\nThis module provides:")
    print("- Confusion matrix")
    print("- ROC curve and AUC")
    print("- Precision-Recall curve")
    print("- Classification report")
    print("- Sample predictions visualization")
    print("\nUse this module with a trained model for evaluation.")
