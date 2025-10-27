"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
Provides visual explanations for model predictions
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from config import *


class GradCAM:
    """
    Implements Grad-CAM for visualizing which regions of an image
    the model focuses on for making predictions
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained Keras model
            layer_name: Name of the convolutional layer to visualize
                       If None, uses the last convolutional layer
        """
        self.model = model
        self.layer_name = layer_name
        
        if self.layer_name is None:
            self.layer_name = self._find_last_conv_layer()
        
        print(f"Using layer: {self.layer_name} for Grad-CAM")
    
    def _find_last_conv_layer(self):
        """
        Find the last convolutional layer in the model
        
        Returns:
            Name of the last convolutional layer
        """
        # Check if model is Sequential or has a base model
        if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
            # Check if first layer is a base model (like ResNet50)
            base_model = self.model.layers[0]
            
            if hasattr(base_model, 'layers'):
                # Iterate through base model layers in reverse
                for layer in reversed(base_model.layers):
                    if 'conv' in layer.name.lower():
                        return layer.name
        
        # Fallback: search through all model layers
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        
        raise ValueError("Could not find a convolutional layer in the model")
    
    def compute_heatmap(self, image, pred_index=None, eps=1e-8):
        """
        Compute Grad-CAM heatmap
        
        Args:
            image: Input image (preprocessed, with batch dimension)
            pred_index: Index of the class to visualize (None for predicted class)
            eps: Small value to avoid division by zero
            
        Returns:
            Heatmap array
        """
        # Create a model that maps the input image to the activations
        # of the target layer and the output predictions
        grad_model = keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, 
                    self.model.output]
        )
        
        # Compute the gradient of the top predicted class for the input image
        # with respect to the activations of the target layer
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # For binary classification with sigmoid
            if predictions.shape[-1] == 1:
                class_channel = predictions[:, 0]
            else:
                class_channel = predictions[:, pred_index]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Compute guided gradients (ReLU on gradients)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + eps)
        heatmap = heatmap.numpy()
        
        return heatmap
    
    def overlay_heatmap(self, heatmap, original_image, alpha=HEATMAP_ALPHA, 
                       colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            heatmap: Grad-CAM heatmap
            original_image: Original image (0-255 range or 0-1 range)
            alpha: Transparency of the heatmap overlay
            colormap: OpenCV colormap to use
            
        Returns:
            Superimposed image
        """
        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        
        # Convert original image to proper format
        if original_image.max() <= 1.0:
            original_image = np.uint8(255 * original_image)
        
        # Ensure original image is in RGB
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif original_image.shape[2] == 4:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
        
        # Superimpose the heatmap on original image
        superimposed = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return superimposed
    
    def generate_gradcam(self, image, original_image=None):
        """
        Generate complete Grad-CAM visualization
        
        Args:
            image: Preprocessed image for model input
            original_image: Original image for overlay (if different from input)
            
        Returns:
            Dictionary containing heatmap and superimposed image
        """
        # Compute heatmap
        heatmap = self.compute_heatmap(image)
        
        # Use the input image if original not provided
        if original_image is None:
            original_image = image[0]  # Remove batch dimension
        
        # Overlay heatmap
        superimposed = self.overlay_heatmap(heatmap, original_image)
        
        return {
            'heatmap': heatmap,
            'superimposed': superimposed,
            'original': original_image
        }
    
    def visualize(self, image, original_image=None, prediction=None, 
                 save_path=None, show=True):
        """
        Create a comprehensive visualization with original image,
        heatmap, and overlay
        
        Args:
            image: Preprocessed image for model input
            original_image: Original image for display
            prediction: Prediction dictionary (optional)
            save_path: Path to save the visualization
            show: Whether to display the plot
        """
        # Generate Grad-CAM
        gradcam_results = self.generate_gradcam(image, original_image)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if original_image is None:
            original_image = image[0]
        
        if original_image.max() <= 1.0:
            display_original = original_image
        else:
            display_original = original_image / 255.0
        
        axes[0].imshow(display_original)
        axes[0].set_title('Original X-Ray', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(gradcam_results['heatmap'], cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(gradcam_results['superimposed'])
        title = 'Grad-CAM Overlay'
        if prediction:
            title += f"\n{prediction['class']} ({prediction['confidence']*100:.2f}%)"
        axes[2].set_title(title, fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return gradcam_results


class GradCAMPlusPlus(GradCAM):
    """
    Enhanced version of Grad-CAM with better localization
    """
    
    def compute_heatmap(self, image, pred_index=None, eps=1e-8):
        """
        Compute Grad-CAM++ heatmap
        
        Args:
            image: Input image (preprocessed, with batch dimension)
            pred_index: Index of the class to visualize
            eps: Small value to avoid division by zero
            
        Returns:
            Heatmap array
        """
        grad_model = keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output,
                    self.model.output]
        )
        
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape3:
                    conv_outputs, predictions = grad_model(image)
                    
                    if pred_index is None:
                        pred_index = tf.argmax(predictions[0])
                    
                    if predictions.shape[-1] == 1:
                        class_channel = predictions[:, 0]
                    else:
                        class_channel = predictions[:, pred_index]
                
                # First order gradients
                grads = tape3.gradient(class_channel, conv_outputs)
            
            # Second order gradients
            grads_2 = tape2.gradient(grads, conv_outputs)
        
        # Third order gradients
        grads_3 = tape1.gradient(grads_2, conv_outputs)
        
        # Compute weights
        global_sum = tf.reduce_sum(conv_outputs, axis=(1, 2), keepdims=True)
        
        alpha_denom = grads_2 * 2.0 + grads_3 * global_sum + eps
        alpha = grads_2 / alpha_denom
        
        weights = tf.reduce_sum(
            alpha * tf.nn.relu(grads),
            axis=(1, 2)
        )
        
        # Compute weighted combination
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(weights[:, tf.newaxis, tf.newaxis, :] * conv_outputs, axis=-1)
        
        # Apply ReLU and normalize
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / (tf.math.reduce_max(heatmap) + eps)
        
        return heatmap.numpy()


def create_gradcam_visualization(model, image_path, output_path=None, 
                                 use_gradcam_pp=False):
    """
    Convenience function to create Grad-CAM visualization from image path
    
    Args:
        model: Trained Keras model
        image_path: Path to input image
        output_path: Path to save visualization
        use_gradcam_pp: Whether to use Grad-CAM++ instead of Grad-CAM
        
    Returns:
        Grad-CAM results dictionary
    """
    from data_preprocessing import preprocess_single_image
    
    # Load and preprocess image
    preprocessed_image = preprocess_single_image(image_path)
    
    # Load original image for overlay
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Make prediction
    prediction = model.predict(preprocessed_image, verbose=0)
    pred_class = CLASS_NAMES[1] if prediction[0][0] > 0.5 else CLASS_NAMES[0]
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    
    pred_dict = {
        'class': pred_class,
        'confidence': confidence,
        'probability': prediction[0][0]
    }
    
    # Create Grad-CAM
    if use_gradcam_pp:
        gradcam = GradCAMPlusPlus(model)
    else:
        gradcam = GradCAM(model)
    
    # Generate visualization
    results = gradcam.visualize(
        preprocessed_image,
        original_image,
        prediction=pred_dict,
        save_path=output_path,
        show=False
    )
    
    return results, pred_dict


if __name__ == "__main__":
    print("Grad-CAM Visualization Module")
    print("=" * 50)
    print("\nThis module provides:")
    print("- Grad-CAM implementation")
    print("- Grad-CAM++ implementation")
    print("- Heatmap generation and overlay")
    print("- Comprehensive visualization")
    print("\nUse this module with a trained model for explainable AI.")
