"""
Single Image Prediction Script
Quick utility to test the model on a single image
"""
import sys
import os
import argparse
from tensorflow import keras
import matplotlib.pyplot as plt

from config import *
from data_preprocessing import preprocess_single_image
from gradcam import create_gradcam_visualization


def predict_single_image(image_path, model_path=BEST_MODEL_PATH, show_gradcam=True):
    """
    Predict pneumonia from a single X-ray image
    
    Args:
        image_path: Path to the X-ray image
        model_path: Path to the trained model
        show_gradcam: Whether to show Grad-CAM visualization
    """
    print("\n" + "="*60)
    print("PNEUMONIA DETECTION - SINGLE IMAGE PREDICTION")
    print("="*60)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    # Preprocess image
    print(f"\nProcessing image: {image_path}")
    preprocessed_image = preprocess_single_image(image_path)
    
    # Make prediction
    print("\nMaking prediction...")
    prediction = model.predict(preprocessed_image, verbose=0)
    probability = prediction[0][0]
    
    predicted_class = CLASS_NAMES[1] if probability > 0.5 else CLASS_NAMES[0]
    confidence = probability if probability > 0.5 else 1 - probability
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Probability (Pneumonia): {probability*100:.2f}%")
    print(f"Probability (Normal): {(1-probability)*100:.2f}%")
    print("="*60)
    
    # Generate Grad-CAM if requested
    if show_gradcam:
        print("\nGenerating Grad-CAM visualization...")
        
        output_path = os.path.join(RESULTS_DIR, f'gradcam_{os.path.basename(image_path)}')
        
        results, pred_dict = create_gradcam_visualization(
            model, image_path, output_path=output_path
        )
        
        print(f"Grad-CAM visualization saved to {output_path}")
        
        # Show visualization
        plt.show()
    
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'probability': float(probability)
    }


def main():
    """
    Main function for command-line usage
    """
    parser = argparse.ArgumentParser(
        description='Predict pneumonia from a chest X-ray image'
    )
    
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the chest X-ray image'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=BEST_MODEL_PATH,
        help='Path to the trained model (default: best_pneumonia_model.h5)'
    )
    
    parser.add_argument(
        '--no-gradcam',
        action='store_true',
        help='Disable Grad-CAM visualization'
    )
    
    args = parser.parse_args()
    
    # Run prediction
    predict_single_image(
        args.image_path,
        model_path=args.model,
        show_gradcam=not args.no_gradcam
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, show example usage
        print("\nPneumonia Detection - Single Image Prediction")
        print("=" * 60)
        print("\nUsage:")
        print("  python predict_single.py <image_path> [options]")
        print("\nExamples:")
        print("  python predict_single.py data/test/Normal/image1.jpeg")
        print("  python predict_single.py data/test/Pneumonia/image2.jpeg --no-gradcam")
        print("  python predict_single.py my_xray.jpg --model models/pneumonia_model.h5")
        print("\nOptions:")
        print("  --model PATH       Path to trained model")
        print("  --no-gradcam       Disable Grad-CAM visualization")
        print()
    else:
        main()
