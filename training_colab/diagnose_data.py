"""
Diagnostic script to check data quality and class distribution
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataset_rsna_fixed import RSNAPneumoniaDataset
from config_rsna import *

def diagnose_dataset():
    """Check dataset for issues"""
    print("="*80)
    print("DATASET DIAGNOSTIC")
    print("="*80)
    
    # Load dataset
    dataset = RSNAPneumoniaDataset()
    df = dataset.load_and_prepare_data()
    
    # Check class distribution
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION")
    print("="*80)
    
    class_counts = df['label'].value_counts()
    print(f"\nClass 0 (Normal): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/len(df)*100:.2f}%)")
    print(f"Class 1 (Pneumonia): {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/len(df)*100:.2f}%)")
    
    # Calculate imbalance ratio
    if 0 in class_counts and 1 in class_counts:
        ratio = class_counts[0] / class_counts[1]
        print(f"\nImbalance ratio (Normal/Pneumonia): {ratio:.2f}:1")
        
        if ratio > 2.0:
            print("⚠️  WARNING: Significant class imbalance detected!")
            print("   Recommendation: Use class weights or oversampling")
    
    # Split dataset
    train_df, val_df, test_df = dataset.split_dataset()
    
    # Check splits
    print("\n" + "="*80)
    print("SPLIT DISTRIBUTION")
    print("="*80)
    
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        split_counts = split_df['label'].value_counts()
        print(f"\n{name}:")
        print(f"  Normal: {split_counts.get(0, 0):,} ({split_counts.get(0, 0)/len(split_df)*100:.2f}%)")
        print(f"  Pneumonia: {split_counts.get(1, 0):,} ({split_counts.get(1, 0)/len(split_df)*100:.2f}%)")
    
    # Compute class weights
    class_weights = dataset.compute_class_weights(train_df)
    
    print("\n" + "="*80)
    print("CLASS WEIGHTS")
    print("="*80)
    print(f"\nComputed class weights: {class_weights}")
    print("\nThese weights will be used during training to balance the loss.")
    
    # Check for file existence
    print("\n" + "="*80)
    print("FILE VALIDATION")
    print("="*80)
    
    missing = 0
    for idx, row in df.head(100).iterrows():
        if not os.path.exists(row['image_path']):
            missing += 1
    
    print(f"\nChecked first 100 files: {100-missing} exist, {missing} missing")
    
    if missing > 10:
        print("⚠️  WARNING: Many files are missing!")
    
    # Sample images
    print("\n" + "="*80)
    print("SAMPLE IMAGES")
    print("="*80)
    
    print("\nLoading 5 sample images...")
    for i, (idx, row) in enumerate(df.sample(5).iterrows()):
        try:
            img = dataset.convert_dicom_to_array(row['image_path'])
            label = "Pneumonia" if row['label'] == 1 else "Normal"
            print(f"  {i+1}. {label}: shape={img.shape}, min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
        except Exception as e:
            print(f"  {i+1}. ERROR: {e}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    if ratio > 2.5:
        recommendations.append("✓ Use class weights (already enabled)")
        recommendations.append("✓ Consider oversampling minority class")
    
    if LEARNING_RATE < 1e-4:
        recommendations.append("⚠️  Learning rate might be too low")
        recommendations.append("   Try: LEARNING_RATE = 3e-4")
    
    if DROPOUT_RATE > 0.5:
        recommendations.append("⚠️  Dropout might be too high")
        recommendations.append("   Try: DROPOUT_RATE = 0.3")
    
    if EARLY_STOPPING_PATIENCE < 10:
        recommendations.append("⚠️  Early stopping patience might be too low")
        recommendations.append("   Try: EARLY_STOPPING_PATIENCE = 10")
    
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("✓ Configuration looks good!")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    diagnose_dataset()
