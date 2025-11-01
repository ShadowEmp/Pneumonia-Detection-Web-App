"""
Training Script for RSNA Pneumonia Detection Dataset
26,684 images, properly labeled for pneumonia detection
"""
import os
import json
import numpy as np
from datetime import datetime
import tensorflow as tf

from config_rsna import *
from dataset_rsna_fixed import RSNAPneumoniaDataset


def create_model():
    """Create and compile model"""
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB1
    
    # Load base model
    base_model = EfficientNetB1(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    )
    
    # Freeze base model
    if FREEZE_BASE_LAYERS:
        base_model.trainable = False
    
    # Build model
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    for units in DENSE_LAYERS:
        x = layers.Dense(units, activation=ACTIVATION)(x)
        if USE_BATCH_NORM:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(DROPOUT_RATE)(x)
    
    outputs = layers.Dense(1, activation=OUTPUT_ACTIVATION)(x)
    
    model = models.Model(inputs, outputs)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=LOSS_FUNCTION,
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model


def get_callbacks():
    """Get training callbacks"""
    callbacks = []
    
    # Model checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor=CHECKPOINT_MONITOR,
        mode=CHECKPOINT_MODE,
        save_best_only=SAVE_BEST_ONLY,
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)
    
    # Reduce LR
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # CSV Logger
    if USE_CSV_LOGGER:
        csv_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(LOGS_DIR, 'training_log.csv')
        )
        callbacks.append(csv_logger)
    
    return callbacks


def evaluate_model(model, test_ds, test_df):
    """Evaluate model and compute metrics"""
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\nEvaluating model...")
    
    # Get predictions
    y_pred_proba = model.predict(test_ds, verbose=1)
    y_pred = (y_pred_proba > PREDICTION_THRESHOLD).astype(int).flatten()
    y_true = test_df['label'].values
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred)),
        'auc': float(roc_auc_score(y_true, y_pred_proba))
    }
    
    print("\nTest Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    return metrics


def main():
    """Main training pipeline"""
    print("=" * 80)
    print("RSNA PNEUMONIA DETECTION TRAINING")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: RSNA Pneumonia Detection Challenge")
    print(f"Total images: {TOTAL_IMAGES:,}")
    print(f"Image size: {IMG_HEIGHT}x{IMG_WIDTH}")
    print(f"Base model: {BASE_MODEL}")
    print("=" * 80)
    
    # Set random seeds
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    # Configure GPU
    if USE_MIXED_PRECISION:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("\n✓ Mixed precision enabled")
    
    # Load dataset
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATASET")
    print("=" * 80)
    
    dataset = RSNAPneumoniaDataset()
    df = dataset.load_and_prepare_data()
    
    # Split dataset
    train_df, val_df, test_df = dataset.split_dataset()
    
    # Compute class weights
    class_weights = dataset.compute_class_weights(train_df)
    
    # Create TensorFlow datasets
    print("\nCreating TensorFlow datasets...")
    train_ds = dataset.create_tf_dataset(train_df, batch_size=BATCH_SIZE, shuffle=True, augment=True)
    val_ds = dataset.create_tf_dataset(val_df, batch_size=BATCH_SIZE, shuffle=False, augment=False)
    test_ds = dataset.create_tf_dataset(test_df, batch_size=BATCH_SIZE, shuffle=False, augment=False)
    
    print(f"✓ Datasets created")
    print(f"  Training batches: {len(train_df) // BATCH_SIZE}")
    print(f"  Validation batches: {len(val_df) // BATCH_SIZE}")
    print(f"  Test batches: {len(test_df) // BATCH_SIZE}")
    
    # Build model
    print("\n" + "=" * 80)
    print("STEP 2: BUILDING MODEL")
    print("=" * 80)
    
    model = create_model()
    model.summary()
    
    # Train
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING")
    print("=" * 80)
    
    callbacks = get_callbacks()
    
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weights if USE_CLASS_WEIGHTS else None,
        verbose=1
    )
    
    # Save model
    model.save(MODEL_PATH)
    print(f"\n✓ Model saved to {MODEL_PATH}")
    
    # Evaluate
    print("\n" + "=" * 80)
    print("STEP 4: EVALUATION")
    print("=" * 80)
    
    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    metrics = evaluate_model(best_model, test_ds, test_df)
    
    # Save metrics
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nFinal Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"\nModel: {BEST_MODEL_PATH}")
    print(f"Results: {RESULTS_DIR}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return model, metrics


if __name__ == "__main__":
    try:
        model, metrics = main()
        print("\n✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
