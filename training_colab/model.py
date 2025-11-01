"""
Model Architecture and Training Utilities
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB1
from config_rsna import *


def create_model():
    """Create and compile model"""
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


class ProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback for better epoch visualization"""
    
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch + 1}/{self.params['epochs']}")
        print(f"{'='*80}")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\n{'─'*80}")
        print(f"📊 EPOCH {epoch + 1} RESULTS:")
        print(f"{'─'*80}")
        
        # Training metrics
        print(f"🔵 Training:")
        print(f"   Loss: {logs.get('loss', 0):.4f}")
        print(f"   Accuracy: {logs.get('accuracy', 0):.4f} ({logs.get('accuracy', 0)*100:.2f}%)")
        print(f"   AUC: {logs.get('auc', 0):.4f}")
        print(f"   Precision: {logs.get('precision', 0):.4f}")
        print(f"   Recall: {logs.get('recall', 0):.4f}")
        
        # Validation metrics
        print(f"\n🟢 Validation:")
        print(f"   Loss: {logs.get('val_loss', 0):.4f}")
        print(f"   Accuracy: {logs.get('val_accuracy', 0):.4f} ({logs.get('val_accuracy', 0)*100:.2f}%)")
        print(f"   AUC: {logs.get('val_auc', 0):.4f}")
        print(f"   Precision: {logs.get('val_precision', 0):.4f}")
        print(f"   Recall: {logs.get('val_recall', 0):.4f}")
        
        # Learning rate
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, 'numpy'):
            lr_value = lr.numpy()
        else:
            lr_value = lr
        print(f"\n⚙️  Learning Rate: {lr_value:.2e}")
        
        # Progress bar
        progress = (epoch + 1) / self.params['epochs']
        bar_length = 50
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\n📈 Progress: [{bar}] {progress*100:.1f}%")
        print(f"{'='*80}\n")


def get_callbacks():
    """Get training callbacks"""
    callbacks = []
    
    # Progress visualization
    progress = ProgressCallback()
    callbacks.append(progress)
    
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
