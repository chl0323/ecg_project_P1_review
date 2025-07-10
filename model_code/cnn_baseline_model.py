# cnn_baseline_model.py
# CNN baseline model for ECG classification tasks.
# Implements a traditional CNN architecture with three convolution blocks for local waveform feature extraction.

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os

class CNNBaselineModel:
    """
    CNN baseline model for ECG classification.
    
    Architecture:
    - Three convolution blocks with kernel sizes 5, 3, 3
    - Output channels: 32, 64, 128
    - Global average pooling
    - Two-layer fully connected network (128→64→2)
    - Dropout (p=0.3) between convolution modules
    - Softmax normalization for final classification
    - Total parameters: ~0.83M
    """
    
    def __init__(self, input_shape, num_classes=2, dropout_rate=0.3):
        """
        Initialize CNN baseline model.
        
        Args:
            input_shape: Input data shape (sequence_length, features)
            num_classes: Number of output classes (default: 2 for binary classification)
            dropout_rate: Dropout rate between convolution modules (default: 0.3)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
    def build(self):
        """
        Build the CNN baseline model.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # First convolution block
        x = layers.Conv1D(filters=32, kernel_size=5, padding='same')(inputs)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Second convolution block
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same')(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Third convolution block
        x = layers.Conv1D(filters=128, kernel_size=3, padding='same')(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Global average pooling to compress time dimension
        x = layers.GlobalAveragePooling1D()(x)
        
        # Fully connected layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer with softmax for multi-class or sigmoid for binary
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model

def create_cnn_model(input_shape, num_classes=2, learning_rate=0.001):
    """
    Create and compile CNN baseline model.
    
    Args:
        input_shape: Input data shape (sequence_length, features)
        num_classes: Number of output classes
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    # Create model
    cnn_model = CNNBaselineModel(input_shape, num_classes)
    model = cnn_model.build()
    
    # Compile model
    if num_classes == 2:
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.AUC()]
    else:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    return model

def train_cnn_model(model, X_train, y_train, X_val, y_val, 
                   batch_size=64, epochs=70, patience=8):
    """
    Train the CNN baseline model.
    
    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        patience: Early stopping patience
        
    Returns:
        Training history
    """
    # Create save directory
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set checkpoint path
    checkpoint_path = os.path.join(checkpoint_dir, 'cnn_model_epoch_{epoch:02d}.weights.h5')
    print(f"Checkpoint save path: {checkpoint_dir}")
    
    # Custom callback to save model periodically
    class PeriodicSaveCallback(tf.keras.callbacks.Callback):
        def __init__(self, save_path, period=1):
            super(PeriodicSaveCallback, self).__init__()
            self.save_path = save_path
            self.period = period

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.period == 0:
                self.model.save_weights(self.save_path.format(epoch=epoch + 1))
                print(f"Model saved at epoch {epoch + 1}")
    
    # Train model
    print("Starting CNN baseline model training...")
    try:
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
                PeriodicSaveCallback(checkpoint_path, period=1)
            ]
        )
        
        # Save training history
        history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnn_history.csv')
        import pandas as pd
        pd.DataFrame(history.history).to_csv(history_path, index=False)
        
        # Save model
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnn_baseline_model.keras')
        model.save(model_path)
        print("CNN baseline model training completed!")
        
        return history
        
    except Exception as e:
        print(f"Training interrupted: {e}")
        interrupted_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnn_interrupted_model.weights.h5')
        model.save_weights(interrupted_model_path)
        print("CNN model state saved. You can resume training later.")
        return None

if __name__ == "__main__":
    # Example usage
    print("CNN Baseline Model for ECG Classification")
    print("=" * 50)
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load('processed_data/X_train_smote.npy')
    y_train = np.load('processed_data/y_train_smote.npy')
    X_val = np.load('processed_data/X_val_smote.npy')
    y_val = np.load('processed_data/y_val_smote.npy')
    
    # Reshape data for CNN (samples, sequence_length, features)
    sequence_length = 10
    n_features = X_train.shape[1] // sequence_length
    X_train = X_train.reshape(-1, sequence_length, n_features)
    X_val = X_val.reshape(-1, sequence_length, n_features)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Create and train CNN model
    input_shape = (sequence_length, n_features)
    cnn_model = create_cnn_model(input_shape, num_classes=2, learning_rate=0.001)
    
    # Print model summary
    print("\nCNN Baseline Model Summary:")
    print("=" * 30)
    cnn_model.summary()
    
    # Train the model
    history = train_cnn_model(cnn_model, X_train, y_train, X_val, y_val) 