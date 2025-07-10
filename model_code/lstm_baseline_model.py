# lstm_baseline_model.py
# LSTM baseline model for ECG classification tasks.
# Implements bidirectional LSTM with attention mechanism for temporal dependency modeling.

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os

class AttentionLayer(layers.Layer):
    """
    Additive attention mechanism for temporal feature fusion.
    """
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.units = units
        self.W = layers.Dense(units, kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.V = layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.001))
        
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        
        # Calculate attention weights
        score = self.V(tf.nn.tanh(self.W(inputs)))  # (batch_size, time_steps, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch_size, time_steps, 1)
        
        # Apply attention weights
        context_vector = attention_weights * inputs  # (batch_size, time_steps, features)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch_size, features)
        
        return context_vector, attention_weights

class LSTMBaselineModel:
    """
    LSTM baseline model for ECG classification.
    
    Architecture:
    - Two stacked bidirectional LSTM layers (64 hidden units each)
    - Additive attention mechanism for temporal fusion
    - Fully connected layer (64→2)
    - Gradient clipping (threshold=5)
    - Balanced regularization to prevent overfitting while maintaining performance
    - Total parameters: ~1.12M
    """
    
    def __init__(self, input_shape, num_classes=2, lstm_units=64, max_sequence_length=10, dropout_rate=0.3):
        """
        Initialize LSTM baseline model.
        
        Args:
            input_shape: Input data shape (sequence_length, features)
            num_classes: Number of output classes (default: 2 for binary classification)
            lstm_units: Number of hidden units in LSTM layers (default: 64)
            max_sequence_length: Maximum sequence length for truncation (default: 10)
            dropout_rate: Dropout rate for regularization (default: 0.3, balanced)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.max_sequence_length = max_sequence_length
        self.dropout_rate = dropout_rate
        
    def build(self):
        """
        Build the LSTM baseline model.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Truncate sequence length if needed
        if self.input_shape[0] > self.max_sequence_length:
            inputs = layers.Lambda(lambda x: x[:, :self.max_sequence_length, :])(inputs)
            print(f"Input sequence truncated to {self.max_sequence_length} time steps")
        
        # First bidirectional LSTM layer with balanced dropout
        lstm1_forward = layers.LSTM(self.lstm_units, return_sequences=True, 
                                   recurrent_dropout=0.1,  # balanced dropout
                                   dropout=0.1,  # balanced dropout
                                   kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
        lstm1_backward = layers.LSTM(self.lstm_units, return_sequences=True, 
                                    recurrent_dropout=0.1,  # balanced dropout
                                    dropout=0.1,  # balanced dropout
                                    go_backwards=True,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
        lstm1_combined = layers.Concatenate()([lstm1_forward, lstm1_backward])
        
        # Second bidirectional LSTM layer with balanced dropout
        lstm2_forward = layers.LSTM(self.lstm_units, return_sequences=True, 
                                   recurrent_dropout=0.1,  # balanced dropout
                                   dropout=0.1,  # balanced dropout
                                   kernel_regularizer=tf.keras.regularizers.l2(0.001))(lstm1_combined)
        lstm2_backward = layers.LSTM(self.lstm_units, return_sequences=True, 
                                    recurrent_dropout=0.1,  # balanced dropout
                                    dropout=0.1,  # balanced dropout
                                    go_backwards=True,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001))(lstm1_combined)
        lstm2_combined = layers.Concatenate()([lstm2_forward, lstm2_backward])
        
        # Additive attention mechanism
        attention_layer = AttentionLayer(self.lstm_units * 2)
        context_vector, attention_weights = attention_layer(lstm2_combined)
        
        # Dropout for regularization (balanced rate)
        x = layers.Dropout(self.dropout_rate)(context_vector)
        
        # Fully connected layer with balanced L2 regularization
        x = layers.Dense(64, activation='relu',  # restored to 64
                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model

def create_lstm_model(input_shape, num_classes=2, learning_rate=0.001, clip_norm=5.0):
    """
    Create and compile LSTM baseline model.
    
    Args:
        input_shape: Input data shape (sequence_length, features)
        num_classes: Number of output classes
        learning_rate: Learning rate for optimizer (restored to 0.001)
        clip_norm: Gradient clipping threshold
        
    Returns:
        Compiled Keras model
    """
    # Create model
    lstm_model = LSTMBaselineModel(input_shape, num_classes)
    model = lstm_model.build()
    
    # Compile model with gradient clipping
    if num_classes == 2:
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.AUC()]
    else:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clip_norm)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model

def train_lstm_model(model, X_train, y_train, X_val, y_val, 
                    batch_size=32, epochs=70, patience=8):
    """
    Train the LSTM baseline model.
    
    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        batch_size: Batch size for training (smaller for LSTM)
        epochs: Maximum number of epochs
        patience: Early stopping patience (restored to 8)
        
    Returns:
        Training history
    """
    # Create save directory
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set checkpoint path
    checkpoint_path = os.path.join(checkpoint_dir, 'lstm_model_epoch_{epoch:02d}.weights.h5')
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
    print("Starting LSTM baseline model training with balanced regularization...")
    try:
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),  # restored patience to 3
                PeriodicSaveCallback(checkpoint_path, period=1)
            ]
        )
        
        # Save training history
        history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lstm_history.csv')
        import pandas as pd
        pd.DataFrame(history.history).to_csv(history_path, index=False)
        
        # Save model
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lstm_baseline_model.keras')
        model.save(model_path)
        print("LSTM baseline model training completed!")
        
        return history
        
    except Exception as e:
        print(f"Training interrupted: {e}")
        interrupted_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lstm_interrupted_model.weights.h5')
        model.save_weights(interrupted_model_path)
        print("LSTM model state saved. You can resume training later.")
        return None

if __name__ == "__main__":
    # Example usage
    print("LSTM Baseline Model for ECG Classification (Balanced Regularization)")
    print("=" * 65)
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load('processed_data/X_train_smote.npy')
    y_train = np.load('processed_data/y_train_smote.npy')
    X_val = np.load('processed_data/X_val_smote.npy')
    y_val = np.load('processed_data/y_val_smote.npy')
    
    # Reshape data for LSTM (samples, sequence_length, features)
    sequence_length = 10
    n_features = X_train.shape[1] // sequence_length
    X_train = X_train.reshape(-1, sequence_length, n_features)
    X_val = X_val.reshape(-1, sequence_length, n_features)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Create and train LSTM model with balanced regularization
    input_shape = (sequence_length, n_features)
    lstm_model = create_lstm_model(input_shape, num_classes=2, learning_rate=0.001, clip_norm=5.0)
    
    # Print model summary
    print("\nLSTM Baseline Model Summary (Balanced Regularization):")
    print("=" * 50)
    lstm_model.summary()
    
    # Train the model
    history = train_lstm_model(lstm_model, X_train, y_train, X_val, y_val, batch_size=32) 