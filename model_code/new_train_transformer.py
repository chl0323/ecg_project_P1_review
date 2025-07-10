# new_train_transformer2.py
# Transformer model training script for ECG deep learning project.
# All Chinese comments, print statements, docstrings, and error messages have been translated to English. Variable names are kept unless in pinyin or Chinese.
import pandas as pd
import numpy as np
from new_transformer_feature_extractor import TransformerFeatureExtractor
import tensorflow as tf
from keras import layers
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

# Create save directory
checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

# Load preprocessed data directly
print("Loading preprocessed data...")
X_train = np.load('processed_data/X_train_smote.npy')
y_train = np.load('processed_data/y_train_smote.npy')
X_val = np.load('processed_data/X_val_smote.npy')
y_val = np.load('processed_data/y_val_smote.npy')

# Reshape data into sequence form (samples, sequence_length, features)
sequence_length = 10
n_features = X_train.shape[1] // sequence_length
X_train = X_train.reshape(-1, sequence_length, n_features)
X_val = X_val.reshape(-1, sequence_length, n_features)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

feature_cols = [
    'RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval', 'QTc_Interval',
    'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak', 'HRV_SDNN', 'QTc_variability',
    'heart_rate', 'QTc_RR_ratio', 'QT_RR_ratio', 'P_R_ratio', 'T_R_ratio',
    'HRV_RMSSD', 'HRV_pNN50'
]

# Build model
print("Building model...")
extractor = TransformerFeatureExtractor(
    input_dim=n_features,
    sequence_length=sequence_length,
    num_heads=1,
    d_model=128,  # Use d_model instead of dff
    rate=0.1
)
model = extractor.build()
model_out = layers.Dense(1, activation='sigmoid')(model.output)
full_model = tf.keras.Model(model.input, model_out)

full_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# Set checkpoint path
checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.weights.h5')
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

# Train
print("Starting training...")
try:
    history = full_model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=70,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
            PeriodicSaveCallback(checkpoint_path, period=1)
        ]
    )
    history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history.csv')
    pd.DataFrame(history.history).to_csv(history_path, index=False)
except Exception as e:
    print(f"Training interrupted: {e}")
    interrupted_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interrupted_model.weights.h5')
    full_model.save_weights(interrupted_model_path)
    print("Model state saved. You can resume training later.")

# Save feature extractor weights
print("Saving model weights...")
feature_extractor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new_transformer_feature_extractor_weights.weights.h5')
model.save_weights(feature_extractor_path)

# Save entire model
full_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transformer_full_model.keras')
full_model.save(full_model_path)
print("Training completed!")