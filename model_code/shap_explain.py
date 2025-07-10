# shap_explain.py
# SHAP (SHapley Additive exPlanations) model interpretability analysis for ECG deep learning project.
# All code, comments, docstrings, and variable names are fully in English for academic submission.

import shap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Feature names
feature_cols = [
    'RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval', 'QTc_Interval',
    'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak', 'HRV_SDNN', 'heart_rate',
    'QTc_RR_ratio', 'QT_RR_ratio', 'P_R_ratio', 'T_R_ratio',
    'HRV_RMSSD', 'HRV_pNN50'
]

# Load model
print("Loading model...")
model = tf.keras.models.load_model('transformer_full_model.keras', compile=False)

# Load test set data
print("Loading test set data...")
X = np.load('processed_data/X_test_smote.npy')
sequence_length = 10
n_features = X.shape[1] // sequence_length
X = X.reshape(-1, sequence_length, n_features)

# Select background samples and samples to explain
background = X[np.random.choice(X.shape[0], 100, replace=False)]
explain_samples = X[:10]

# Create SHAP explainer
print("Creating SHAP explainer...")
explainer = shap.DeepExplainer(model, background)

# Calculate SHAP values
print("Calculating SHAP values...")
shap_values = explainer.shap_values(explain_samples)

# Average over time steps to get mean SHAP value for each feature
shap_mean = np.mean(shap_values[0], axis=1)  # shape: (number of samples, number of features)

# Visualize
print("Plotting SHAP summary plot...")
shap.summary_plot(shap_mean, feature_names=feature_cols)
plt.show() 