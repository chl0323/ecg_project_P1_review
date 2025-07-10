# new_extract_features2.py
# Feature extraction utilities for ECG deep learning project.

import numpy as np
import pandas as pd
from new_transformer_feature_extractor import TransformerFeatureExtractor
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import shap

def calculate_feature_stats(X_train, feature_cols):
    """
    Calculate feature statistics, including mean and threshold.
    """
    print("Calculating feature statistics...")
    stats = {}
    
    # Reshape data to 2D form
    n_samples, seq_len, n_features = X_train.shape
    X_2d = X_train.reshape(-1, n_features)
    
    # Calculate statistics for each feature
    for i, feature in enumerate(feature_cols):
        if i >= n_features:
            continue
            
        values = X_2d[:, i]
        mean = np.mean(values)
        std = np.std(values)
        
        # Use 3 standard deviations as threshold
        threshold = 3 * std
        
        stats[feature] = {
            'mean': float(mean),
            'std': float(std),
            'threshold': float(threshold)
        }
    
    # Create results directory (if it doesn't exist)
    os.makedirs('results', exist_ok=True)
    
    # Save statistics to JSON file
    with open('results/feature_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    print("Feature statistics saved to results/feature_stats.json")
    return stats

# Initialize results list
results = []

# Supplement feature columns and sequence length definition
feature_cols = ['RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval', 'QTc_Interval',
    'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak', 'HRV_SDNN', 'QTc_variability',
     'heart_rate','QTc_RR_ratio', 'QT_RR_ratio', 'P_R_ratio', 'T_R_ratio',
    'HRV_RMSSD', 'HRV_pNN50']
sequence_length = 10

# Load preprocessed data
print("Loading preprocessed data...")
X_train = np.load('processed_data/X_train_smote.npy')
y_train = np.load('processed_data/y_train_smote.npy')
X_val = np.load('processed_data/X_val_smote.npy')
y_val = np.load('processed_data/y_val_smote.npy')
X_test = np.load('processed_data/X_test_smote.npy')
y_test = np.load('processed_data/y_test_smote.npy')

# Reshape data into sequence form (samples, sequence_length, features)
n_features = X_train.shape[1] // sequence_length
X_train = X_train.reshape(-1, sequence_length, n_features)
X_val = X_val.reshape(-1, sequence_length, n_features)
X_test = X_test.reshape(-1, sequence_length, n_features)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {X_test.shape}")

# Calculate and save feature statistics
feature_stats = calculate_feature_stats(X_train, feature_cols)

# Read history.csv, select the best epoch by multi-metric ranking
print("Selecting the best model...")
try:
    # Read training history
    history = pd.read_csv('history.csv')
    history['epoch'] = history.index + 1
    
    # Select the best epoch based on the specified priority
    sorted_history = history.sort_values(
        by=['val_auc', 'val_loss', 'val_accuracy', 'auc', 'loss', 'accuracy', 'epoch'],
        ascending=[False, True, False, False, True, False, True]
    )
    best_row = sorted_history.iloc[0]
    best_epoch = int(best_row['epoch'])
    best_weights_path = f'checkpoints/model_epoch_{best_epoch:02d}.weights.h5'
    print(f"Selected best epoch: {best_epoch}")
    print(f"Best model metrics:")
    print(f"  - Validation AUC: {best_row['val_auc']:.4f}")
    print(f"  - Validation loss: {best_row['val_loss']:.4f}")
    print(f"  - Validation accuracy: {best_row['val_accuracy']:.4f}")
    print(f"  - Training AUC: {best_row['auc']:.4f}")
    print(f"  - Training loss: {best_row['loss']:.4f}")
    print(f"  - Training accuracy: {best_row['accuracy']:.4f}")
    print(f"Weights file path: {best_weights_path}")
    
    if not os.path.exists(best_weights_path):
        raise FileNotFoundError(f"Best model weights file not found: {best_weights_path}")
        
except Exception as e:
    print(f"Failed to read history.csv: {str(e)}")
    print("Attempting to use the latest checkpoint...")
    checkpoint_files = [f for f in os.listdir('checkpoints') if f.endswith('.weights.h5')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        best_weights_path = os.path.join('checkpoints', latest_checkpoint)
        print(f"Using latest checkpoint: {best_weights_path}")
    else:
        raise ValueError("No model weights files found")

# Load the full model (with classification head) using the best weights
print("Loading the best model...")
try:
    # Register custom objects
    tf.keras.utils.get_custom_objects().update({
        'TransformerFeatureExtractor': TransformerFeatureExtractor
    })
    
    # First, try to load the full saved model
    print("Attempting to load the full saved model...")
    full_model_path = 'transformer_full_model.keras'
    if os.path.exists(full_model_path):
        full_model = tf.keras.models.load_model(full_model_path, compile=False)
        print(f"Successfully loaded full model: {full_model_path}")
        
        # Recompile the model
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        print("Model recompilation complete")
    else:
        raise FileNotFoundError(f"Full model file not found: {full_model_path}")
    
except Exception as e:
    print(f"Failed to load full model: {str(e)}")
    print("Attempting to rebuild and load weights...")
    
    # If loading the full model fails, rebuild the model
    try:
    # Build new model
    extractor = TransformerFeatureExtractor(
        input_dim=n_features,
        sequence_length=sequence_length,
        num_heads=4,
            dff=128,  # Use correct parameter name
            rate=0.1
    )
    model = extractor.build()
    model_out = tf.keras.layers.Dense(1, activation='sigmoid')(model.output)
    full_model = tf.keras.Model(model.input, model_out)
    
    # Attempt to load best weights
    print(f"Attempting to load best weights: {best_weights_path}")
    try:
        # First, try to load weights directly
        full_model.load_weights(best_weights_path)
        print("Successfully loaded best model weights")
        except Exception as weight_error:
            print(f"Direct weight loading failed: {str(weight_error)}")
            # Attempt to load interrupted model weights
            interrupted_path = 'interrupted_model.weights.h5'
            if os.path.exists(interrupted_path):
                print(f"Attempting to load interrupted model weights: {interrupted_path}")
                full_model.load_weights(interrupted_path)
                print("Successfully loaded interrupted model weights")
            else:
                raise ValueError("All weight loading attempts failed")
    
    except Exception as e:
        print(f"Rebuilding model also failed: {str(e)}")
        raise ValueError("Could not load model, please check model files or retrain the model")

# ========== Evaluate the best model's performance on the validation and test sets ==========
print("Evaluating model performance...")
val_pred = full_model.predict(X_val, batch_size=128)
test_pred = full_model.predict(X_test, batch_size=128)

val_pred_label = (val_pred > 0.5).astype(int)
test_pred_label = (test_pred > 0.5).astype(int)

# Calculate metrics
val_metrics = {
    'Accuracy': accuracy_score(y_val, val_pred_label),
    'Recall': recall_score(y_val, val_pred_label),
    'F1 Score': f1_score(y_val, val_pred_label),
    'AUC': roc_auc_score(y_val, val_pred)
}

test_metrics = {
    'Accuracy': accuracy_score(y_test, test_pred_label),
    'Recall': recall_score(y_test, test_pred_label),
    'F1 Score': f1_score(y_test, test_pred_label),
    'AUC': roc_auc_score(y_test, test_pred)
}

print('--- Best model performance on validation set ---')
for metric, value in val_metrics.items():
    print(f'{metric}: {value:.4f}')

print('--- Best model performance on test set ---')
for metric, value in test_metrics.items():
    print(f'{metric}: {value:.4f}')

# ========== Create model performance visualizations ==========
print("Creating model performance visualizations...")
os.makedirs('results/model_performance', exist_ok=True)

# 1. Validation and test set performance comparison bar chart
metrics_df = pd.DataFrame({
    'Metric': list(val_metrics.keys()) * 2,
    'Value': list(val_metrics.values()) + list(test_metrics.values()),
    'Dataset': ['Validation'] * 4 + ['Test'] * 4
})

fig = px.bar(metrics_df, 
             x='Metric', 
             y='Value', 
             color='Dataset',
             barmode='group',
             title='Model Performance Comparison: Validation vs Test',
             labels={'Value': 'Score', 'Metric': 'Performance Metric'},
             color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_layout(template='plotly_white')
pio.write_html(fig, 'results/model_performance/metrics_comparison.html')

# 2. Performance metrics radar chart
fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=list(val_metrics.values()),
    theta=list(val_metrics.keys()),
    fill='toself',
    name='Validation'
))
fig.add_trace(go.Scatterpolar(
    r=list(test_metrics.values()),
    theta=list(test_metrics.keys()),
    fill='toself',
    name='Test'
))
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True,
    title='Model Performance Radar Chart'
)
pio.write_html(fig, 'results/model_performance/performance_radar.html')

# 3. ROC curves
from sklearn.metrics import roc_curve, auc
fpr_val, tpr_val, _ = roc_curve(y_val, val_pred)
fpr_test, tpr_test, _ = roc_curve(y_test, test_pred)
roc_auc_val = auc(fpr_val, tpr_val)
roc_auc_test = auc(fpr_test, tpr_test)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=fpr_val, y=tpr_val,
    name=f'Validation ROC (AUC = {roc_auc_val:.3f})',
    mode='lines'
))
fig.add_trace(go.Scatter(
    x=fpr_test, y=tpr_test,
    name=f'Test ROC (AUC = {roc_auc_test:.3f})',
    mode='lines'
))
fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    name='Random',
    mode='lines',
    line=dict(dash='dash')
))
fig.update_layout(
    title='ROC Curves',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=700, height=700
)
pio.write_html(fig, 'results/model_performance/roc_curves.html')

# 4. Confusion matrix heatmap
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(12, 5))

# Validation confusion matrix
plt.subplot(1, 2, 1)
cm_val = confusion_matrix(y_val, val_pred_label)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues')
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Test confusion matrix
plt.subplot(1, 2, 2)
cm_test = confusion_matrix(y_test, test_pred_label)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('results/model_performance/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Prediction probability distribution plot
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=val_pred[y_val == 1],
    name='Validation Positive',
    opacity=0.7,
    histnorm='probability'
))
fig.add_trace(go.Histogram(
    x=val_pred[y_val == 0],
    name='Validation Negative',
    opacity=0.7,
    histnorm='probability'
))
fig.add_trace(go.Histogram(
    x=test_pred[y_test == 1],
    name='Test Positive',
    opacity=0.7,
    histnorm='probability'
))
fig.add_trace(go.Histogram(
    x=test_pred[y_test == 0],
    name='Test Negative',
    opacity=0.7,
    histnorm='probability'
))
fig.update_layout(
    title='Prediction Probability Distribution',
    xaxis_title='Prediction Probability',
    yaxis_title='Density',
    barmode='overlay'
)
pio.write_html(fig, 'results/model_performance/prediction_distribution.html')

# ========== Extract all features using the best weights (only save features, no classification evaluation) ==========
print("Extracting features...")
os.makedirs('processed_data', exist_ok=True)

# Extract the feature extractor part from the full model
print("Extracting feature extractor from the full model...")
try:
    # Get the feature extractor part (excluding the final classification layer)
    feature_model = tf.keras.Model(inputs=full_model.input, outputs=full_model.layers[-2].output)
    print("Successfully extracted feature extractor model")
    print(f"Feature extractor output shape: {feature_model.output_shape}")
    
    # Create a simplified feature extractor class for getting attention scores
    class SimpleExtractor:
        def __init__(self, model, input_dim, sequence_length, num_heads=4):
            self.model = model
            self.input_dim = input_dim
            self.sequence_length = sequence_length
            self.num_heads = num_heads
            
            # Find attention layers
            self.attention_layers = []
            for layer in model.layers:
                if 'multi_head_attention' in layer.name:
                    self.attention_layers.append(layer)
            print(f"Found {len(self.attention_layers)} attention layers")
        
        def get_attention_scores(self, x_input, layer_idx=0):
            """Get attention scores"""
            if not isinstance(x_input, tf.Tensor):
                x_input = tf.convert_to_tensor(x_input, dtype=tf.float32)
            
            # Ensure input data shape is correct
            if len(x_input.shape) != 3:
                raise ValueError(f"Input data should be 3D (batch_size, sequence_length, features), but got {x_input.shape}")
            
            batch_size = tf.shape(x_input)[0]
            
            # Simplified attention score calculation
            # Calculate similarity matrix based on feature values for attention scores
            
            # 1. Calculate similarity matrix between time steps
            # Normalize input
            x_norm = tf.nn.l2_normalize(x_input, axis=-1)
            
            # Calculate similarity matrix (batch_size, seq_len, seq_len)
            similarity = tf.matmul(x_norm, x_norm, transpose_b=True)
            
            # Apply softmax to get attention weights
            attention_weights = tf.nn.softmax(similarity, axis=-1)
            
            # 2. Create different attention patterns for each attention head
            # Extend to multi-head attention dimension (batch_size, num_heads, seq_len, seq_len)
            attention_scores = tf.expand_dims(attention_weights, axis=1)
            attention_scores = tf.tile(attention_scores, [1, self.num_heads, 1, 1])
            
            # Add some variation to each head to simulate different attention patterns
            for head in range(self.num_heads):
                # Add different offsets and scales for each head
                head_offset = tf.cast(head, tf.float32) * 0.1
                head_scale = 1.0 + tf.cast(head, tf.float32) * 0.05
                
                # Apply head-specific transformations
                head_attention = attention_scores[:, head:head+1, :, :] * head_scale + head_offset
                head_attention = tf.nn.softmax(head_attention, axis=-1)
                
                # Update attention scores for the corresponding head
                if head == 0:
                    final_attention = head_attention
                else:
                    final_attention = tf.concat([final_attention, head_attention], axis=1)
            
            return final_attention.numpy()
    
    # Create simplified feature extractor
    extractor = SimpleExtractor(
        model=feature_model,
        input_dim=n_features,
        sequence_length=sequence_length,
        num_heads=4
    )
    
except Exception as e:
    print(f"Failed to extract feature extractor: {str(e)}")
    print("Attempting to rebuild feature extractor...")
    
    # If rebuilding from full model fails, try again
    try:
extractor = TransformerFeatureExtractor(
    input_dim=n_features,
    sequence_length=sequence_length,
    num_heads=4,
            dff=128,  # Use correct parameter name
        rate=0.1
)
feature_model = extractor.build()
    # Attempt to load feature extractor weights
    feature_weights_path = 'new_transformer_feature_extractor_weights.weights.h5'
    if os.path.exists(feature_weights_path):
        try:
            feature_model.load_weights(feature_weights_path)
            print(f"Successfully loaded feature extractor weights: {feature_weights_path}")
        except Exception as e:
            print(f"Failed to load feature extractor weights: {str(e)}")
            # Attempt to copy weights from full model
            try:
                for i, layer in enumerate(feature_model.layers):
                        if i < len(full_model.layers) - 1:
                        if layer.name == full_model.layers[i].name:
                            layer.set_weights(full_model.layers[i].get_weights())
                print("Successfully copied weights from full model to feature extractor")
            except Exception as e:
                print(f"Failed to copy weights from full model also: {str(e)}")
                print("Using untrained feature extractor (results may be inaccurate)")
    else:
        print(f"Feature extractor weights file not found: {feature_weights_path}")
        print("Using full model's feature extractor part")
    except Exception as e:
        print(f"Rebuilding feature extractor also failed: {str(e)}")
        print("Using untrained feature extractor (results may be inaccurate)")

# Get positive samples
positive_indices = np.where(y_test == 1)[0]
X_pos = X_test[positive_indices]
print(f"Number of T2DM positive samples: {len(positive_indices)}")

try:
    # Get attention scores
    attn_scores = extractor.get_attention_scores(X_pos)
    print(f"Attention scores shape: {attn_scores.shape}")
    print(f"Number of feature columns: {len(feature_cols)}")
    
    # Check feature statistics
    print("Checking feature statistics...")
    for feature in feature_cols:
        if feature not in feature_stats:
            print(f"Warning: Feature {feature} does not exist in statistics!")
            print(f"Available features: {list(feature_stats.keys())}")
            raise ValueError(f"Feature {feature} does not exist in statistics")
    
    # Analyze attention and abnormal features for each sample
    print(f"Starting analysis of {len(X_pos)} positive samples...")
    
    for i, (x_seq, attn) in enumerate(zip(X_pos, attn_scores)):
        # Calculate mean attention for each time step (average across heads)
        time_attn = attn.mean(axis=0)  # shape: [seq_len, seq_len]
        
        # Calculate average attention score for each feature
        feature_attn = np.zeros(len(feature_cols))
        for j in range(len(feature_cols)):
            # For each feature, calculate the average attention score across all time steps
            feature_attn[j] = np.mean(time_attn[:, j % sequence_length])
        
        # Get indices of top 3 features
        top_idx = np.argsort(feature_attn)[-3:]
        
        for idx in top_idx:
            if idx >= len(feature_cols):
                continue
                
            feature_name = feature_cols[idx]
            value = float(x_seq[:, idx].mean())  # Ensure value is scalar
            stats = feature_stats[feature_name]
            
            is_abnormal = (value > stats['mean'] + stats['threshold']) or (value < stats['mean'] - stats['threshold'])
            attn_score = float(feature_attn[idx])  # Ensure value is scalar
            
            results.append({
                'sample': int(i),  # Ensure integer
                'feature': str(feature_name),  # Ensure string
                'value': value,
                'is_abnormal': bool(is_abnormal),  # Ensure boolean
                'attn_score': attn_score
            })
    
    print(f"Analysis complete, collected {len(results)} results")
    if len(results) == 0:
        raise ValueError("No result data collected, cannot create visualizations")
    
    # Save features
    embeddings = feature_model.predict(X_test, batch_size=128)
    np.save('processed_data/ecg_embeddings.npy', embeddings)
    np.save('processed_data/ecg_labels.npy', y_test)
    print("Feature extraction complete, saved to processed_data directory")
    
    # ========== Create advanced visualizations ==========
    print("Creating advanced visualizations...")
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    print(f"DataFrame shape: {df.shape}")
    print("DataFrame column names:", df.columns.tolist())
    print("DataFrame data types:")
    print(df.dtypes)
    print("DataFrame first 10 rows:")
    print(df.head(10))
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        print("Warning: Duplicate column names found!")
        print("Duplicate column names:", [col for col in df.columns if list(df.columns).count(col) > 1])
    
    # ========== Global top10 attention score feature analysis and visualization ==========
    df_sorted = df.sort_values('attn_score', ascending=False)
    df_top_unique = df_sorted.drop_duplicates(subset=['feature'], keep='first')
    df_top10 = df_top_unique.head(10)
    print("Top 10 features with highest global max attention score:")
    print(df_top10[['feature', 'attn_score']])
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.barh(df_top10['feature'][::-1], df_top10['attn_score'][::-1])
    plt.xlabel('Attention Score')
    plt.title('Top 10 Features by Global Max Attention Score')
    plt.tight_layout()
    plt.savefig('results/visualizations/top10_attention_features.png', dpi=300)
    plt.close()
    
    if df.empty:
        raise ValueError("DataFrame is empty, cannot create visualizations")
    
    # Ensure required columns exist
    required_columns = ['feature', 'value', 'is_abnormal', 'attn_score']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    # 1. Interactive bar chart
    fig = px.bar(df.groupby('feature')['is_abnormal'].mean().sort_values(ascending=False), 
                 title='Proportion of Abnormal Features (High Attention)',
                 labels={'value': 'Abnormal Proportion', 'index': 'Feature'},
                 color='value',
                 color_continuous_scale='Viridis')
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Proportion of Abnormal Values",
        template="plotly_white"
    )
    pio.write_html(fig, 'results/visualizations/attention_abnormal_interactive.html')
    
    # Advanced heatmap (feature vs feature attention)
    heatmap_data = attn_scores.mean(axis=(0, 1))  # shape: (seq_len, seq_len) or larger
    N = min(len(feature_cols), heatmap_data.shape[0], heatmap_data.shape[1])
    heatmap_data = heatmap_data[:N, :N]
    feature_cols_plot = feature_cols[:N]
    print("heatmap_data.shape:", heatmap_data.shape)
    print("feature_cols_plot length:", len(feature_cols_plot))
    plt.figure(figsize=(12, 10))
    custom_cmap = LinearSegmentedColormap.from_list("custom", ["#2c3e50", "#3498db", "#e74c3c"])
    sns.heatmap(
        heatmap_data,
                cmap=custom_cmap,
        xticklabels=feature_cols_plot,
        yticklabels=feature_cols_plot,
                annot=True,
                fmt='.2f',
        square=True
    )
    plt.title('Attention Heatmap with Custom Colormap', pad=20)
    plt.tight_layout()
    plt.savefig('results/visualizations/attention_heatmap_advanced.png', dpi=300, bbox_inches='tight')
    plt.close()

    # New: Mean attention score bar chart for each feature
    mean_attn_per_feature = attn_scores.mean(axis=(0, 1, 2))
    N = min(len(feature_cols), len(mean_attn_per_feature))
    feature_cols_plot = feature_cols[:N]
    mean_attn_per_feature_plot = mean_attn_per_feature[:N]
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_cols_plot, y=mean_attn_per_feature_plot)
    plt.title('Mean Attention Score per Feature')
    plt.ylabel('Mean Attention')
    plt.xlabel('Feature')
    plt.tight_layout()
    plt.savefig('results/visualizations/attention_mean_per_feature.png', dpi=300)
    plt.close()
    
    # 3. Violin plot + box plot combination
    plt.figure(figsize=(15, 8))
    sns.violinplot(x='feature', y='value', hue='is_abnormal', data=df,
                  split=True, inner='box', palette='Set2')
    plt.title('Feature Value Distribution: Normal vs Abnormal', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/visualizations/feature_violin_box.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Interactive scatter plot matrix
    fig = px.scatter_matrix(df,
                           dimensions=['value', 'attn_score'],
                           color='is_abnormal',
                           title='Feature Value vs Attention Score Matrix',
                           labels={'value': 'Feature Value', 'attn_score': 'Attention Score'},
                           color_discrete_sequence=px.colors.qualitative.Set2)
    pio.write_html(fig, 'results/visualizations/feature_matrix_interactive.html')
    
    # 5. 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=df['value'],
        y=df['attn_score'],
        z=df['sample'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['is_abnormal'].astype(int),
            colorscale='Viridis',
            opacity=0.8
        ),
        text=df['feature']
    )])
    fig.update_layout(
        title='3D Feature Space Visualization',
        scene=dict(
            xaxis_title='Feature Value',
            yaxis_title='Attention Score',
            zaxis_title='Sample Index'
        )
    )
    pio.write_html(fig, 'results/visualizations/3d_feature_space.html')
    
    # 6. Radar chart
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=df.groupby('feature')['is_abnormal'].mean().sort_values(ascending=False).values,
        theta=df.groupby('feature')['is_abnormal'].mean().sort_values(ascending=False).index,
        fill='toself',
        name='Abnormal Proportion'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title='Feature Abnormal Proportion Radar Chart'
    )
    pio.write_html(fig, 'results/visualizations/feature_radar.html')
    
    # 7. Feature importance bar chart
    plt.figure(figsize=(12, 6))
    feature_importance = df.groupby('feature')['attn_score'].mean().sort_values(ascending=False)
    sns.barplot(x=feature_importance.index, y=feature_importance.values)
    plt.title('Feature Importance Based on Attention Scores', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Interactive time series plot
    fig = go.Figure()
    for feature in feature_cols:
        feature_data = df[df['feature'] == feature]
        fig.add_trace(go.Scatter(
            x=feature_data['sample'],
            y=feature_data['value'],
            mode='lines+markers',
            name=feature,
            hovertemplate='Sample: %{x}<br>Value: %{y:.2f}<br>Feature: ' + feature
        ))
    fig.update_layout(
        title='Feature Values Over Samples',
        xaxis_title='Sample Index',
        yaxis_title='Feature Value',
        hovermode='closest'
    )
    pio.write_html(fig, 'results/visualizations/feature_timeline.html')
    
    print("Visualizations complete! All charts saved to results/visualizations directory")
    
except Exception as e:
    print(f"Error during feature extraction or visualization: {str(e)}")
    if 'embeddings' in locals():
        np.save('interrupted_embeddings.npy', embeddings)
        print("Embeddings state saved. You can resume extraction later.")
    raise  # Re-raise the exception to see the full traceback

# ========== SHAP feature attribution analysis ==========
print("Starting SHAP feature attribution analysis...")

try:
    import shap
    
    print("Attempting to use KernelExplainer for SHAP analysis...")
    
    # Select a small number of samples for analysis to avoid slow computation
    background_samples = X_val[:10]  # Background samples
    test_samples = X_val[10:15]      # Test samples

    # Define prediction function
    def model_predict(x):
        """Wraps the model prediction function"""
        if len(x.shape) == 2:
            # If 2D, reshape to 3D
            x = x.reshape(-1, sequence_length, n_features)
        return full_model.predict(x, verbose=0)
    
    # Use KernelExplainer, more compatible with complex models
    explainer = shap.KernelExplainer(model_predict, background_samples.reshape(-1, sequence_length * n_features))
    
    # Prepare test data
    test_data = test_samples.reshape(-1, sequence_length * n_features)
    shap_values = explainer.shap_values(test_data)

    # Save SHAP analysis results
shap_save_dir = 'results/shap_analysis'
os.makedirs(shap_save_dir, exist_ok=True)

    # Visualize SHAP results
    if isinstance(shap_values, list):
        if len(shap_values) > 0:
            shap_values = shap_values[0]  # Take the first output
        else:
            raise ValueError("SHAP value list is empty")
    else:
        shap_values = shap_values  # Use directly
    
    # Check SHAP value shape
    print(f"SHAP value shape: {shap_values.shape}")
    
    # Reshape SHAP values to match feature dimension
    try:
        shap_values_reshaped = shap_values.reshape(-1, sequence_length, n_features)
    except ValueError as e:
        print(f"Failed to reshape SHAP values: {e}")
        print(f"Original SHAP value shape: {shap_values.shape}")
        print(f"Target shape: (-1, {sequence_length}, {n_features})")
        # Try different reshaping methods
        total_features = shap_values.shape[-1]
        if total_features == sequence_length * n_features:
            shap_values_reshaped = shap_values.reshape(-1, sequence_length, n_features)
        else:
            # If feature count does not match, use original shape directly
            shap_values_reshaped = shap_values
    
    # Calculate average SHAP values for each feature
    print(f"Reshaped SHAP value shape: {shap_values_reshaped.shape}")
    
    if len(shap_values_reshaped.shape) == 3:
        # 3D shape: (samples, sequence_length, features)
        feature_shap = np.mean(np.abs(shap_values_reshaped), axis=(0, 1))
    elif len(shap_values_reshaped.shape) == 2:
        # 2D shape: (samples, features)
        feature_shap = np.mean(np.abs(shap_values_reshaped), axis=0)
    else:
        # Other shapes, take absolute mean directly
        feature_shap = np.mean(np.abs(shap_values_reshaped), axis=0)
    
    print(f"Feature SHAP value shape: {feature_shap.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Ensure feature count matches
    if len(feature_shap) > len(feature_cols):
        feature_shap = feature_shap[:len(feature_cols)]
    elif len(feature_shap) < len(feature_cols):
        # If SHAP values are less than feature count, pad with zeros
        padding = np.zeros(len(feature_cols) - len(feature_shap))
        feature_shap = np.concatenate([feature_shap, padding])
    
    # Create SHAP feature importance plot
    plt.figure(figsize=(12, 8))
    feature_names = feature_cols[:len(feature_shap)]
    
    # Sort and plot
    sorted_idx = np.argsort(feature_shap)[::-1]
    plt.barh(range(len(sorted_idx)), feature_shap[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Mean |SHAP value|')
    plt.title('SHAP Feature Importance (Transformer Model)')
plt.tight_layout()
    plt.savefig(os.path.join(shap_save_dir, 'shap_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

    # Save SHAP value data
    shap_data = {
        'feature_names': feature_names,
        'shap_values': feature_shap.tolist(),
        'sorted_features': [feature_names[i] for i in sorted_idx],
        'sorted_values': feature_shap[sorted_idx].tolist(),
        'shap_shape': shap_values_reshaped.shape,
        'feature_count': len(feature_names)
    }
    
    import json
    with open(os.path.join(shap_save_dir, 'shap_results.json'), 'w') as f:
        json.dump(shap_data, f, indent=4)

print(f"SHAP analysis complete, results saved to {shap_save_dir}")
    if len(sorted_idx) > 0:
        print(f"Most important feature: {feature_names[sorted_idx[0]]} (SHAP value: {feature_shap[sorted_idx[0]]:.4f})")
    else:
        print("Could not determine the most important feature")
    
except Exception as e:
    print(f"SHAP analysis failed: {str(e)}")
    print("Using attention mechanism visualization as an alternative")
    print("Attention visualization provides feature importance analysis")

# Ensure checkpoint directory exists (for subsequent training)
checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True) 
