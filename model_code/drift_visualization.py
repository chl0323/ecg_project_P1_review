# drift_visualization.py
# Visualization utilities for drift detection results in ECG deep learning project.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
import os

def analyze_embedding_drift(embeddings_list, labels_list, task_names=None,
                           save_dir='analysis_results/embedding_drift'):
    """
    Analyze embedding drift.
    
    Args:
        embeddings_list: List of embeddings for different tasks.
        labels_list: Corresponding label lists.
        task_names: List of task names.
        save_dir: Directory to save results.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if task_names is None:
        task_names = [f'Task {i+1}' for i in range(len(embeddings_list))]
    
    print(f"\n=== Embedding Drift Analysis ===")
    print(f"Number of tasks: {len(embeddings_list)}")
    
    # 1. Calculate center distance between tasks
    centers = []
    for embeddings, labels in zip(embeddings_list, labels_list):
        # Calculate center for each class
        unique_labels = np.unique(labels)
        task_centers = {}
        for label in unique_labels:
            mask = labels == label
            center = np.mean(embeddings[mask], axis=0)
            task_centers[label] = center
        centers.append(task_centers)
    
    # 2. Visualize embedding distribution for different tasks
    if len(embeddings_list) == 1:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        axes = np.array([[ax1], [ax2]])
    else:
        fig, axes = plt.subplots(2, len(embeddings_list), figsize=(5*len(embeddings_list), 10))
    
    # Perform PCA and visualize for each task
    for i, (embeddings, labels, task_name) in enumerate(zip(embeddings_list, labels_list, task_names)):
        # PCA dimensionality reduction to 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Upper row: PCA visualization
        ax = axes[0, i] if len(embeddings_list) > 1 else axes[0, 0]
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           c=labels, cmap='viridis', alpha=0.6)
        ax.set_title(f'{task_name}\nPC1 vs PC2')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        
        # Lower row: Density distribution
        ax = axes[1, i] if len(embeddings_list) > 1 else axes[1, 0]
        for label in np.unique(labels):
            mask = labels == label
            ax.hist(np.linalg.norm(embeddings[mask], axis=1), 
                   alpha=0.5, label=f'Class {label}', bins=30)
        ax.set_title(f'{task_name}\nEmbedding Magnitude Distribution')
        ax.set_xlabel('Embedding Magnitude')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/embedding_drift_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Calculate drift metrics
    drift_metrics = {}
    if len(embeddings_list) > 1:
        for i in range(1, len(embeddings_list)):
            # Calculate center drift
            center_drift = {}
            for label in centers[0].keys():
                if label in centers[i]:
                    drift = np.linalg.norm(centers[0][label] - centers[i][label])
                    center_drift[f'class_{label}'] = float(drift)
            
            drift_metrics[f'task_{i}_vs_baseline'] = {
                'center_drift': center_drift,
                'mean_center_drift': float(np.mean(list(center_drift.values())))
            }
    
    # Save drift analysis results
    with open(f'{save_dir}/drift_metrics.json', 'w') as f:
        json.dump(drift_metrics, f, indent=4)
    
    print(f"Embedding drift analysis completed, results saved to: {save_dir}")
    return drift_metrics 
