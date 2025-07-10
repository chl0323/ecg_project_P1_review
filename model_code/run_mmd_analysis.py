# run_mmd_analysis.py
# Script to run MMD-based drift analysis for ECG deep learning project.


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import os
import json
import sys
from pathlib import Path
from typing import List, Union, Optional, Dict

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

class MMDDriftDetector:
    """
    Maximum Mean Discrepancy (MMD) drift detector
    
    Reference:
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
    A kernel two-sample test. Journal of Machine Learning Research, 13(Mar), 723-773.
    """
    
    def __init__(self, reference_data: np.ndarray, kernel: str = 'rbf', 
                 gamma: float = 1.0, threshold: float = 0.1):
        """
        Initialize MMD drift detector
        
        Args:
            reference_data: Reference dataset
            kernel: Kernel type ('rbf', 'linear', 'polynomial')
            gamma: RBF kernel parameter
            threshold: Drift detection threshold
        """
        self.reference_data = reference_data
        self.kernel = kernel
        self.gamma = gamma
        self.threshold = threshold
        self.mmd_history = []
        self.drift_points = []
        self.warning_points = []
        
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, gamma: float = None) -> np.ndarray:
        """Calculate RBF kernel matrix"""
        if gamma is None:
            gamma = self.gamma
        
        # Calculate squared Euclidean distance
        X_norm = np.sum(X**2, axis=1, keepdims=True)
        Y_norm = np.sum(Y**2, axis=1, keepdims=True)
        dist_sq = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)
        
        return np.exp(-gamma * dist_sq)
    
    def _linear_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Calculate linear kernel matrix"""
        return np.dot(X, Y.T)
    
    def _polynomial_kernel(self, X: np.ndarray, Y: np.ndarray, 
                          degree: int = 3, coef0: float = 1.0) -> np.ndarray:
        """Calculate polynomial kernel matrix"""
        return (np.dot(X, Y.T) + coef0) ** degree
    
    def _compute_kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Calculate kernel matrix based on kernel type"""
        if self.kernel == 'rbf':
            return self._rbf_kernel(X, Y)
        elif self.kernel == 'linear':
            return self._linear_kernel(X, Y)
        elif self.kernel == 'polynomial':
            return self._polynomial_kernel(X, Y)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")
    
    def compute_mmd(self, X: np.ndarray, Y: np.ndarray, unbiased: bool = True) -> float:
        """
        Calculate MMD between two datasets
        
        Args:
            X: First dataset
            Y: Second dataset
            unbiased: Whether to use unbiased estimation
            
        Returns:
            MMD value
        """
        m, n = len(X), len(Y)
        
        # Calculate kernel matrix
        K_XX = self._compute_kernel_matrix(X, X)
        K_YY = self._compute_kernel_matrix(Y, Y)
        K_XY = self._compute_kernel_matrix(X, Y)
        
        if unbiased:
            # Unbiased estimation (remove diagonal elements)
            K_XX_sum = np.sum(K_XX) - np.trace(K_XX)
            K_YY_sum = np.sum(K_YY) - np.trace(K_YY)
            K_XY_sum = np.sum(K_XY)
            
            mmd = (K_XX_sum / (m * (m - 1)) + 
                   K_YY_sum / (n * (n - 1)) - 
                   2 * K_XY_sum / (m * n))
        else:
            # Biased estimation
            mmd = (np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY))
        
        return max(0, mmd)  # Ensure MMD is non-negative
    
    def update(self, new_data: np.ndarray) -> Dict:
        """
        Update detector and detect drift
        
        Args:
            new_data: New data batch
            
        Returns:
            Detection result dictionary
        """
        # Calculate MMD with reference data
        mmd_value = self.compute_mmd(self.reference_data, new_data)
        self.mmd_history.append(mmd_value)
        
        # Detect drift
        drift_detected = mmd_value > self.threshold
        warning_detected = mmd_value > self.threshold * 0.8  # Warning threshold is 80%
        
        if drift_detected:
            self.drift_points.append(len(self.mmd_history) - 1)
        
        if warning_detected and not drift_detected:
            self.warning_points.append(len(self.mmd_history) - 1)
        
        return {
            'mmd': mmd_value,
            'drift_detected': drift_detected,
            'warning_detected': warning_detected,
            'threshold': self.threshold
        }
    
    def plot_mmd_history(self, save_path: Optional[str] = None):
        """Plot MMD history"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.mmd_history, label='MMD', color='blue', linewidth=2)
        plt.axhline(y=self.threshold, color='red', linestyle='--', 
                   label=f'Drift Threshold ({self.threshold})')
        plt.axhline(y=self.threshold * 0.8, color='orange', linestyle='--', 
                   label=f'Warning Threshold ({self.threshold * 0.8:.3f})')
        
        # Mark drift points
        if self.drift_points:
            plt.scatter(self.drift_points, [self.mmd_history[i] for i in self.drift_points], 
                       color='red', label='Drift Detected', marker='*', s=100)
        
        # Mark warning points
        if self.warning_points:
            plt.scatter(self.warning_points, [self.mmd_history[i] for i in self.warning_points], 
                       color='orange', label='Warning', marker='^', s=60)
        
        plt.title('MMD-based Drift Detection History')
        plt.xlabel('Time Step')
        plt.ylabel('MMD Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def analyze_embedding_drift(embeddings_list: list, labels_list: list, 
                          task_names: list = None, kernel: str = 'rbf'):
    """
    Analyze drift in embedding space
    
    Args:
        embeddings_list: Embedding lists for each task
        labels_list: Label lists for each task
        task_names: List of task names
        kernel: Kernel type used for MMD calculation
    """
    # Create save directory
    save_dir = 'results/drift_detection'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'metrics'), exist_ok=True)
    
    if task_names is None:
        task_names = [f'Task_{i+1}' for i in range(len(embeddings_list))]
    
    # Initialize result dictionary
    results = {
        'mmd_matrix': np.zeros((len(task_names), len(task_names))),
        'class_wise_mmd': {},
        'statistics': {}
    }
    
    # Calculate statistics for each task
    print("\nCalculating task statistics...")
    for i, (emb, lab) in enumerate(zip(embeddings_list, labels_list)):
        task_stats = {
            'mean': np.mean(emb, axis=0),
            'std': np.std(emb, axis=0),
            'min': np.min(emb, axis=0),
            'max': np.max(emb, axis=0),
            'class_distribution': {
                '0': np.sum(lab == 0),
                '1': np.sum(lab == 1)
            }
        }
        results['statistics'][task_names[i]] = task_stats
        
        print(f"\n{task_names[i]} Statistics:")
        print(f"  Mean range: [{task_stats['mean'].min():.4f}, {task_stats['mean'].max():.4f}]")
        print(f"  Std range: [{task_stats['std'].min():.4f}, {task_stats['std'].max():.4f}]")
        print(f"  Class distribution: 0={task_stats['class_distribution']['0']}, 1={task_stats['class_distribution']['1']}")
    
    # Calculate MMD between tasks
    print("\nCalculating MMD between tasks...")
    for i in range(len(task_names)):
        for j in range(i+1, len(task_names)):
            print(f"\nCalculating MMD for {task_names[i]} vs {task_names[j]}:")
            print(f"  {task_names[i]} data range: [{embeddings_list[i].min():.4f}, {embeddings_list[i].max():.4f}]")
            print(f"  {task_names[j]} data range: [{embeddings_list[j].min():.4f}, {embeddings_list[j].max():.4f}]")
            
            # Calculate median distance of data dimensions as a reference for gamma
            X = embeddings_list[i]
            Y = embeddings_list[j]
            X_norm = np.sum(X**2, axis=1, keepdims=True)
            Y_norm = np.sum(Y**2, axis=1, keepdims=True)
            dist_sq = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)
            median_dist = np.median(dist_sq[dist_sq > 0])
            gamma = 1.0 / median_dist if median_dist > 0 else 1.0
            
            print(f"  Using gamma = {gamma:.4f}")
            detector = MMDDriftDetector(embeddings_list[i], kernel=kernel, gamma=gamma)
            mmd_value = detector.compute_mmd(embeddings_list[i], embeddings_list[j], unbiased=False)
            
            # Print statistics of kernel matrix
            K_XX = detector._compute_kernel_matrix(embeddings_list[i], embeddings_list[i])
            K_YY = detector._compute_kernel_matrix(embeddings_list[j], embeddings_list[j])
            K_XY = detector._compute_kernel_matrix(embeddings_list[i], embeddings_list[j])
            
            print(f"  K_XX range: [{K_XX.min():.4f}, {K_XX.max():.4f}], mean: {K_XX.mean():.4f}")
            print(f"  K_YY range: [{K_YY.min():.4f}, {K_YY.max():.4f}], mean: {K_YY.mean():.4f}")
            print(f"  K_XY range: [{K_XY.min():.4f}, {K_XY.max():.4f}], mean: {K_XY.mean():.4f}")
            print(f"  MMD value: {mmd_value:.4f}")
            
            results['mmd_matrix'][i, j] = mmd_value
            results['mmd_matrix'][j, i] = mmd_value
    
    # Visualize MMD matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(results['mmd_matrix'], annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=task_names, yticklabels=task_names)
    plt.title('Task-wise MMD Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plots', 'mmd_matrix.png'))
    plt.close()
    
    # Class-wise MMD analysis
    print("\nPerforming class-wise MMD analysis...")
    unique_labels = np.unique(np.concatenate(labels_list))
    for label in unique_labels:
        class_results = np.zeros((len(task_names), len(task_names)))
        for i in range(len(task_names)):
            for j in range(i+1, len(task_names)):
                # Get samples for current class
                mask_i = labels_list[i] == label
                mask_j = labels_list[j] == label
                if np.sum(mask_i) > 0 and np.sum(mask_j) > 0:
                    detector = MMDDriftDetector(embeddings_list[i][mask_i], kernel=kernel)
                    mmd_value = detector.compute_mmd(embeddings_list[i][mask_i], 
                                                   embeddings_list[j][mask_j],
                                                   unbiased=False)
                    class_results[i, j] = mmd_value
                    class_results[j, i] = mmd_value
        
        results['class_wise_mmd'][f'class_{int(label)}'] = [[float(val) for val in row] 
                                                           for row in class_results]
        
        # Visualize class-wise MMD matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(class_results, annot=True, fmt='.4f', cmap='YlOrRd',
                    xticklabels=task_names, yticklabels=task_names)
        plt.title(f'Class {int(label)} MMD Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'plots', f'class_{int(label)}_mmd_matrix.png'))
        plt.close()
    
    # Temporal MMD analysis
    if len(task_names) > 1:
        print("\nPerforming temporal MMD analysis...")
        temporal_mmd = []
        for i in range(len(task_names)-1):
            detector = MMDDriftDetector(embeddings_list[i], kernel=kernel)
            mmd_value = detector.compute_mmd(embeddings_list[i], embeddings_list[i+1])
            temporal_mmd.append(mmd_value)
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(temporal_mmd)), temporal_mmd, 'b-o')
        plt.title('Temporal MMD Analysis')
        plt.xlabel('Time Step')
        plt.ylabel('MMD Value')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'plots', 'temporal_mmd.png'))
        plt.close()
        
        results['temporal_mmd'] = temporal_mmd
    
    # Create interactive MMD matrix visualization
    fig = go.Figure(data=go.Heatmap(
        z=results['mmd_matrix'],
        x=task_names,
        y=task_names,
        colorscale='YlOrRd',
        text=[[f'{val:.4f}' for val in row] for row in results['mmd_matrix']],
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Interactive MMD Matrix',
        xaxis_title='Tasks',
        yaxis_title='Tasks'
    )
    
    pio.write_html(fig, os.path.join(save_dir, 'plots', 'interactive_mmd_matrix.html'))
    
    # Visualize data distribution
    print("\nGenerating data distribution visualization...")
    plt.figure(figsize=(15, 10))
    
    # 1. Mean distribution comparison
    plt.subplot(2, 2, 1)
    means = [stats['mean'] for stats in results['statistics'].values()]
    plt.boxplot(means, tick_labels=task_names)
    plt.title('Mean Distribution Across Tasks')
    plt.ylabel('Mean Value')
    
    # 2. Standard deviation distribution comparison
    plt.subplot(2, 2, 2)
    stds = [stats['std'] for stats in results['statistics'].values()]
    plt.boxplot(stds, tick_labels=task_names)
    plt.title('Standard Deviation Distribution Across Tasks')
    plt.ylabel('Standard Deviation')
    
    # 3. Class distribution comparison
    plt.subplot(2, 2, 3)
    class_dist = np.array([[stats['class_distribution']['0'], stats['class_distribution']['1']] 
                          for stats in results['statistics'].values()])
    x = np.arange(len(task_names))
    width = 0.35
    plt.bar(x - width/2, class_dist[:, 0], width, label='Class 0')
    plt.bar(x + width/2, class_dist[:, 1], width, label='Class 1')
    plt.title('Class Distribution Across Tasks')
    plt.xticks(x, task_names)
    plt.ylabel('Number of Samples')
    plt.legend()
    
    # 4. MMD matrix heatmap
    plt.subplot(2, 2, 4)
    sns.heatmap(results['mmd_matrix'], annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=task_names, yticklabels=task_names)
    plt.title('MMD Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plots', 'distribution_analysis.png'))
    plt.close()
    
    # Save statistics
    with open(os.path.join(save_dir, 'metrics', 'statistics.json'), 'w') as f:
        json.dump({
            'task_statistics': {
                name: {
                    'mean_range': [float(stats['mean'].min()), float(stats['mean'].max())],
                    'std_range': [float(stats['std'].min()), float(stats['std'].max())],
                    'class_distribution': {
                        '0': int(stats['class_distribution']['0']),
                        '1': int(stats['class_distribution']['1'])
                    }
                }
                for name, stats in results['statistics'].items()
            },
            'mmd_matrix': [[float(val) for val in row] for row in results['mmd_matrix']]
        }, f, indent=4)
    
    print(f"\nAnalysis complete. Results saved to: {save_dir}")

def main():
    # Load data
    print("Loading data...")
    try:
        embeddings = np.load('processed_data/ecg_embeddings.npy')
        labels = np.load('processed_data/ecg_labels.npy')
        print(f"Embedding data shape: {embeddings.shape}")
        print(f"Label data shape: {labels.shape}")
        print(f"Embedding data range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
        print(f"Unique label values: {np.unique(labels)}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Split data into three tasks
    task_size = len(embeddings) // 3
    embeddings_list = [
        embeddings[:task_size],
        embeddings[task_size:2*task_size],
        embeddings[2*task_size:]
    ]
    labels_list = [
        labels[:task_size],
        labels[task_size:2*task_size],
        labels[2*task_size:]
    ]
    
    print("\nTask data statistics:")
    for i, (emb, lab) in enumerate(zip(embeddings_list, labels_list)):
        print(f"Task{i+1}:")
        print(f"  Number of samples: {len(emb)}")
        print(f"  Embedding range: [{emb.min():.4f}, {emb.max():.4f}]")
        print(f"  Unique labels: {np.unique(lab)}")
    
    # Perform analysis
    analyze_embedding_drift(
        embeddings_list=embeddings_list,
        labels_list=labels_list,
        task_names=['Task1', 'Task2', 'Task3'],
        kernel='rbf'
    )

if __name__ == '__main__':
    main() 
