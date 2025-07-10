# drift_detection_module.py
# Module for detecting data distribution drift in ECG deep learning project.
# All Chinese comments, print statements, docstrings, and error messages have been translated to English. Variable names are kept unless in pinyin or Chinese.

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Optional, Callable, Dict, Tuple
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os

class BaseDriftDetector:
    """Base class for drift detectors."""
    def __init__(self):
        self.warning_detected = False
        self.drift_detected = False
        self.history = {
            'error_rate': [],
            'warning_level': [],
            'drift_level': [],
            'detection_level': []
        }
    
    def reset(self):
        """Reset detector state."""
        raise NotImplementedError
    
    def update(self, error: Union[bool, int, float]):
        """Update detector state."""
        raise NotImplementedError
    
    def plot_history(self, save_path: Optional[str] = None):
        """Plot detection history."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['error_rate'], label='Error Rate', color='blue')
        plt.plot(self.history['warning_level'], label='Warning Level', color='orange', linestyle='--')
        plt.plot(self.history['drift_level'], label='Drift Level', color='red', linestyle='--')
        plt.plot(self.history['detection_level'], label='Detection Level', color='green')
        
        # Mark warning and drift points
        warning_points = np.where(np.array(self.history['detection_level']) > self.warning_level)[0]
        drift_points = np.where(np.array(self.history['detection_level']) > self.drift_level)[0]
        
        plt.scatter(warning_points, np.array(self.history['detection_level'])[warning_points], 
                   color='orange', label='Warning Detected', marker='^')
        plt.scatter(drift_points, np.array(self.history['detection_level'])[drift_points], 
                   color='red', label='Drift Detected', marker='*')
        
        plt.title('Drift Detection History')
        plt.xlabel('Sample Index')
        plt.ylabel('Error Rate / Level')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

class DDM(BaseDriftDetector):
    """
    Drift Detection Method (DDM)
    
    Reference:
    Gama, J., Medas, P., Castillo, G., & Rodrigues, P. (2004).
    Learning with drift detection. In Brazilian symposium on artificial intelligence (pp. 286-295).
    """
    
    def __init__(self, min_num_instances: int = 30, warning_level: float = 2.0, drift_level: float = 3.0):
        super().__init__()
        self.min_num_instances = min_num_instances
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.reset()
    
    def reset(self):
        """Reset detector."""
        super().reset()
        self.n_i = 1
        self.p_i = 1
        self.s_i = 0
        self.psi = 0
        
        self.n_min = None
        self.p_min = None
        self.s_min = None
        self.psmin = None
    
    def update(self, error: Union[bool, int, float]):
        """
        Update detector state
        
        Args:
            error: 0 for correct prediction, 1 for incorrect prediction
        """
        if isinstance(error, bool):
            error = 1 if error else 0
        
        # Increment instance count
        self.n_i += 1
        
        # Update error rate
        self.p_i = self.p_i + (error - self.p_i) / self.n_i
        self.s_i = np.sqrt(self.p_i * (1 - self.p_i) / self.n_i)
        
        self.warning_detected = False
        self.drift_detected = False
        
        if self.n_i < self.min_num_instances:
            return
        
        if self.n_min is None or self.p_i + self.s_i < self.p_min + self.s_min:
            self.n_min = self.n_i
            self.p_min = self.p_i
            self.s_min = self.s_i
            self.psmin = self.p_min + self.s_min
        
        # Detect drift
        current_level = (self.p_i + self.s_i - self.psmin) / self.s_min
        
        # Update history
        self.history['error_rate'].append(self.p_i)
        self.history['warning_level'].append(self.warning_level)
        self.history['drift_level'].append(self.drift_level)
        self.history['detection_level'].append(current_level)
        
        if current_level > self.drift_level:
            self.drift_detected = True
            self.reset()
        elif current_level > self.warning_level:
            self.warning_detected = True

class EDDM(BaseDriftDetector):
    """
    Early Drift Detection Method (EDDM)
    
    Reference:
    Baena-García, M., del Campo-Ávila, J., Fidalgo, R., Bifet, A., Gavaldà, R., & Morales-Bueno, R. (2006).
    Early drift detection method. In Fourth international workshop on knowledge discovery from data streams (pp. 77-86).
    """
    
    def __init__(self, min_num_instances: int = 30, warning_level: float = 0.95, drift_level: float = 0.9):
        super().__init__()
        self.min_num_instances = min_num_instances
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.reset()
    
    def reset(self):
        """Reset detector."""
        super().reset()
        self.n_i = 0
        self.p_i = 0
        self.s_i = 0
        self.p2_i = 0
        
        self.n_max = None
        self.p_max = None
        self.s_max = None
        self.p2_max = None
    
    def update(self, error: Union[bool, int, float]):
        """
        Update detector state
        
        Args:
            error: 0 for correct prediction, 1 for incorrect prediction
        """
        if isinstance(error, bool):
            error = 1 if error else 0
        
        # Increment instance count
        self.n_i += 1
        
        # Update error rate
        self.p_i = self.p_i + (error - self.p_i) / self.n_i
        self.p2_i = self.p2_i + (error * error - self.p2_i) / self.n_i
        self.s_i = np.sqrt(self.p2_i - self.p_i * self.p_i)
        
        self.warning_detected = False
        self.drift_detected = False
        
        if self.n_i < self.min_num_instances:
            return
        
        if self.n_max is None or self.p_i + 2 * self.s_i > self.p_max + 2 * self.s_max:
            self.n_max = self.n_i
            self.p_max = self.p_i
            self.s_max = self.s_i
            self.p2_max = self.p2_i
        
        # Detect drift
        current_level = (self.p_i + 2 * self.s_i) / (self.p_max + 2 * self.s_max)
        
        # Update history
        self.history['error_rate'].append(self.p_i)
        self.history['warning_level'].append(self.warning_level)
        self.history['drift_level'].append(self.drift_level)
        self.history['detection_level'].append(current_level)
        
        if current_level < self.drift_level:
            self.drift_detected = True
            self.reset()
        elif current_level < self.warning_level:
            self.warning_detected = True

class MMDDriftDetector:
    """
    Maximum Mean Discrepancy (MMD) based drift detector
    
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
            kernel: Type of kernel function ('rbf', 'linear', 'polynomial')
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
        """Calculate RBF kernel matrix."""
        if gamma is None:
            gamma = self.gamma
        
        # Calculate squared Euclidean distance
        X_norm = np.sum(X**2, axis=1, keepdims=True)
        Y_norm = np.sum(Y**2, axis=1, keepdims=True)
        dist_sq = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)
        
        return np.exp(-gamma * dist_sq)
    
    def _linear_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Calculate linear kernel matrix."""
        return np.dot(X, Y.T)
    
    def _polynomial_kernel(self, X: np.ndarray, Y: np.ndarray, 
                          degree: int = 3, coef0: float = 1.0) -> np.ndarray:
        """Calculate polynomial kernel matrix."""
        return (np.dot(X, Y.T) + coef0) ** degree
    
    def _compute_kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Calculate kernel matrix based on kernel type."""
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
        Compute MMD between two datasets
        
        Args:
            X: First dataset
            Y: Second dataset
            unbiased: Whether to use unbiased estimation
            
        Returns:
            MMD value
        """
        m, n = len(X), len(Y)
        
        # Calculate kernel matrices
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
    
    def compute_mmd_with_permutation_test(self, X: np.ndarray, Y: np.ndarray, 
                                        n_permutations: int = 1000, 
                                        alpha: float = 0.05) -> Dict:
        """
        Compute MMD and evaluate significance using permutation test
        
        Args:
            X: First dataset
            Y: Second dataset
            n_permutations: Number of permutations
            alpha: Significance level
            
        Returns:
            Dictionary containing MMD value, p-value, and significance
        """
        # Compute real MMD
        real_mmd = self.compute_mmd(X, Y)
        
        # Combine data and perform permutation test
        combined_data = np.vstack([X, Y])
        m, n = len(X), len(Y)
        
        permutation_mmds = []
        for _ in range(n_permutations):
            # Random permutation
            indices = np.random.permutation(len(combined_data))
            X_perm = combined_data[indices[:m]]
            Y_perm = combined_data[indices[m:]]
            
            # Compute permuted MMD
            perm_mmd = self.compute_mmd(X_perm, Y_perm)
            permutation_mmds.append(perm_mmd)
        
        # Calculate p-value
        p_value = np.mean(np.array(permutation_mmds) >= real_mmd)
        
        return {
            'mmd': real_mmd,
            'p_value': p_value,
            'significant': p_value < alpha,
            'threshold': np.percentile(permutation_mmds, (1 - alpha) * 100),
            'permutation_mmds': permutation_mmds
        }
    
    def update(self, new_data: np.ndarray) -> Dict:
        """
        Update detector and detect drift
        
        Args:
            new_data: New data batch
            
        Returns:
            Detection result dictionary
        """
        # Compute MMD with reference data
        mmd_value = self.compute_mmd(self.reference_data, new_data)
        self.mmd_history.append(mmd_value)
        
        # Detect drift
        drift_detected = mmd_value > self.threshold
        warning_detected = mmd_value > self.threshold * 0.8  # Warning threshold is 80% of drift threshold
        
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
        """Plot MMD history."""
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

def detect_drift_in_sequence(sequence: np.ndarray, detector: BaseDriftDetector, 
                           window_size: int = 100) -> dict:
    """
    Detect drift in a sequence of data.
    
    Args:
        sequence: Input sequence
        detector: Drift detector instance
        window_size: Size of sliding window
    
    Returns:
        Dictionary containing detection results.
    """
    results = {
        'drift_points': [],
        'warning_points': [],
        'error_rates': [],
        'detection_levels': []
    }
    
    for i in range(0, len(sequence), window_size):
        window = sequence[i:i+window_size]
        error_rate = np.mean(window)
        
        detector.update(error_rate)
        results['error_rates'].append(error_rate)
        results['detection_levels'].append(detector.history['detection_level'][-1])
        
        if detector.drift_detected:
            results['drift_points'].append(i)
        if detector.warning_detected:
            results['warning_points'].append(i)
    
    return results

def analyze_embedding_drift_with_mmd(embeddings_list: List[np.ndarray], 
                                   labels_list: List[np.ndarray],
                                   task_names: Optional[List[str]] = None,
                                   kernel: str = 'rbf',
                                   save_dir: str = '/content/drive/MyDrive/ecg_colab_final/results/drift_detection') -> Dict:
    """
    Analyze drift in embedding space using MMD.
    
    Args:
        embeddings_list: List of embeddings for different tasks
        labels_list: Corresponding labels for each task
        task_names: List of task names
        kernel: Type of kernel function
        save_dir: Directory to save results
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'metrics'), exist_ok=True)
    
    if task_names is None:
        task_names = [f'Task {i+1}' for i in range(len(embeddings_list))]
    
    print(f"\n=== MMD Embedding Drift Analysis ===")
    print(f"Number of tasks: {len(embeddings_list)}")
    print(f"Kernel: {kernel}")
    print(f"Results will be saved to: {save_dir}")
    
    # Initialize results dictionary
    results = {
        'mmd_matrix': np.zeros((len(embeddings_list), len(embeddings_list))),
        'class_wise_mmd': {},
        'task_names': task_names,
        'kernel': kernel
    }
    
    # 1. Calculate overall MMD between tasks
    print("Calculating overall MMD between tasks...")
    for i in range(len(embeddings_list)):
        for j in range(i + 1, len(embeddings_list)):
            # Create MMD detector
            detector = MMDDriftDetector(embeddings_list[i], kernel=kernel)
            mmd_value = detector.compute_mmd(embeddings_list[i], embeddings_list[j])
            
            results['mmd_matrix'][i, j] = mmd_value
            results['mmd_matrix'][j, i] = mmd_value
            
            print(f"MMD({task_names[i]} -> {task_names[j]}): {mmd_value:.6f}")
    
    # 2. Class-wise MMD analysis
    print("Calculating class-wise MMD...")
    unique_labels = np.unique(np.concatenate(labels_list))
    
    for label in unique_labels:
        class_mmd_matrix = np.zeros((len(embeddings_list), len(embeddings_list)))
        
        for i in range(len(embeddings_list)):
            for j in range(i + 1, len(embeddings_list)):
                # Extract embeddings for specific classes
                mask_i = labels_list[i] == label
                mask_j = labels_list[j] == label
                
                if np.sum(mask_i) > 0 and np.sum(mask_j) > 0:
                    class_emb_i = embeddings_list[i][mask_i]
                    class_emb_j = embeddings_list[j][mask_j]
                    
                    detector = MMDDriftDetector(class_emb_i, kernel=kernel)
                    mmd_value = detector.compute_mmd(class_emb_i, class_emb_j)
                    
                    class_mmd_matrix[i, j] = mmd_value
                    class_mmd_matrix[j, i] = mmd_value
        
        results['class_wise_mmd'][f'class_{label}'] = class_mmd_matrix
    
    # 3. Visualize MMD matrices
    fig, axes = plt.subplots(1, len(unique_labels) + 1, 
                            figsize=(5 * (len(unique_labels) + 1), 4))
    if len(unique_labels) == 0:
        axes = [axes]
    
    # Overall MMD heatmap
    ax = axes[0] if len(axes) > 1 else axes
    sns.heatmap(results['mmd_matrix'], annot=True, fmt='.4f', 
               xticklabels=task_names, yticklabels=task_names,
               cmap='Reds', ax=ax)
    ax.set_title('Overall MMD Matrix')
    
    # Class-wise MMD heatmaps
    for idx, label in enumerate(unique_labels):
        if len(axes) > idx + 1:
            ax = axes[idx + 1]
            sns.heatmap(results['class_wise_mmd'][f'class_{label}'], 
                       annot=True, fmt='.4f',
                       xticklabels=task_names, yticklabels=task_names,
                       cmap='Reds', ax=ax)
            ax.set_title(f'Class {label} MMD Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'visualizations', 'mmd_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Temporal MMD analysis (using the first task as reference)
    if len(embeddings_list) > 1:
        print("Performing temporal drift analysis...")
        reference_embeddings = embeddings_list[0]
        detector = MMDDriftDetector(reference_embeddings, kernel=kernel, threshold=0.1)
        
        mmd_timeline = []
        for i in range(1, len(embeddings_list)):
            result = detector.update(embeddings_list[i])
            mmd_timeline.append(result['mmd'])
        
        # Plot temporal MMD
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(embeddings_list)), mmd_timeline, 'bo-', linewidth=2)
        plt.axhline(y=detector.threshold, color='red', linestyle='--', 
                   label=f'Threshold ({detector.threshold})')
        plt.xlabel('Task Index')
        plt.ylabel('MMD from Reference')
        plt.title('Temporal MMD Drift Analysis')
        plt.xticks(range(1, len(embeddings_list)), task_names[1:])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'visualizations', 'temporal_mmd_drift.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        results['temporal_mmd'] = mmd_timeline
    
    # 5. Interactive visualization
    # Create interactive MMD matrix
    fig = go.Figure(data=go.Heatmap(
        z=results['mmd_matrix'],
        x=task_names,
        y=task_names,
        colorscale='Reds',
        text=results['mmd_matrix'],
        texttemplate="%{text:.4f}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Interactive MMD Matrix",
        xaxis_title="Tasks",
        yaxis_title="Tasks"
    )
    
    pio.write_html(fig, os.path.join(save_dir, 'visualizations', 'interactive_mmd_matrix.html'))
    
    # 6. Save results
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'mmd_matrix': results['mmd_matrix'].tolist(),
        'task_names': results['task_names'],
        'kernel': results['kernel']
    }
    
    if 'temporal_mmd' in results:
        json_results['temporal_mmd'] = results['temporal_mmd']
    
    # Process class_wise_mmd
    json_results['class_wise_mmd'] = {}
    for key, value in results['class_wise_mmd'].items():
        json_results['class_wise_mmd'][key] = value.tolist()
    
    with open(os.path.join(save_dir, 'metrics', 'mmd_analysis_results.json'), 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"MMD drift analysis completed, results saved to: {save_dir}")
    return results

def optimize_mmd_kernel_parameters(X: np.ndarray, Y: np.ndarray, 
                                 kernel: str = 'rbf',
                                 param_range: Optional[List] = None) -> Dict:
    """
    Optimize MMD kernel parameters.
    
    Args:
        X: First dataset
        Y: Second dataset
        kernel: Type of kernel function
        param_range: Parameter search range
        
    Returns:
        Optimal parameter and corresponding MMD value.
    """
    if param_range is None:
        if kernel == 'rbf':
            param_range = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        else:
            param_range = [1.0]
    
    best_mmd = 0
    best_param = param_range[0]
    mmd_values = []
    
    for param in param_range:
        if kernel == 'rbf':
            detector = MMDDriftDetector(X, kernel=kernel, gamma=param)
        else:
            detector = MMDDriftDetector(X, kernel=kernel)
        
        mmd = detector.compute_mmd(X, Y)
        mmd_values.append(mmd)
        
        if mmd > best_mmd:
            best_mmd = mmd
            best_param = param
    
    return {
        'best_parameter': best_param,
        'best_mmd': best_mmd,
        'parameter_range': param_range,
        'mmd_values': mmd_values
    }