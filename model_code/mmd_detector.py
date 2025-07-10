# mmd_detector.py
# Maximum Mean Discrepancy (MMD) detector for drift detection in ECG deep learning project.
# All Chinese comments, print statements, docstrings, and error messages have been translated to English. Variable names are kept unless in pinyin or Chinese.

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Optional, Dict
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import os

class MMDDriftDetector:
    """
    Maximum Mean Discrepancy (MMD) drift detector.
    
    Reference:
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
    A kernel two-sample test. Journal of Machine Learning Research, 13(Mar), 723-773.
    """
    
    def __init__(self, reference_data: np.ndarray, kernel: str = 'rbf', 
                 gamma: float = None, threshold: float = 0.1):
        """
        Initialize the MMD drift detector.
        
        Args:
            reference_data: Reference dataset.
            kernel: Kernel type ('rbf', 'linear', 'polynomial').
            gamma: RBF kernel gamma parameter, defaults to 1/n if None.
            threshold: Drift detection threshold.
        """
        self.reference_data = reference_data
        self.kernel = kernel
        self.gamma = gamma if gamma is not None else 1.0 / reference_data.shape[1]
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
        """Compute kernel matrix based on kernel type."""
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
        Compute MMD between two datasets.
        
        Args:
            X: First dataset.
            Y: Second dataset.
            unbiased: Whether to use unbiased estimation.
            
        Returns:
            MMD value.
        """
        m, n = len(X), len(Y)
        
        # Compute kernel matrices
        K_XX = self._compute_kernel_matrix(X, X)
        K_YY = self._compute_kernel_matrix(Y, Y)
        K_XY = self._compute_kernel_matrix(X, Y)
        
        if unbiased:
            # Unbiased estimation (remove diagonal elements)
            K_XX_sum = np.sum(K_XX) - np.trace(K_XX)
            K_YY_sum = np.sum(K_YY) - np.trace(K_YY)
            K_XY_sum = np.sum(K_XY)
            
            # Add numerical stability check
            if m <= 1 or n <= 1:
                print(f"Warning: Insufficient sample size (m={m}, n={n})")
                return 0.0
                
            # Calculate contributions
            term1 = K_XX_sum / (m * (m - 1))
            term2 = K_YY_sum / (n * (n - 1))
            term3 = 2 * K_XY_sum / (m * n)
            
            # Print debug information
            print(f"  MMD calculation details:")
            print(f"    term1 (K_XX): {term1:.6f}")
            print(f"    term2 (K_YY): {term2:.6f}")
            print(f"    term3 (K_XY): {term3:.6f}")
            
            mmd = term1 + term2 - term3
        else:
            # Biased estimation
            term1 = np.mean(K_XX)
            term2 = np.mean(K_YY)
            term3 = 2 * np.mean(K_XY)
            
            # Print debug information
            print(f"  MMD calculation details (biased estimation):")
            print(f"    term1 (K_XX): {term1:.6f}")
            print(f"    term2 (K_YY): {term2:.6f}")
            print(f"    term3 (K_XY): {term3:.6f}")
            
            mmd = term1 + term2 - term3
        
        # Ensure MMD is non-negative and add a small numerical stability constant
        mmd = max(0, mmd) + 1e-10
        
        return mmd
    
    def update(self, new_data: np.ndarray) -> Dict:
        """
        Update detector and detect drift.
        
        Args:
            new_data: New data batch.
            
        Returns:
            Detection result dictionary.
        """
        # Compute MMD with reference data
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