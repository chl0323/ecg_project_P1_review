# evaluation.py
# Evaluation utilities for continual learning models in ECG deep learning project.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

# Add other necessary imports as needed

def evaluate_method(method, task_datasets):
    """
    Evaluate a continual learning method on a sequence of tasks.
    Args:
        method: ContinualLearningMethod instance
        task_datasets: List of (X, y) tuples for each task
    Returns:
        Dictionary of metric matrices (accuracy, f1, auc, recall)
    """
    n_tasks = len(task_datasets)
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    results = {metric: np.zeros((n_tasks, n_tasks)) for metric in metrics}

    for task_id, (X_train, y_train) in enumerate(task_datasets):
        method.train(task_id, X_train, y_train)
        for eval_id, (X_eval, y_eval) in enumerate(task_datasets):
            scores = method.evaluate(X_eval, y_eval)
            for metric in metrics:
                results[metric][task_id, eval_id] = scores[metric]
    return results


def save_results(results, save_dir):
    """
    Save evaluation results as .npy files in the specified directory.
    Args:
        results: Dictionary of results
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    for method, method_results in results.items():
        for metric, matrix in method_results.items():
            np.save(os.path.join(save_dir, f'{method}_{metric}.npy'), matrix)


def plot_individual_results(results, save_dir='results'):
    """
    Plot heatmaps for each method and metric.
    Args:
        results: Dictionary of results
        save_dir: Directory to save plots
    """
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    methods = list(results.keys())
    for method in methods:
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            matrix = results[method][metric]
            sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd')
            plt.title(f'{method} - {metric.upper()}')
            plt.xlabel('Task')
            plt.ylabel('Task')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{method}_results.png')
        plt.close()


def plot_comparison_metrics(results, save_dir='results'):
    """
    Plot line charts comparing all methods for each metric across tasks.
    Args:
        results: Dictionary of results
        save_dir: Directory to save plots
    """
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    methods = list(results.keys())
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        task_means = []
        for method in methods:
            matrix = results[method][metric]
            means = np.mean(matrix, axis=1)
            task_means.append(means)
        for i, method in enumerate(methods):
            plt.plot(range(1, len(task_means[i]) + 1), task_means[i], marker='o', label=method)
        plt.title(f'{metric.upper()} Comparison Across Tasks')
        plt.xlabel('Task')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/{metric}_comparison.png')
        plt.close()


def find_best_method(results):
    """
    Find the best method for each metric based on average diagonal score.
    Args:
        results: Dictionary of results
    Returns:
        Dictionary mapping metric to (best_method, best_score)
    """
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    methods = list(results.keys())
    best_methods = {}
    for metric in metrics:
        best_score = -1
        best_method = None
        for method in methods:
            matrix = results[method][metric]
            current_task_scores = [matrix[i, i] for i in range(matrix.shape[0])]
            mean_score = np.mean(current_task_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_method = method
        best_methods[metric] = (best_method, best_score)
    return best_methods

def calculate_transfer_metrics(results):
    """
    Calculate transfer learning metrics (BWT, FWT, Forget Rate)
    Args:
        results: Evaluation results dictionary
    Returns:
        transfer_metrics: Dictionary containing transfer metrics
    """
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    methods = list(results.keys())
    
    transfer_metrics = {}
    for method in methods:
        transfer_metrics[method] = {}
        for metric in metrics:
            matrix = results[method][metric]
            
            # Calculate BWT (Backward Transfer)
            bwt = np.mean([matrix[i, -1] - matrix[i, i] 
                          for i in range(matrix.shape[0] - 1)])
            
            # Calculate FWT (Forward Transfer)
            fwt = np.mean([matrix[i, i-1] - matrix[i, 0] 
                          for i in range(1, matrix.shape[0])])
            
            # Calculate Forget Rate
            forget_rate = np.mean([np.max(matrix[i, :i+1]) - matrix[i, -1] 
                                 for i in range(matrix.shape[0] - 1)])
            
            transfer_metrics[method][metric] = {
                'BWT': bwt,
                'FWT': fwt,
                'Forget Rate': forget_rate
            }
    
    return transfer_metrics

def plot_transfer_metrics(transfer_metrics, save_dir='results'):
    """
    Plot comparison charts for transfer learning metrics
    Args:
        transfer_metrics: Transfer metrics dictionary
        save_dir: Directory to save plots
    """
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    transfer_types = ['BWT', 'FWT', 'Forget Rate']
    methods = list(transfer_metrics.keys())
    
    for transfer_type in transfer_types:
        plt.figure(figsize=(12, 6))
        
        for metric in metrics:
            values = [transfer_metrics[method][metric][transfer_type] 
                     for method in methods]
            plt.plot(methods, values, marker='o', label=metric.upper())
        
        plt.title(f'{transfer_type} Comparison')
        plt.xlabel('Method')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{transfer_type}_comparison.png')
        plt.close()

def plot_forgetting_curve(results, save_dir='results'):
    """
    Plot forgetting curves for each method and metric
    Args:
        results: Evaluation results dictionary
        save_dir: Directory to save plots
    """
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    methods = list(results.keys())
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        for method in methods:
            matrix = results[method][metric]
            # Calculate max performance and final performance for each task
            max_performance = np.max(matrix, axis=1)
            final_performance = matrix[:, -1]
            forgetting = max_performance - final_performance
            
            plt.plot(range(1, len(forgetting)+1), forgetting, 
                    marker='o', label=method)
        
        plt.title(f'{metric.upper()} Forgetting Curve')
        plt.xlabel('Task')
        plt.ylabel('Forgetting')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/{metric}_forgetting_curve.png')
        plt.close()

def plot_learning_curve(results, save_dir='results'):
    """
    Plot learning curves for each method and metric
    Args:
        results: Evaluation results dictionary
        save_dir: Directory to save plots
    """
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    methods = list(results.keys())
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        for method in methods:
            matrix = results[method][metric]
            # Calculate mean performance for each task
            mean_performance = np.mean(matrix, axis=1)
            
            plt.plot(range(1, len(mean_performance)+1), mean_performance, 
                    marker='o', label=method)
        
        plt.title(f'{metric.upper()} Learning Curve')
        plt.xlabel('Task')
        plt.ylabel('Average Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/{metric}_learning_curve.png')
        plt.close()
