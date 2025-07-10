# mian.py
# Main script for running continual learning experiments in ECG deep learning project.

import numpy as np
import torch
from replay import Replay
from ewc import EWC
from lwf import LwF
from gem import GEM
from icarl import iCaRL
from ranpac import RanPAC
from evaluation import evaluate_method, save_results, plot_individual_results, plot_comparison_metrics, find_best_method
from models import MLP
import os


def main():
    """
    Main function to run continual learning experiments.
    Loads data, splits into tasks, initializes models and methods, evaluates all methods, and saves results.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    # Load data
    data = np.load('processed_data/ecg_embeddings.npy', allow_pickle=True)
    # If ecg_embeddings.npy is a dict or structured array, extract embeddings and anchor_age
    if isinstance(data, dict) and 'embeddings' in data and 'anchor_age' in data:
        embeddings = data['embeddings']
        ages = data['anchor_age']
    elif hasattr(data, 'dtype') and 'anchor_age' in data.dtype.names:
        embeddings = data['embeddings']
        ages = data['anchor_age']
    else:
        # fallback: assume embeddings only, and load ages separately
        embeddings = data
        ages = np.load('processed_data/ecg_ages.npy')
    labels = np.load('processed_data/ecg_labels.npy')

    # Prepare tasks by age group: 18-35, 36-50, 51-70
    task_datasets = []
    age_bins = [(18, 35), (36, 50), (51, 70)]
    for age_min, age_max in age_bins:
        indices = np.where((ages >= age_min) & (ages <= age_max))[0]
        X_task = embeddings[indices]
        y_task = labels[indices]
        task_datasets.append((X_task, y_task))
    # Now task_datasets[0] is 18-35, [1] is 36-50, [2] is 51-70

    # Initialize model and methods
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_dim=embeddings.shape[1]).to(device)

    methods = {
        'EWC': EWC(model, device, lambda_=5000),
        'Replay': Replay(model, device, memory_size=1000),
        'RanPAC': RanPAC(model, device, projection_dim=128, lambda_=1000),
        'LwF': LwF(model, device, temperature=2.0, lambda_=1.0),
        'GEM': GEM(model, device, memory_size=1000),
        'iCaRL': iCaRL(model, device, memory_size=1000)
    }

    # Evaluate all methods
    results = {}
    for method_name, method in methods.items():
        print(f"\nEvaluating {method_name}...")
        results[method_name] = evaluate_method(method, task_datasets)

    # Save results
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    save_results(results, save_dir)

    # Plot individual results
    plot_individual_results(results, save_dir)

    # Plot comparison metrics
    plot_comparison_metrics(results, save_dir)

    # Find best method
    best_methods = find_best_method(results)

    # Print best methods
    print("\nBest method analysis:")
    for metric, (method, score) in best_methods.items():
        print(f"{metric.upper()}: {method} (Average score: {score:.4f})")

    # Save best methods
    with open(f'{save_dir}/best_methods.txt', 'w') as f:
        f.write("Best method analysis:\n")
        for metric, (method, score) in best_methods.items():
            f.write(f"{metric.upper()}: {method} (Average score: {score:.4f})\n")

if __name__ == "__main__":
    main()
