# new_ranpac_classifier.py
# RanPAC classifier implementation for ECG deep learning project.

import numpy as np
import torch
import torch.nn as nn
import os
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from torch.utils.data import TensorDataset, DataLoader

# Load deep feature embeddings and labels
print("Loading deep feature embeddings...")
try:
    embeddings = np.load('processed_data/ecg_embeddings.npy')
    labels = np.load('processed_data/ecg_labels.npy')
    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded labels: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels.astype(int))}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    print("Please run new_train_transformer2.py to generate embedding files.")
    exit(1)
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

# Split into N tasks (here: 3 tasks)
N_TASKS = 3
indices = np.arange(embeddings.shape[0])
np.random.shuffle(indices)
split_indices = np.array_split(indices, N_TASKS)

task_datasets = []
for idx in split_indices:
    X_task = embeddings[idx]
    y_task = labels[idx]
    task_datasets.append((X_task, y_task))

class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for classification.
    """
    def __init__(self, input_dim, hidden_dim=128):
        """
        Initialize the MLP model.
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Forward pass of the MLP.
        Args:
            x: Input tensor
        Returns:
            Output tensor
        """
        return self.net(x).squeeze(-1)

# Training and evaluation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = embeddings.shape[1]
model = MLP(input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
ewc_lambda = 5000  # EWC regularization strength

ewc_list = []
acc_matrix = np.zeros((N_TASKS, N_TASKS))
f1_matrix = np.zeros((N_TASKS, N_TASKS))
auc_matrix = np.zeros((N_TASKS, N_TASKS))
recall_matrix = np.zeros((N_TASKS, N_TASKS))

for task_id, (X_task, y_task) in enumerate(task_datasets):
    print(f"\nTraining task {task_id + 1}/{N_TASKS}")
    X_tensor = torch.tensor(X_task, dtype=torch.float32)
    y_tensor = torch.tensor(y_task, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Training loop
    for epoch in range(70):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = nn.BCEWithLogitsLoss()(output, y)
            if ewc_list:
                ewc_loss = sum([ewc.penalty(model) for ewc in ewc_list])
                loss += ewc_lambda * ewc_loss
            loss.backward()
            optimizer.step()

    # Save EWC object for current task
    ewc_list.append(EWC(model, loader, device=device))

    # Evaluation on all tasks
    for eval_id, (X_eval, y_eval) in enumerate(task_datasets):
        model.eval()
        X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)
        y_eval_tensor = torch.tensor(y_eval, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X_eval_tensor)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
        acc_matrix[task_id, eval_id] = accuracy_score(y_eval, preds.cpu())
        f1_matrix[task_id, eval_id] = f1_score(y_eval, preds.cpu())
        auc_matrix[task_id, eval_id] = roc_auc_score(y_eval, probs.cpu())
        recall_matrix[task_id, eval_id] = recall_score(y_eval, preds.cpu())

# Save results
np.save('results/model_performance/acc_matrix.npy', acc_matrix)
np.save('results/model_performance/f1_matrix.npy', f1_matrix)
np.save('results/model_performance/auc_matrix.npy', auc_matrix)
np.save('results/model_performance/recall_matrix.npy', recall_matrix)
print("Results saved to results/model_performance/")
