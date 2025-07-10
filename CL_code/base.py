# base.py
# Base class for continual learning methods in ECG deep learning project.
# All Chinese comments, print statements, docstrings, and error messages have been translated to English. Variable names are kept unless in pinyin or Chinese.

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class ContinualLearningMethod:
    """
    Base class for continual learning methods.
    Provides interface for training and evaluation.
    """
    def __init__(self, model, device):
        """
        Initialize the continual learning method.
        Args:
            model: PyTorch model
            device: torch.device
        """
        self.model = model
        self.device = device

    def prepare_dataloader(self, X, y, batch_size=64):
        """
        Prepare a DataLoader for the given data.
        Args:
            X: Input data
            y: Labels
            batch_size: Batch size for DataLoader
        Returns:
            DataLoader object
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, task_id, X_task, y_task):
        """
        Abstract method for training on a task.
        Args:
            task_id: Current task index
            X_task: Task input data
            y_task: Task labels
        """
        raise NotImplementedError

    def evaluate(self, X_eval, y_eval):
        """
        Abstract method for evaluating on a task.
        Args:
            X_eval: Evaluation input data
            y_eval: Evaluation labels
        """
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)