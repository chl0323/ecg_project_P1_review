# gem.py
# Implementation of Gradient Episodic Memory (GEM) continual learning method for ECG deep learning project.
# All Chinese comments, print statements, docstrings, and error messages have been translated to English. Variable names are kept unless in pinyin or Chinese.

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from base import ContinualLearningMethod

class GEM(ContinualLearningMethod):
    """
    Gradient Episodic Memory (GEM) method for continual learning.
    Uses gradient projection to prevent interference with previous tasks.
    """
    def __init__(self, model, device, memory_size=1000):
        """
        Initialize GEM method.
        Args:
            model: PyTorch model
            device: torch.device
            memory_size: Number of samples to store in memory
        """
        super().__init__(model, device)
        self.memory_size = memory_size
        self.memory = []
        self.grad_dims = []
        self.grads = []

    def _compute_grad(self, x, y):
        """
        Compute the gradient for the current batch.
        Args:
            x: Input data
            y: Labels
        Returns:
            Flattened gradient vector
        """
        self.model.zero_grad()
        output = self.model(x)
        loss = nn.BCEWithLogitsLoss()(output, y)
        loss.backward()
        grad = []
        for p in self.model.parameters():
            if p.grad is not None:
                grad.append(p.grad.data.view(-1))
        return torch.cat(grad)

    def _project_grad(self, grad):
        """
        Project the gradient to prevent interference with previous tasks.
        Args:
            grad: Current gradient vector
        Returns:
            Projected gradient vector
        """
        if not self.grads:
            return grad

        for g in self.grads:
            dot_product = torch.dot(grad, g)
            if dot_product < 0:
                grad -= (dot_product / torch.dot(g, g)) * g
        return grad

    def train(self, task_id, X_task, y_task):
        """
        Train the model with GEM gradient projection.
        Args:
            task_id: Current task index
            X_task: Task input data
            y_task: Task labels
        """
        dataloader = self.prepare_dataloader(X_task, y_task)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Update memory with random samples
        indices = np.random.choice(len(X_task), min(self.memory_size, len(X_task)), replace=False)
        self.memory = [(X_task[i], y_task[i]) for i in indices]

        for epoch in range(70):
            self.model.train()
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                # Compute current gradient
                grad = self._compute_grad(x, y)

                # Project gradient if previous gradients exist
                if self.grads:
                    grad = self._project_grad(grad)

                # Update parameters
                optimizer.zero_grad()
                output = self.model(x)
                loss = nn.BCEWithLogitsLoss()(output, y)
                loss.backward()

                # Apply projected gradient
                idx = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.data = grad[idx:idx + p.grad.numel()].view(p.grad.shape)
                        idx += p.grad.numel()

                optimizer.step()

            # Save gradient after each task
            if task_id > 0:
                self.grads.append(grad.detach())

    def evaluate(self, X_eval, y_eval):
        """
        Evaluate the model on the given data.
        Args:
            X_eval: Evaluation input data
            y_eval: Evaluation labels
        Returns:
            Dictionary with accuracy, f1, auc, and recall scores
        """
        self.model.eval()
        X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_eval_tensor).cpu().numpy()
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            preds = (probs > 0.5).astype(int)

        return {
            'accuracy': accuracy_score(y_eval, preds),
            'f1': f1_score(y_eval, preds),
            'auc': roc_auc_score(y_eval, probs),
            'recall': recall_score(y_eval, preds)
        }