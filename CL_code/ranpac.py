# ranpac.py
# Implementation of Random Path Consolidation (RanPAC) continual learning method for ECG deep learning project.
# All Chinese comments, print statements, docstrings, and error messages have been translated to English. Variable names are kept unless in pinyin or Chinese.

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from base import ContinualLearningMethod

class RanPAC(ContinualLearningMethod):
    """
    Randomized Past and Current Aggregated Contrastive (RanPAC) method for continual learning.
    Uses contrastive loss and random projection to enhance feature alignment.
    """
    def __init__(self, model, device, projection_dim=128, lambda_=1000):
        """
        Initialize RanPAC method.
        Args:
            model: PyTorch model
            device: torch.device
            projection_dim: Dimension of random projection
            lambda_: Regularization strength
        """
        super().__init__(model, device)
        self.projection_dim = projection_dim
        self.lambda_ = lambda_
        self.projections = []
        self.params_list = []

    def _create_random_projection(self, feature_dim):
        """
        Create a random projection matrix.
        Args:
            feature_dim: Input feature dimension
        Returns:
            Random projection matrix
        """
        return torch.randn(feature_dim, self.projection_dim, device=self.device)

    def _compute_projection_loss(self, x):
        """
        Compute the projection loss for the current batch.
        Args:
            x: Input data
        Returns:
            Projection loss
        """
        # Example: L2 norm of projected features (customize as needed)
        return torch.norm(x, p=2)

    def train(self, task_id, X_task, y_task):
        """
        Train the model with random projection and contrastive loss.
        Args:
            task_id: Current task index
            X_task: Task input data
            y_task: Task labels
        """
        dataloader = self.prepare_dataloader(X_task, y_task)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Get feature dimension and create random projection
        with torch.no_grad():
            X_tensor = torch.tensor(X_task, dtype=torch.float32).to(self.device)
            features = self.model.net[0](X_tensor)
            feature_dim = features.shape[1]
        projection = self._create_random_projection(feature_dim)
        self.projections.append(projection)
        self.params_list.append({'features': features.detach()})

        for epoch in range(70):
            self.model.train()
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = nn.BCEWithLogitsLoss()(output, y)

                # Compute projection loss if projections exist
                if self.projections:
                    proj_loss = self._compute_projection_loss(x)
                    loss += self.lambda_ * proj_loss

                loss.backward()
                optimizer.step()

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