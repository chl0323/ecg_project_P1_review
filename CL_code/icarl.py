# icarl.py
# Implementation of Incremental Classifier and Representation Learning (iCaRL) continual learning method for ECG deep learning project.

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from base import ContinualLearningMethod

class iCaRL(ContinualLearningMethod):
    """
    Incremental Classifier and Representation Learning (iCaRL) method for continual learning.
    Uses class mean representations and exemplars for incremental classification.
    """
    def __init__(self, model, device, memory_size=1000):
        """
        Initialize iCaRL method.
        Args:
            model: PyTorch model
            device: torch.device
            memory_size: Number of exemplars to store
        """
        super().__init__(model, device)
        self.memory_size = memory_size
        self.memory = []
        self.exemplars = []

    def _select_exemplars(self, X, y, n_exemplars):
        """
        Select representative exemplars using K-means clustering.
        Args:
            X: Input data
            y: Labels
            n_exemplars: Number of exemplars to select
        Returns:
            Array of selected exemplars
        """
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_exemplars)
        kmeans.fit(X)

        exemplars = []
        for i in range(n_exemplars):
            cluster_samples = X[kmeans.labels_ == i]
            if len(cluster_samples) > 0:
                distances = np.linalg.norm(cluster_samples - kmeans.cluster_centers_[i], axis=1)
                exemplar_idx = np.argmin(distances)
                exemplars.append(cluster_samples[exemplar_idx])

        return np.array(exemplars)

    def train(self, task_id, X_task, y_task):
        """
        Train the model and update exemplars for incremental learning.
        Args:
            task_id: Current task index
            X_task: Task input data
            y_task: Task labels
        """
        # Select exemplars for the current task
        n_exemplars = min(self.memory_size // (task_id + 1), len(X_task))
        exemplars = self._select_exemplars(X_task, y_task, n_exemplars)
        self.exemplars.append(exemplars)

        # Prepare training data (current task + memory exemplars)
        X_train = X_task
        y_train = y_task
        if self.exemplars:
            X_memory = np.vstack(self.exemplars)
            y_memory = np.zeros(len(X_memory))  # Use 0 as memory label
            X_train = np.vstack([X_train, X_memory])
            y_train = np.concatenate([y_train, y_memory])

        dataloader = self.prepare_dataloader(X_train, y_train)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for epoch in range(70):
            self.model.train()
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = nn.BCEWithLogitsLoss()(output, y)
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
