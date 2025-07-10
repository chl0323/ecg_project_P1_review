# lwf.py
# Implementation of Learning without Forgetting (LwF) continual learning method for ECG deep learning project.

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from base import ContinualLearningMethod

class LwF(ContinualLearningMethod):
    """
    Learning without Forgetting (LwF) method for continual learning.
    Uses knowledge distillation from old model outputs to retain previous knowledge.
    """
    def __init__(self, model, device, temperature=2.0, lambda_=1.0):
        """
        Initialize LwF method.
        Args:
            model: PyTorch model
            device: torch.device
            temperature: Distillation temperature
            lambda_: Loss weight for distillation
        """
        super().__init__(model, device)
        self.temperature = temperature
        self.lambda_ = lambda_
        self.old_model = None

    def _distill_loss(self, new_logits, old_logits):
        """
        Compute the knowledge distillation loss between new and old logits.
        Args:
            new_logits: Logits from the current model
            old_logits: Logits from the previous model
        Returns:
            Distillation loss
        """
        new_logits = new_logits.unsqueeze(-1)
        old_logits = old_logits.unsqueeze(-1)
        return nn.KLDivLoss(reduction='batchmean')(
            torch.log_softmax(new_logits / self.temperature, dim=0),
            torch.softmax(old_logits / self.temperature, dim=0)
        ) * (self.temperature ** 2)

    def train(self, task_id, X_task, y_task):
        """
        Train the model with knowledge distillation from the previous model.
        Args:
            task_id: Current task index
            X_task: Task input data
            y_task: Task labels
        """
        dataloader = self.prepare_dataloader(X_task, y_task)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Save the old model before training on the new task
        if self.old_model is None:
            self.old_model = type(self.model)(self.model.net[0].in_features).to(self.device)
        self.old_model.load_state_dict(self.model.state_dict())

        for epoch in range(70):
            self.model.train()
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                # Compute new task loss
                new_logits = self.model(x)
                loss = nn.BCEWithLogitsLoss()(new_logits, y)

                # Compute distillation loss if not the first task
                if task_id > 0:
                    with torch.no_grad():
                        old_logits = self.old_model(x)
                    distill_loss = self._distill_loss(new_logits, old_logits)
                    loss += self.lambda_ * distill_loss

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
