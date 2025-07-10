# ewc.py
# Implementation of Elastic Weight Consolidation (EWC) continual learning method for ECG deep learning project.

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from base import ContinualLearningMethod

class EWC(ContinualLearningMethod):
    """
    Elastic Weight Consolidation (EWC) method for continual learning.
    Regularizes important parameters to retain knowledge from previous tasks.
    """
    def __init__(self, model, device, lambda_=5000):
        """
        Initialize EWC method.
        Args:
            model: PyTorch model
            device: torch.device
            lambda_: Regularization strength
        """
        super().__init__(model, device)
        self.lambda_ = lambda_
        self.fisher_list = []  # Store Fisher information
        self.params_list = []  # Store model parameters

    def _compute_fisher(self, dataloader):
        """
        Compute Fisher information matrix for the current task.
        Args:
            dataloader: DataLoader for the current task
        Returns:
            Dictionary of Fisher information for each parameter
        """
        fisher = {n: torch.zeros_like(p, device=self.device)
                  for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            output = self.model(x)
            loss = nn.BCEWithLogitsLoss()(output, y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    fisher[n] += p.grad.data.pow(2)
        for n in fisher:
            fisher[n] /= len(dataloader)
        return fisher

    def compute_ewc_loss(self):
        """
        Compute the EWC regularization loss for all previous tasks.
        Returns:
            Total EWC loss
        """
        loss = 0
        for fisher, params in zip(self.fisher_list, self.params_list):
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    _loss = fisher[n] * (p - params[n]).pow(2)
                    loss += _loss.sum()
        return loss

    def train(self, task_id, X_task, y_task):
        """
        Train the model with EWC regularization.
        Args:
            task_id: Current task index
            X_task: Task input data
            y_task: Task labels
        """
        dataloader = self.prepare_dataloader(X_task, y_task)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for epoch in range(70):
            self.model.train()
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = nn.BCEWithLogitsLoss()(output, y)

                if self.fisher_list:  # If there are previous tasks
                    ewc_loss = self.compute_ewc_loss()
                    loss += self.lambda_ * ewc_loss

                loss.backward()
                optimizer.step()

        # Save current parameters and Fisher information
        current_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        current_fisher = self._compute_fisher(dataloader)

        self.params_list.append(current_params)
        self.fisher_list.append(current_fisher)

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
