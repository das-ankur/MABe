import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score



class MultiLabelEvaluator:
    """
    Evaluation functions for multi-label classification.

    All functions assume:
        logits: raw model output (B, n_outputs)
        targets: binary labels (B, n_outputs)
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def sigmoid(self, logits):
        return torch.sigmoid(logits)

    def thresholded_predictions(self, logits):
        probs = self.sigmoid(logits)
        return (probs > self.threshold).float()

    def accuracy(self, logits, targets):
        """
        Returns multi-label accuracy: # correct labels / total labels
        """
        preds = self.thresholded_predictions(logits)
        correct = (preds == targets).float().sum()
        total = targets.numel()
        return (correct / total).item()

    def f1_score(self, logits, targets, average='micro'):
        preds = self.thresholded_predictions(logits).cpu().numpy()
        targets = targets.cpu().numpy()
        return f1_score(targets, preds, average=average, zero_division=0)

    def precision(self, logits, targets, average='micro'):
        preds = self.thresholded_predictions(logits).cpu().numpy()
        targets = targets.cpu().numpy()
        return precision_score(targets, preds, average=average, zero_division=0)

    def recall(self, logits, targets, average='micro'):
        preds = self.thresholded_predictions(logits).cpu().numpy()
        targets = targets.cpu().numpy()
        return recall_score(targets, preds, average=average, zero_division=0)

    def auroc(self, logits, targets, average='micro'):
        """
        AUROC per class. logits -> probabilities via sigmoid
        """
        probs = self.sigmoid(logits).cpu().numpy()
        targets = targets.cpu().numpy()
        try:
            return roc_auc_score(targets, probs, average=average)
        except ValueError:
            # Happens if a class has only one label in the batch
            return float('nan')
