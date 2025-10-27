import torch
import torch.nn as nn
import torch.nn.functional as F



# Base class for multi-label loss
class MultiLabelLoss(nn.Module):
    """
    Base class for multi-label classification loss.
    Model outputs logits.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        raise NotImplementedError("Forward method must be implemented by subclass")


# BCE Loss
class BCELoss(MultiLabelLoss):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)
    

# Weighted BCE Loss
class WeightedBCELoss(MultiLabelLoss):
    def __init__(self, pos_weight: torch.Tensor, reduction='mean'):
        """
        pos_weight: torch.Tensor of shape (n_classes,)
        """
        super().__init__(reduction)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)
    

# Focal Loss
class FocalLoss(MultiLabelLoss):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__(reduction)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits: raw model outputs (B, n_outputs)
        targets: binary labels (B, n_outputs)
        """
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_factor * (1 - pt) ** self.gamma

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss