import torch
import torch.nn as nn
import torch.nn.functional as F



# Base class for multi-label loss
class MultiLabelLoss(nn.Module):
    """
    Base class for multi-label classification loss.
    Model outputs logits.
    """
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        raise NotImplementedError("Forward method must be implemented by subclass")


# BCE Loss
class BCELoss(MultiLabelLoss):
    def __init__(self, reduction='none'):
        super().__init__(reduction)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)
    

# Weighted BCE Loss
class WeightedBCELoss(MultiLabelLoss):
    def __init__(self, pos_weight: torch.Tensor, reduction='none'):
        """
        pos_weight: torch.Tensor of shape (n_classes,)
        """
        super().__init__(reduction)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)
    

# Focal Loss
class FocalLoss(MultiLabelLoss):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super().__init__(reduction)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits: raw model outputs (B, n_outputs)
        targets: binary labels (B, n_outputs)
        """
        # Clip logits to prevent extreme values
        logits = torch.clamp(logits, min=-100, max=100)
        
        # Calculate probabilities
        p = torch.sigmoid(logits)
        
        # Calculate pt (probability of true class)
        pt = p * targets + (1 - p) * (1 - targets)
        pt = torch.clamp(pt, min=1e-7, max=1 - 1e-7)  # Prevent taking log of 0
        
        # Calculate focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Calculate alpha weight
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Combine weights
        weight = alpha_weight * focal_weight
        
        # Calculate BCE loss (using logits for better numerical stability)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Calculate final loss
        loss = weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss