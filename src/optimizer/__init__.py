import torch
import torch.optim as optim



# Adam optimizer
def get_adam_optimizer(model: torch.nn.Module, lr: float = 1e-4, weight_decay: float = 0.0, betas=(0.9, 0.999), eps=1e-8):
    """
    Returns Adam optimizer for the given model.

    Args:
        model: PyTorch model
        lr: learning rate
        weight_decay: L2 regularization
        betas: Adam betas
        eps: Adam epsilon
    """
    return optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


# SGD optimizer
def get_sgd_optimizer(model: torch.nn.Module, lr: float = 1e-2, momentum: float = 0.9, weight_decay: float = 0.0):
    """
    Returns SGD optimizer for the given model.

    Args:
        model: PyTorch model
        lr: learning rate
        momentum: momentum factor
        weight_decay: L2 regularization
    """
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
