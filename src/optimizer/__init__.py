import torch
import torch.optim as optim



# Adam optimizer
def get_adam_optimizer(model: torch.nn.Module, **optimizer_params):
    """
    Returns Adam optimizer for the given model.

    Args:
        model: PyTorch model
        lr: learning rate
        weight_decay: L2 regularization
        betas: Adam betas
        eps: Adam epsilon
    """
    return optim.Adam(model.parameters(), **optimizer_params)


# SGD optimizer
def get_sgd_optimizer(model: torch.nn.Module, **optimizer_params):
    """
    Returns SGD optimizer for the given model.

    Args:
        model: PyTorch model
        lr: learning rate
        momentum: momentum factor
        weight_decay: L2 regularization
    """
    return optim.SGD(model.parameters(), **optimizer_params)
