import torch
from torch.autograd import grad 

def gradients_wrt_params(
    net: torch.nn.Module, loss_tensor: torch.Tensor
):
    # Dictionary to store gradients for each parameter
    # Compute gradients with respect to each parameter
    for name, param in net.named_parameters():
        g = grad(loss_tensor, param, retain_graph=True)[0]
        param.grad = g

def update_params(net: torch.nn.Module, lr: float) -> None:
    # Update parameters for the network
    for name, param in net.named_parameters():
        param.data += lr * param.grad