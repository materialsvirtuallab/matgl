"""
Custome Activation Fucntions
"""
import torch
import torch.nn as nn


class SoftPlus2(nn.Module):
    """
    SoftPlus2 activation function:
    out = log(alpha*exp(beta*x)+1*alpha)
    softplus function that is 0 at x=0, the implementation aims at avoiding overflow

    Arguments:
        alpha (float): parameter for the SoftPlus2 activation function
        beta (flaot): paramter for the SoftPlus2 activation function
    """

    def __init__(self, alpha: float = 0.5, beta: float = 1.0) -> None:
        """Initializes the SoftPlus2 class."""
        super(SoftPlus2, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate activation function given the input tensor x.
        Arguments:
            x (torch.tensor): Input tensor

        Returns:
            out (torch.tensor): Output tensor
        """
        out = self.relu(x) + torch.log(self.alpha * torch.exp(-torch.abs(x * self.beta)) + 1.0 * self.alpha)
        return out
