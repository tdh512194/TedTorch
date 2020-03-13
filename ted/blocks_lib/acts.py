# import pytorch
import torch
from torch import nn
import torch.nn.functional as F

def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    '''
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    """
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Mish: Self Regularized Non-Monotonic Activation Function
    https://arxiv.org/pdf/1908.08681.pdf
    """
    def __init__(self, **kwargs): super(Mish, self).__init__()
    def forward(self, x): return mish(x)


def eswish(input, beta=1.75):
    """
    Applies the E-Swish function element-wise:
    ESwish(x, beta) = beta * x * sigmoid(x)
    """
    return beta * input * torch.sigmoid(input)

class ESwish(nn.Module):
    """
    Applies the E-Swish function element-wise:
    ESwish(x, beta) = beta * x * sigmoid(x)
    """
    def __init__(self, beta=1.75, **kwargs): super(ESwish, self).__init__(); self.beta = beta
    def forward(self, x): return eswish(x, self.beta)


def swish(input, beta=1.25):
    """
    Applies the Swish function element-wise:
    Swish(x, \\beta) = x*sigmoid(\\beta*x) = \\frac{x}{(1+e^{-\\beta*x})}
    """
    return input * torch.sigmoid(beta * input)

class Swish(nn.Module):
    """
    Applies the E-Swish function element-wise:
    Swish(x, beta) = beta * x * sigmoid(x)
    """
    def __init__(self, beta=1.25, **kwargs): super(Swish, self).__init__(); self.beta = beta
    def forward(self, x): return swish(x, self.beta)

def mila(input, beta=-0.25):
    """
    Applies the mila function element-wise:
    mila(x) = x * tanh(softplus(\\beta + x)) = x * tanh(ln(1 + e^{\\beta + x}))
    """
    return input * torch.tanh(F.softplus(input + beta))

class Mila(nn.Module):
    """
    Applies the mila function element-wise:
    mila(x) = x * tanh(softplus(\\beta + x)) = x * tanh(ln(1 + e^{\\beta + x}))
    """
    def __init__(self, beta=-0.25, **kwargs): super(Mila, self).__init__(); self.beta = beta
    def forward(self, x): return mila(x, self.beta)