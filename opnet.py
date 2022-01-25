import numpy as np
from torch import Tensor, zeros, matmul
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn as nn

class OperatorLayer(nn.Module):
    def __init__(self, dim):
        super(OperatorLayer, self).__init__()
        self.dim = dim
        self.A = Parameter(Tensor(dim, dim))
        self.B = Parameter(Tensor(dim, dim))
        self.b = Parameter(Tensor(dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization adapted from torch.nn.Bilinear
        # I have no idea if this is good or not 
        bound = 1 / np.sqrt(self.dim)
        init.uniform_(self.A, -bound, bound)
        init.uniform_(self.B, -bound, bound)
        init.uniform_(self.b, -bound, bound)

    def forward(self, X, h):
        return self.b + matmul(self.A, h) + matmul(self.B, matmul(X, h))

class OperatorNet(nn.Module):
    def __init__(self, dim, num_layers):
        super(OperatorNet, self).__init__()
        self.dim = dim
        self.layers = nn.ModuleList(
            [OperatorLayer(dim) for i in range(num_layers)])

    def forward(self, X):
        h = zeros(self.dim, 1) # a column vector
        for layer in self.layers:
            h = layer(X, h)
        return h
