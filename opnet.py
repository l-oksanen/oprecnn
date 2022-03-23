import numpy as np
from torch import Tensor, zeros, matmul, reshape
from torch.nn.parameter import Parameter
import torch.nn.functional as F
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

class DotProdLayer(nn.Module):
    def __init__(self, dim):
        super(DotProdLayer, self).__init__()
        self.dim = dim
        self.b = Parameter(Tensor(1, dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization adapted from torch.nn.Bilinear
        # I have no idea if this is good or not 
        bound = 1 / np.sqrt(self.dim)
        init.uniform_(self.b, -bound, bound)

    def forward(self, h):
        return reshape(F.linear(reshape(h, (-1,self.dim)), self.b), (-1,))


class OperatorNet(nn.Module):
    def __init__(self, dim, num_layers, scalar_output=False, useReLU=False):
        super(OperatorNet, self).__init__()
        self.dim = dim
        self.scalar_output = scalar_output
        self.op_layers = [OperatorLayer(dim) for i in range(num_layers)]
        self.layers = nn.ModuleList(self.op_layers)
        self.useReLU = useReLU
        if self.useReLU:
            self.op_layers1 = [OperatorLayer(dim) for i in range(num_layers)]
            self.relu_layers = [nn.ReLU() for i in range(num_layers)]
            self.layers.extend(self.op_layers1)
            self.layers.extend(self.relu_layers)
        if scalar_output: 
            self.dp_layer = DotProdLayer(dim)
            self.layers.append(self.dp_layer)

    def forward(self, X):
        h = zeros(self.dim, 1) # a column vector
        for i in range(len(self.op_layers)):
            op0 = self.op_layers[i]
            if self.useReLU: 
                op1 = self.op_layers1[i]
                relu = self.relu_layers[i]
                h = op0(X, h) + relu(op1(X, h))
            else:
                h = op0(X, h)
        if self.scalar_output:
            h = self.dp_layer(h)
        return h
