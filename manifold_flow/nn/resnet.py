import torch
import numpy as np
from torch import nn
from torch.nn import functional as F, init
from typing import Optional

from manifold_flow.utils import various

class KDRightLayer(nn.Module):
    """ Custom KDRightLayer"""
    def __init__(self, input_dims, out_dims, activation_string='Identity', zero_initialization=False):
        super().__init__()
        
        self.activation_string = activation_string
        self.activation = getattr(nn, self.activation_string)()  

        self.input_dims = input_dims
        self.out_dims = out_dims
        '''
        Operation: XW + B
        X = input : m x n
        W = weight: n X target
        B = bias: m x target
        Output: m x target
        '''
        assert self.input_dims[1] == self.out_dims[0]
        weights = torch.rand(self.out_dims)
        self.weights = nn.Parameter(weights, requires_grad=True)

        if zero_initialization:
            stdv = 1e-3
        else:
            stdv = 1 / (math.sqrt(self.out_dims[1]))
        self.weights.data.uniform_(-stdv, stdv)
        bias = torch.rand((self.input_dims[0], self.out_dims[1]))

        self.bias = nn.Parameter(bias, requires_grad=True)
        self.bias.data.uniform_(-stdv, stdv)


    def forward(self, inputs):
        return self.activation(torch.matmul(inputs, self.weights) + self.bias)

class KDLeftLayer(nn.Module):
    """ Custom KDLeftLayer"""
    def __init__(self, input_dims, out_dims, activation_string='Identity', zero_initialization=False):
        super(KDLeftLayer, self).__init__()
        # self.n = n
        self.activation_string = activation_string
        self.activation =  getattr(nn, self.activation_string)()  

        self.input_dims = input_dims
        self.out_dims = out_dims
        '''
        Operation: WX but not possible directly
        Modified Operation: (X^t W^t)^t + B
        X = input: m x n, X^t: m x n
                 = i1 X i2, X^T = i2 X i1
        W = weight: o1 X o2
        (X^t W^t):
        (X^t W^t)^t: 
             = o1 X o2
        B = Bias: 
                = o1 X o2 
        '''
        weights = torch.rand(( self.out_dims[1], self.out_dims[0]))
        self.weights = nn.Parameter(weights, requires_grad=True) # transposed here --> # o2 X o1
        if zero_initialization:
            stdv = 1e-3
        else:
            stdv = 1 / (math.sqrt(self.out_dims[1]))
        self.weights.data.uniform_(-stdv, stdv)

        bias = torch.rand(self.out_dims) 
        self.bias = nn.Parameter(bias, requires_grad=True)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        # print(f"shapes KDR| input = {inputs.shape} | IN_DIMS = {self.input_dims} | out_dims = {self.out_dims}")
        return self.activation(torch.permute(torch.matmul(torch.permute(inputs,dims=[0, 2, 1]),
                                                  self.weights),dims=[0, 2, 1]) + self.bias)

import math
def factors(number):
    return [(int(x), int(number / x))  for x in range(int(math.sqrt(number)))[2:] if not number % x][-1]

def nearest_root_factors(n):
    def isqrt(n):
        x = n
        y = (x + 1) // 2
        while y < x:
            x = y
            y = (x + n // x) // 2
        return x
    sqrt = isqrt(n)
    if sqrt * sqrt == n:
        return int(sqrt), int(sqrt)
    else:
        s, t = 0, 0
        for i in range(sqrt, 0, -1):
            if n % i == 0:
                s = i
                t = n // i
                break
        return int(s), int(t)

class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(self, features: int, context_features: int, dropout_probability: float=0.0, use_batch_norm: bool=False, zero_initialization: bool=True):
        super().__init__()
        self.activation = F.relu

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)])
        # if context_features is not None:
        #     self.context_layer = nn.Linear(context_features, features)
        # else:
        #     self.context_layer = torch.zeros(features, features)

        self.input_dim1, self.input_dim2 = nearest_root_factors(features)
        self.out_dim1, self.out_dim2 = nearest_root_factors(features)

        # print(f"$$$$$$ features1 = {self.features_dim1} features2 = {self.features_dim2}")

        self.Kronecker_layer1 = nn.Sequential(KDLeftLayer(input_dims=[self.input_dim1,self.input_dim2], out_dims=[self.out_dim1, self.input_dim1],),
                                              KDRightLayer(input_dims=[self.out_dim1,self.input_dim1],out_dims=[self.input_dim1, self.out_dim2]),
                                               nn.Flatten())

        self.Kronecker_layer2 = nn.Sequential(KDLeftLayer(input_dims=[self.out_dim1,self.out_dim2], out_dims=[self.out_dim1, self.out_dim1], zero_initialization=True),
                                            KDRightLayer(input_dims=[self.out_dim1,self.out_dim1], out_dims=[self.out_dim1, self.out_dim2], zero_initialization=True),
                                               nn.Flatten())
        
        self.Kronecker_module = nn.ModuleList([self.Kronecker_layer1,self.Kronecker_layer2])

        self.dropout = nn.Dropout(p=dropout_probability)
        # if zero_initialization:
        #     init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
        #     init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs: torch.Tensor, context: Optional[torch.Tensor]=None) -> torch.Tensor:
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = temps.reshape(-1, self.input_dim1, self.input_dim2)
        temps = self.Kronecker_module[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        # there is no nn.reshape module, so manually reshape before using second Kronecker layer
        temps = temps.reshape((-1,self.out_dim1,self.out_dim2))
        temps = self.Kronecker_module[1](temps)
        # if context is not None:
        #     temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps

class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(self, in_features: int, out_features: int, hidden_features: int, context_features: int=None, num_blocks: int=2, dropout_probability:float=0.0, use_batch_norm: bool=False):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(in_features + context_features, hidden_features)
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        # print(f"$$$$$$ hidden features = {hidden_features} $$$$$$$$$")
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features, context_features=context_features, dropout_probability=dropout_probability, use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.i1, self.i2 = nearest_root_factors(hidden_features)
        o1, o2 = nearest_root_factors(out_features)
        # print(f"out_features = {out_features} = {o1} X {o2}")
        # print(f"hidden_features = {hidden_features} = {self.i1} X {self.i2}")

        self.final_layer = nn.Sequential(KDLeftLayer(input_dims=[self.i1, self.i2], out_dims=[o1, self.i1]), KDRightLayer(input_dims=[o1, self.i1], out_dims=[self.i1, o2]), nn.Flatten())
        # self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, inputs: torch.Tensor, context: Optional[torch.Tensor]=None)->torch.Tensor:
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        # print(f"temps shape before reshape = {temps.shape}")
        temps = temps.reshape(-1, self.i1, self.i2)
        outputs = self.final_layer(temps)
        return outputs

