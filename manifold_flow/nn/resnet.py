import torch

from torch import nn
from torch.nn import functional as F, init
from typing import Optional

from manifold_flow.utils import various


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
        self.linear_layers = nn.ModuleList([nn.Linear(features, features) for _ in range(2)])
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    # @torch.jit.export
    def forward(self, inputs: torch.Tensor, context: Optional[torch.Tensor]=None) -> torch.Tensor:
        temps = inputs
        # if self.use_batch_norm:
        #     temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        # if self.use_batch_norm:
        #     temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        # if context is not None:
        #     temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        # else:
            # temps = torch.zeros_like(inputs)
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
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features, context_features=context_features, dropout_probability=dropout_probability, use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)

    # @torch.jit.export
    def forward(self, inputs: torch.Tensor, context: Optional[torch.Tensor]=None)->torch.Tensor:
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)
        return outputs

