from torch.nn import functional as F
import logging
import torch
from torch import nn
from typing import List, Tuple, Optional


from manifold_flow.utils import various
from .vector_transform_helpers import RandomPermutation, SVDLinear
from .coupling_transforms import PiecewiseRationalQuadraticCouplingTransform

logger = logging.getLogger(__name__)


class CompositeTransform(nn.Module):
    """Composes several transforms into one, in the order they are given."""

    def __init__(self, transforms):
        """Constructor.

        Args:
            transforms: an iterable of `Transform` objects.
        """
        super().__init__()
        self._transforms: torch.nn.ModuleList = nn.ModuleList(transforms)

    def forward(self, inputs: torch.Tensor, context: Optional[torch.Tensor]=None, full_jacobian: bool=False)-> Tuple[torch.Tensor, torch.Tensor]:
        funcs = self._transforms
        batch_size = various.shapes_to_tensor(inputs.shape)[0]
        outputs = inputs
        if full_jacobian:
            total_jacobian = None
            for func in funcs:
                inputs = outputs
                outputs, jacobian = func(inputs, context, full_jacobian=True)
                total_jacobian = jacobian if total_jacobian is None else torch.bmm(jacobian, total_jacobian)

            return outputs, total_jacobian

        else:
            total_logabsdet = torch.zeros(batch_size)
            for func in funcs:
                outputs, logabsdet = func(outputs, context)
                total_logabsdet += logabsdet
            return outputs, total_logabsdet

    def inverse(self, inputs: torch.Tensor, context: Optional[torch.Tensor]=None, full_jacobian: bool=False)-> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = various.shapes_to_tensor(inputs.shape)[0]
        outputs = inputs

        if full_jacobian:
            total_jacobian = None
            for transform in self._transforms[::-1]:
                inputs = outputs
                outputs, jacobian = transform.inverse(inputs, context, full_jacobian=True)
                total_jacobian = jacobian if total_jacobian is None else torch.bmm(jacobian, total_jacobian)

            return outputs, total_jacobian

        else:
            total_logabsdet = torch.zeros(batch_size)
            for transform in self._transforms[::-1]:
                outputs, logabsdet = transform.inverse(outputs, context)
                total_logabsdet += logabsdet
            return outputs, total_logabsdet


def _create_vector_linear_transform(linear_transform_type, features):
    if linear_transform_type == "permutation":
        return RandomPermutation(features=features)
    elif linear_transform_type == "svd":
        return CompositeTransform([RandomPermutation(features=features), SVDLinear(features, num_householder=10)])
    else:
        raise ValueError


def _create_vector_base_transform(
    i,
    base_transform_type,
    features,
    hidden_features,
    num_transform_blocks,
    dropout_probability,
    use_batch_norm,
    num_bins,
    tail_bound,
    apply_unconditional_transform,
    context_features,
):

    if base_transform_type == "rq-coupling":
        return PiecewiseRationalQuadraticCouplingTransform(
            mask=various.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            apply_unconditional_transform=apply_unconditional_transform,
            hidden_features=hidden_features, # ResNet params
            context_features=context_features,
            num_blocks=num_transform_blocks,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
    else:
        raise ValueError


def create_vector_transform(
    dim,
    flow_steps,
    linear_transform_type="permutation",
    base_transform_type="rq-coupling",
    hidden_features=100,
    num_transform_blocks=2,
    dropout_probability=0.0,
    use_batch_norm=False,
    num_bins=8,
    tail_bound=3,
    apply_unconditional_transform=False,
    context_features=None,
):
    transform = CompositeTransform(
        [
            CompositeTransform(
                [
                    _create_vector_linear_transform(linear_transform_type, dim),
                    _create_vector_base_transform(
                        i,
                        base_transform_type,
                        dim,
                        hidden_features,
                        num_transform_blocks,
                        dropout_probability,
                        use_batch_norm,
                        num_bins,
                        tail_bound,
                        apply_unconditional_transform,
                        context_features,
                    ),
                ]
            )
            for i in range(flow_steps)
        ]
        + [_create_vector_linear_transform(linear_transform_type, dim)]
    )
    return transform
