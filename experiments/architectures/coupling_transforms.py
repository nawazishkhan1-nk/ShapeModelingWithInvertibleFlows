from torch.nn import functional as F
import logging
import torch
from torch import nn
from torch.nn import init
import math
import warnings
from manifold_flow import nn as nn_

from manifold_flow.utils import various
from .rational_quadratic_splines import DEFAULT_MIN_BIN_WIDTH, DEFAULT_MIN_BIN_HEIGHT, DEFAULT_MIN_DERIVATIVE, unconstrained_rational_quadratic_spline, rational_quadratic_spline

from typing import List, Tuple, Optional
logger = logging.getLogger(__name__)

class PiecewiseRationalQuadraticCouplingTransform(nn.Module):
    def __init__(
        self,
        mask: torch.Tensor,
        hidden_features: int, # ResNet params
        context_features: int,
        num_blocks: int,
        dropout_probability: float,
        use_batch_norm: bool, # end resnet params
        num_bins: int=10,
        tails: str=None,
        tail_bound: float=1.0,
        apply_unconditional_transform: bool=False,
        img_shape: List[int]=None,
        min_bin_width: float=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float=DEFAULT_MIN_DERIVATIVE,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound


        unconditional_transform = None # Always False

        # Init from cupling transform
        mask = torch.as_tensor(mask)
        if mask.dim() != 1:
            raise ValueError("Mask must be a 1-dim tensor.")
        # if mask.numel() <= 0:
        #     raise ValueError("Mask can't be empty.")

        self.features = len(mask)
        features_vector = torch.arange(self.features)

        self.register_buffer("identity_features", features_vector.masked_select(mask <= 0))
        self.register_buffer("transform_features", features_vector.masked_select(mask > 0))

        assert (len(self.identity_features)) + (len(self.transform_features))== self.features

        in_features_for_transform_net = len(self.identity_features)
        out_features_for_transorm_net = len(self.transform_features) * self._transform_dim_multiplier()
        self.transform_net = nn_.ResidualNet(
                                            in_features=in_features_for_transform_net,
                                            out_features=out_features_for_transorm_net,
                                            hidden_features=hidden_features,
                                            context_features=context_features,
                                            num_blocks=num_blocks,
                                            dropout_probability=dropout_probability,
                                            use_batch_norm=use_batch_norm,
                                        )



        if unconditional_transform is None:
            self.unconditional_transform = None

    # @property
    # def num_identity_features(self):
    #     return len(self.identity_features)

    # @property
    # def num_transform_features(self):
    #     return len(self.transform_features)
    
    def forward(self, inputs: torch.Tensor, context: Optional[torch.Tensor]=None, full_jacobian: bool=False)->Tuple[torch.Tensor, torch.Tensor]:
        # if inputs.dim() not in [2, 4]:
        #     raise ValueError("Inputs must be a 2D or a 4D tensor.")

        # if inputs.shape[1] != self.features:
        #     raise ValueError("Expected features = {}, got {}.".format(self.features, inputs.shape[1]))

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        # logger.debug("identity_split depends on inputs: %s", utils.check_dependence(identity_split, inputs))
        # logger.debug("transform_split depends on inputs: %s", utils.check_dependence(transform_split, inputs))

        if full_jacobian:
            transform_params = self.transform_net(identity_split, context)
            transform_split, _ = self._coupling_transform_forward(inputs=transform_split, transform_params=transform_params)
            jacobian_transform = various.batch_jacobian(transform_split, inputs)

            # Put together full Jacobian
            batchsize = inputs.size(0)
            jacobian = torch.zeros((batchsize,) + inputs.size()[1:] + inputs.size()[1:])
            jacobian[:, self.identity_features, self.identity_features] = 1.0
            jacobian[:, self.transform_features, :] = jacobian_transform

            outputs = torch.empty_like(inputs)
            outputs[:, self.identity_features, ...] = identity_split
            outputs[:, self.transform_features, ...] = transform_split

            return outputs, jacobian

        else:
            transform_params = self.transform_net(identity_split, context)

            transform_split, logabsdet = self._coupling_transform_forward(inputs=transform_split, transform_params=transform_params)

            outputs = torch.empty_like(inputs)
            outputs[:, self.identity_features, ...] = identity_split
            outputs[:, self.transform_features, ...] = transform_split
            return outputs, logabsdet

    def inverse(self, inputs: torch.Tensor, context: Optional[torch.Tensor]=None, full_jacobian: bool=False)->Tuple[torch.Tensor, torch.Tensor]:
        # if inputs.dim() not in [2, 4]:
        #     raise ValueError("Inputs must be a 2D or a 4D tensor.")

        # if inputs.shape[1] != self.features:
        #     raise ValueError("Expected features = {}, got {}.".format(self.features, inputs.shape[1]))

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        if full_jacobian:
            # timer.timer(start="Jacobian inverse coupling transform")

            # if self.unconditional_transform is not None:
            #     # identity_split_after_unconditional_transform, jacobian_identity = self.unconditional_transform.inverse(
            #     #     identity_split, context, full_jacobian=True
            #     # )
            #     raise NotImplementedError()
            # else:
            identity_split_after_unconditional_transform = identity_split
            #     jacobian_identity = torch.eye(self.num_identity_features).unsqueeze(0)  # (1, n, n)

            transform_params = self.transform_net(identity_split, context)

            transform_split, _ = self._coupling_transform_inverse(inputs=transform_split, transform_params=transform_params)
            jacobian_transform = various.batch_jacobian(transform_split, inputs)

            # Put together full Jacobian
            batchsize = inputs.size(0)
            jacobian = torch.zeros((batchsize,) + inputs.size()[1:] + inputs.size()[1:])
            jacobian[:, self.identity_features, self.identity_features] = 1.0
            jacobian[:, self.transform_features, :] = jacobian_transform

            outputs = torch.empty_like(inputs)
            outputs[:, self.identity_features] = identity_split_after_unconditional_transform
            outputs[:, self.transform_features] = transform_split

            # timer.timer(stop="Jacobian inverse coupling transform")

            return outputs, jacobian

        else:
            logabsdet = 0.0
            # if self.unconditional_transform is not None:
            #     identity_split, logabsdet = self.unconditional_transform.inverse(identity_split, context)

            transform_params = self.transform_net(identity_split, context)
            transform_split, logabsdet_split = self._coupling_transform_inverse(inputs=transform_split, transform_params=transform_params)
            logabsdet += logabsdet_split

            outputs = torch.empty_like(inputs)
            outputs[:, self.identity_features] = identity_split
            outputs[:, self.transform_features] = transform_split

            return outputs, logabsdet

    def _coupling_transform_forward(self, inputs: torch.Tensor, transform_params: torch.Tensor, full_jacobian: bool=False)->Tuple[torch.Tensor, torch.Tensor]:
        return self._coupling_transform(inputs, transform_params, inverse=False, full_jacobian=full_jacobian)

    def _coupling_transform_inverse(self, inputs: torch.Tensor, transform_params: torch.Tensor, full_jacobian: bool=False)->Tuple[torch.Tensor, torch.Tensor]:
        return self._coupling_transform(inputs, transform_params, inverse=True, full_jacobian=full_jacobian)

    def _coupling_transform(self, inputs: torch.Tensor, transform_params: torch.Tensor, inverse: bool=False, full_jacobian: bool=False)->Tuple[torch.Tensor, torch.Tensor]:
        # if inputs.dim() == 4:
        #     b, c, h, w = inputs.shape
        #     # For images, reshape transform_params from Bx(C*?)xHxW to BxCxHxWx?
        #     transform_params = transform_params.reshape(b, c, -1, h, w).permute(0, 1, 3, 4, 2)
        # elif inputs.dim() == 2:
        if True:
            b, d = inputs.size(0), inputs.size(1)
            # For 2D data, reshape transform_params from Bx(D*?) to BxDx?
            transform_params = transform_params.reshape(b, d, -1)

        if full_jacobian:
            outputs, jacobian = self._piecewise_cdf(inputs, transform_params, inverse, full_jacobian=True)
            return outputs, jacobian
        else:
            outputs, logabsdet = self._piecewise_cdf(inputs, transform_params, inverse)
            return outputs, various.sum_except_batch(logabsdet)

    def _piecewise_cdf(self, inputs: torch.Tensor, transform_params: torch.Tensor, inverse: bool=False, full_jacobian: bool=False)->Tuple[torch.Tensor, torch.Tensor]:
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= math.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= math.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, "hidden_channels"):
            unnormalized_widths /= math.sqrt(self.transform_net.hidden_channels)
            unnormalized_heights /= math.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn("Inputs to the softmax are not scaled down: initialization might be bad.")

        if self.tails is None:
            # spline_fn = splines.rational_quadratic_spline
            # spline_kwargs = {}
            return rational_quadratic_spline(
                                                    inputs=inputs,
                                                    unnormalized_widths=unnormalized_widths,
                                                    unnormalized_heights=unnormalized_heights,
                                                    unnormalized_derivatives=unnormalized_derivatives,
                                                    inverse=inverse,
                                                    min_bin_width=self.min_bin_width,
                                                    min_bin_height=self.min_bin_height,
                                                    min_derivative=self.min_derivative,
                                                    full_jacobian=full_jacobian
                                                    )

        else:
            # spline_fn = splines.unconstrained_rational_quadratic_spline
            # spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
            return unconstrained_rational_quadratic_spline(
                                                                    inputs=inputs,
                                                                    unnormalized_widths=unnormalized_widths,
                                                                    unnormalized_heights=unnormalized_heights,
                                                                    unnormalized_derivatives=unnormalized_derivatives,
                                                                    inverse=inverse,
                                                                    min_bin_width=self.min_bin_width,
                                                                    min_bin_height=self.min_bin_height,
                                                                    min_derivative=self.min_derivative,
                                                                    full_jacobian=full_jacobian,
                                                                    tails=self.tails,
                                                                    tail_bound=self.tail_bound
                                                                )

    def _transform_dim_multiplier(self) -> int:
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        else:
            return self.num_bins * 3 + 1
