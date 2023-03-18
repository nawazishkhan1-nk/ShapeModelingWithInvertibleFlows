from torch.nn import functional as F
import logging
import torch
from torch import nn
from torch.nn import init
import math
from typing import Tuple, Optional

from manifold_flow.utils import various

logger = logging.getLogger(__name__)


def _apply_transforms_for_householder(inputs: torch.Tensor, q_vectors: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the sequence of transforms parameterized by given q_vectors to inputs.

        Costs O(KDN), where:
        - K is number of transforms
        - D is dimensionality of inputs
        - N is number of inputs

        Args:
            inputs: Tensor of shape [N, D]
            q_vectors: Tensor of shape [K, D]

        Returns:
            A tuple of:
            - A Tensor of shape [N, D], the outputs.
            - A Tensor of shape [N], the log absolute determinants of the total transform.
        """
        squared_norms = torch.sum(q_vectors ** 2, dim=-1)
        outputs = inputs
        for q_vector, squared_norm in zip(q_vectors, squared_norms):
            temp = outputs @ q_vector  # Inner product.
            temp = torch.ger(temp, (2.0 / squared_norm) * q_vector)  # Outer product.
            outputs = outputs - temp
        batch_size = inputs.shape[0]
        logabsdet = torch.zeros(batch_size)
        return outputs, logabsdet

def _permute(inputs: torch.Tensor, permutation: torch.Tensor, dim:int = -1, full_jacobian: bool=False)-> Tuple[torch.Tensor, torch.Tensor]:
        if dim >= inputs.dim():
            raise ValueError("No dimension {} in inputs.".format(dim))
        if inputs.shape[dim] != len(permutation):
            raise ValueError("Dimension {} in inputs must be of size {}.".format(dim, len(permutation)))
        batch_size = inputs.shape[0]
        if full_jacobian:
            # inputs.requires_grad = True
            outputs = torch.index_select(inputs, dim, permutation)

            # The brute force way does not seem to work, not sure why, maybe index_select breaks autodiff
            # jacobian = utils.batch_jacobian(outputs, inputs)

            # timer.timer(start="Jacobian permutation")

            # First build the Jacobian as a 2D matrix
            jacobian = torch.zeros((outputs.size()[dim], inputs.size()[dim]))
            jacobian[permutation, torch.arange(0, len(permutation), 1)] = 1.0

            # Add dummy dimensions for batch size...
            jacobian = jacobian.unsqueeze(0)  # (1, n, n)
            # ... and for every dimension smaller than dim...
            for i in range(dim - 1):
                jacobian = jacobian.unsqueeze(2 + 2 * i)
                jacobian = jacobian.unsqueeze(1 + i)
            # ... and for every dimension larger than dim...
            for i in range(len(inputs.size()) - dim - 1):
                jacobian = jacobian.unsqueeze(1 + 2 * dim + 2 * i)
                jacobian = jacobian.unsqueeze(1 + dim + i)

            # Broadcast to full size
            jacobian = torch.ones(outputs.size() + inputs.size()[1:]) * jacobian

            # Finally, view it as a (batch, n, n) Jacobian
            bs_var = inputs.size()[0]
            n_var = 1
            for x in inputs.size()[1:]:
                n_var *= x
            jacobian = jacobian.view((bs_var, n_var, n_var))

            # logger.debug("Jacobian from permutation: \n %s", jacobian[0])
            # timer.timer(stop="Jacobian permutation")

            return outputs, jacobian
        else:
            # print(f"in permute {inputs.get_device()} | dim = {dim} | permutation {permutation.get_device()}")
            outputs = torch.index_select(inputs, dim, permutation)
            logabsdet = torch.zeros(batch_size)
            return outputs, logabsdet

class RandomPermutation(nn.Module):
    """Permutes inputs on a given dimension using a given permutation."""

    def __init__(self, features: int, dim: int=1):
        super().__init__()
        if not (isinstance(features, int) and features > 0):
            raise ValueError("Number of features must be a positive integer.")
        permutation = torch.randperm(features)
        if permutation.dim() != 1:
            raise ValueError("Permutation must be a 1D tensor.")
        if not (isinstance(dim, int) and dim > 0):
            raise ValueError("dim must be a positive integer.")
        self._dim: int = dim
        self.register_buffer("_permutation", permutation)

    @property
    def _inverse_permutation(self)-> torch.Tensor:
        return torch.argsort(self._permutation)

    def forward(self, inputs: torch.Tensor, context: Optional[torch.Tensor]=None, full_jacobian: bool=False)-> Tuple[torch.Tensor, torch.Tensor]:
        return _permute(inputs, self._permutation, self._dim, full_jacobian=full_jacobian)
    
    @torch.jit.export
    def inverse(self, inputs: torch.Tensor, context: Optional[torch.Tensor]=None, full_jacobian: bool=False)-> Tuple[torch.Tensor, torch.Tensor]:
        return _permute(inputs, self._inverse_permutation, self._dim, full_jacobian=full_jacobian)

class LinearCache(object):
    """Helper class to store the cache of a linear transform.

    The cache consists of: the weight matrix, its inverse and its log absolute determinant.
    """

    def __init__(self):
        self.weight: Optional[torch.Tensor] = None
        self.inverse: Optional[torch.Tensor] = None
        self.logabsdet: Optional[torch.Tensor] = None

    def invalidate(self):
        self.weight = None
        self.inverse = None
        self.logabsdet = None

class HouseholderSequence(nn.Module):
    """A sequence of Householder transforms.

    This class can be used as a way of parameterizing an orthogonal matrix.
    """

    def __init__(self, features: int, num_transforms: int=10):
        """Constructor.

        Args:
            features: int, dimensionality of the input.
            num_transforms: int, number of Householder transforms to use.

        Raises:
            TypeError: if arguments are not the right type.
        """
        super().__init__()
        if not (isinstance(features, int) and features > 0):
            raise TypeError("Number of features must be a positive integer.")
        if not (isinstance(num_transforms, int) and num_transforms > 0):
            raise TypeError("Number of transforms must be a positive integer.")

        self.features: int = features
        self.num_transforms: int = num_transforms
        # TODO: are randn good initial values?
        self.q_vectors = nn.Parameter(torch.randn(num_transforms, features))

    def forward(self, inputs: torch.Tensor, context: Optional[torch.Tensor]=None)-> Tuple[torch.Tensor, torch.Tensor]:
        return _apply_transforms_for_householder(inputs, self.q_vectors)

    def inverse(self, inputs: torch.Tensor, context: Optional[torch.Tensor]=None)-> Tuple[torch.Tensor, torch.Tensor]:
        # Each householder transform is its own inverse, so the total inverse is given by
        # simply performing each transform in the reverse order.
        reverse_idx = torch.arange(self.num_transforms - 1, -1, -1)
        return _apply_transforms_for_householder(inputs, self.q_vectors[reverse_idx])

    def matrix(self)-> torch.Tensor:
        """Returns the orthogonal matrix that is equivalent to the total transform.

        Costs O(KD^2), where:
        - K is number of transforms
        - D is dimensionality of inputs

        Returns:
            A Tensor of shape [D, D].
        """
        identity = torch.eye(self.features, self.features)
        outputs, _ = self.inverse(identity)
        return outputs

class SVDLinear(nn.Module):
    """A linear module using the SVD decomposition for the weight matrix."""

    def __init__(self, features: int, num_householder: int=10, using_cache: bool=False):
        super().__init__()
        if not (isinstance(features, int) and features > 0):
            raise TypeError("Number of features must be a positive integer.")
        self.features = features
        self.bias = nn.Parameter(torch.zeros(features))

        # Caching flag and values.
        self.using_cache = using_cache
        # self.cache = LinearCache()

        # First orthogonal matrix (U).
        self.orthogonal_1 = HouseholderSequence(features=features, num_transforms=num_householder)

        # Logs of diagonal entries of the diagonal matrix (S).
        self.log_diagonal = nn.Parameter(torch.zeros(features))

        # Second orthogonal matrix (V^T).
        self.orthogonal_2 = HouseholderSequence(features=features, num_transforms=num_householder)

    #     self._initialize()

    # def _initialize(self):
        stdv = 1.0 / math.sqrt(self.features)
        init.uniform_(self.log_diagonal, -stdv, stdv)
        init.constant_(self.bias, 0.0)

    def forward(self, inputs: torch.Tensor, context: Optional[torch.tensor]=None, full_jacobian: bool=False)-> Tuple[torch.Tensor, torch.Tensor]:
        # if not self.training and self.using_cache:
        #     self._check_forward_cache()
        #     outputs = F.linear(inputs, self.cache.weight, self.bias)
        #     if full_jacobian:
        #         jacobian = torch.ones(outputs.shape[0]) * self.cache.weight.unsqueeze(0)
        #         return outputs, jacobian
        #     else:
        #         logabsdet = self.cache.logabsdet * torch.ones(outputs.shape[0])
        #         return outputs, logabsdet
        # else:
        return self.forward_no_cache(inputs, full_jacobian=full_jacobian)
    
    def use_cache(self, mode: bool=True):
        if not isinstance(mode, bool):
            raise TypeError("Mode must be boolean.")
        self.using_cache = mode
    
    def inverse(self, inputs: torch.Tensor, context: Optional[torch.tensor]=None, full_jacobian: bool=False)-> Tuple[torch.Tensor, torch.Tensor]:
        # if not self.training and self.using_cache:
        #     self._check_inverse_cache()
        #     outputs = F.linear(inputs - self.bias, self.cache.inverse)
        #     if full_jacobian:
        #         jacobian = torch.ones(outputs.shape[0]) * self.cache.inverse.unsqueeze(0)
        #         return outputs, jacobian
        #     else:
        #         logabsdet = (-self.cache.logabsdet) * torch.ones(outputs.shape[0])
        #         return outputs, logabsdet
        # else:
            return self.inverse_no_cache(inputs, full_jacobian=full_jacobian)

    # def train(self, mode=True):
    #     if mode:
    #         # If training again, invalidate cache.
    #         self.cache.invalidate()
    #     return super().train(mode)

    # def _check_forward_cache(self):
    #     if self.cache.weight is None and self.cache.logabsdet is None:
    #         self.cache.weight, self.cache.logabsdet = self.weight_and_logabsdet()

    #     elif self.cache.weight is None:
    #         self.cache.weight = self.weight()

    #     elif self.cache.logabsdet is None:
    #         self.cache.logabsdet = self.logabsdet()

    # def _check_inverse_cache(self):
    #     if self.cache.inverse is None and self.cache.logabsdet is None:
    #         self.cache.inverse, self.cache.logabsdet = self.weight_inverse_and_logabsdet()

    #     elif self.cache.inverse is None:
    #         self.cache.inverse = self.weight_inverse()

    #     elif self.cache.logabsdet is None:
    #         self.cache.logabsdet = self.logabsdet()


    def forward_no_cache(self, inputs: torch.Tensor, full_jacobian: bool=False)-> Tuple[torch.Tensor, torch.Tensor]:
        """Cost:
            output = O(KDN)
            logabsdet = O(D)
        where:
            K = num of householder transforms
            D = num of features
            N = num of inputs
        """
        if full_jacobian:
            raise NotImplementedError
        outputs, _ = self.orthogonal_2(inputs)  # Ignore logabsdet as we know it's zero.
        outputs *= torch.exp(self.log_diagonal)
        outputs, _ = self.orthogonal_1(outputs)  # Ignore logabsdet as we know it's zero.
        outputs += self.bias

        logabsdet = self.logabsdet() * torch.ones(outputs.shape[0])

        return outputs, logabsdet


    def inverse_no_cache(self, inputs: torch.Tensor, full_jacobian: bool=False)-> Tuple[torch.Tensor, torch.Tensor]:
        """Cost:
            output = O(KDN)
            logabsdet = O(D)
        where:
            K = num of householder transforms
            D = num of features
            N = num of inputs
        """
        if full_jacobian:
            raise NotImplementedError
        outputs = inputs - self.bias
        outputs, _ = self.orthogonal_1.inverse(outputs)  # Ignore logabsdet since we know it's zero.
        outputs *= torch.exp(-self.log_diagonal)
        outputs, _ = self.orthogonal_2.inverse(outputs)  # Ignore logabsdet since we know it's zero.
        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * torch.ones(outputs.shape[0])
        return outputs, logabsdet

    # def weight(self)-> torch.Tensor:
    #     """Cost:
    #         weight = O(KD^2)
    #     where:
    #         K = num of householder transforms
    #         D = num of features
    #     """
    #     diagonal = torch.diag(torch.exp(self.log_diagonal))
    #     weight, _ = self.orthogonal_2.inverse(diagonal)
    #     weight, _ = self.orthogonal_1(weight.t())
    #     return weight.t()

    # def weight_inverse(self)-> torch.Tensor:
    #     """Cost:
    #         inverse = O(KD^2)
    #     where:
    #         K = num of householder transforms
    #         D = num of features
    #     """
    #     diagonal_inv = torch.diag(torch.exp(-self.log_diagonal))
    #     weight_inv, _ = self.orthogonal_1(diagonal_inv)
    #     weight_inv, _ = self.orthogonal_2.inverse(weight_inv.t())
    #     return weight_inv.t()

    def logabsdet(self)-> torch.Tensor:
        """Cost:
            logabsdet = O(D)
        where:
            D = num of features
        """
        return torch.sum(self.log_diagonal)

    # def weight_and_logabsdet(self)-> Tuple[torch.Tensor, torch.Tensor]:
    #     # if it is more efficient to compute the weight matrix
    #     # and its logabsdet together.
    #     return self.weight(), self.logabsdet()

    # def weight_inverse_and_logabsdet(self)-> Tuple[torch.Tensor, torch.Tensor]:
    #     #  if it is more efficient to compute the weight matrix
    #     # inverse and weight matrix logabsdet together.
    #     return self.weight_inverse(), self.logabsdet()