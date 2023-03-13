from typing import Tuple, List, Optional
import torch
import logging
from torch import nn
import math


from manifold_flow.utils import various
from manifold_flow.utils.various import product
# from manifold_flow import distributions, transforms
from manifold_flow import transforms
# from manifold_flow.flows import BaseFlow

logger = logging.getLogger(__name__)


def compute_jacobian(inputs: torch.Tensor, output: torch.Tensor)->torch.Tensor:
        """
        :param inputs: B X dM 
        :param output: B X L
        :return: jacobian: B X L X dM
        """
        assert inputs.requires_grad

        num_classes = output.size()[1]

        jacobian = torch.zeros(num_classes, *inputs.size())
        print(f"jacb size {jacobian.shape}")
        grad_output = torch.zeros(*output.size())
        if inputs.is_cuda:
            grad_output = grad_output.cuda()
            jacobian = jacobian.cuda()

        for i in range(num_classes):
            if inputs.grad is not None:
                inputs.grad.zero_()
            grad_output.zero_()
            grad_output[:, i] = 1
            output.backward(grad_output, retain_graph=True)
            jacobian[i] = inputs.grad.data

        jac = torch.transpose(jacobian, dim0=0, dim1=1)
        return jac


# Jit Compatible Distributions
class StandardNormalModified(nn.Module):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape: List[int]):
        super().__init__()
        self._shape = torch.Size(shape)
        shape_prod = 1
        for x in shape:
            shape_prod *= x
        self._log_z = 0.5 * shape_prod * math.log(2 * math.pi)

    def log_prob(self, inputs: torch.Tensor, context: Optional[torch.Tensor]=None)-> torch.Tensor:
        """Calculate log probability under the distribution.

        Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.

        Returns:
            A Tensor of shape [input_size], the log probability of the inputs given the context.
        """
        inputs = torch.as_tensor(inputs)
        if context is not None:
            context = torch.as_tensor(context)
            if inputs.shape[0] != context.shape[0]:
                raise ValueError("Number of input items must be equal to number of context items.")
        return self._log_prob(inputs, context)

    def _log_prob(self, inputs: torch.Tensor, context: torch.Tensor)-> torch.Tensor:
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError("Expected input of shape {}, got {}".format(self._shape, inputs.shape[1:]))
        neg_energy = -0.5 * various.sum_except_batch(inputs ** 2, num_batch_dims=1)
        return neg_energy - self._log_z

    def _sample(self, num_samples: int, context: torch.Tensor)-> torch.Tensor:
        if context is None:
            return torch.randn(num_samples, *self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            context_size = context.shape[0]
            samples = torch.randn(context_size * num_samples, *self._shape)
            return various.split_leading_dim(samples, [context_size, num_samples])


    def sample(self, num_samples: int, context: Optional[torch.Tensor]=None, batch_size: int=None)-> torch.Tensor:
        """Generates samples from the distribution. Samples can be generated in batches.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored.
            batch_size: int or None, number of samples per batch. If None, all samples are generated
                in one batch.

        Returns:
            A Tensor containing the samples, with shape [num_samples, ...] if context is None, or
            [context_size, num_samples, ...] if context is given.
        """
        if not (isinstance(num_samples, int) and num_samples > 0):
            raise TypeError("Number of samples must be a positive integer.")

        if context is not None:
            context = torch.as_tensor(context)

        if batch_size is None:
            return self._sample(num_samples, context)

        else:
            if not (isinstance(batch_size, int) and batch_size > 0):
                raise TypeError("Batch size must be a positive integer.")

            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, context) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, context))
            return torch.cat(samples, dim=0)

    def _mean(self, context: torch.Tensor)-> torch.Tensor:
        if context is None:
            return torch.zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return torch.zeros(context.shape[0], *self._shape)

    def mean(self, context: Optional[torch.Tensor]=None)-> torch.Tensor:
        if context is not None:
            context = torch.as_tensor(context)
        return self._mean(context)
    
    def sample_and_log_prob(self, num_samples: int, context: Optional[torch.Tensor]=None)-> Tuple[torch.Tensor, torch.Tensor]:
        """Generates samples from the distribution together with their log probability.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored.

        Returns:
            A tuple of:
                * A Tensor containing the samples, with shape [num_samples, ...] if context is None,
                  or [context_size, num_samples, ...] if context is given.
                * A Tensor containing the log probabilities of the samples, with shape
                  [num_samples, ...] if context is None, or [context_size, num_samples, ...] if
                  context is given.
        """
        samples = self.sample(num_samples, context=context)

        if context is not None:
            # Merge the context dimension with sample dimension in order to call log_prob.
            samples = various.merge_leading_dims(samples, num_dims=2)
            context = various.repeat_rows(context, num_reps=num_samples)
            assert samples.shape[0] == context.shape[0]

        log_prob = self.log_prob(samples, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = various.split_leading_dim(samples, shape=[-1, num_samples])
            log_prob = various.split_leading_dim(log_prob, shape=[-1, num_samples])

        return samples, log_prob



class RescaledNormalModified(nn.Module):
    """A multivariate Normal with zero mean and a diagonal covariance that is epsilon^2 along each diagonal entry of the matrix."""

    def __init__(self, shape: List[int], std: float=1.0, clip: float=10.0):
        super().__init__()
        self._shape = torch.Size(shape)
        self.std = std
        self._clip = clip
        shape_prod = 1
        for x in shape:
            shape_prod *= x
        self._log_z = 0.5 * shape_prod * math.log(2 * math.pi) + shape_prod * math.log(self.std)

    def _log_prob(self, inputs: torch.Tensor, context: torch.Tensor)-> torch.Tensor:
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError("Expected input of shape {}, got {}".format(self._shape, inputs.shape[1:]))
        if self._clip is not None:
            inputs = torch.clamp(inputs, -self._clip, self._clip)
        neg_energy = -0.5 * various.sum_except_batch(inputs ** 2, num_batch_dims=1) / self.std ** 2
        return neg_energy - self._log_z

    def _sample(self, num_samples: int, context: torch.Tensor)-> torch.Tensor:
        if context is None:
            return self.std * torch.randn(num_samples, *self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            context_size = context.shape[0]
            samples = self.std * torch.randn(context_size * num_samples, *self._shape)
            return various.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context: torch.Tensor)-> torch.Tensor:
        if context is None:
            return torch.zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return torch.zeros(context.shape[0], *self._shape)
    
    def mean(self, context: Optional[torch.Tensor]=None)->torch.Tensor:
        if context is not None:
            context = torch.as_tensor(context)
        return self._mean(context)

    def sample(self, num_samples: int, context: Optional[torch.Tensor]=None, batch_size: int=None)-> torch.Tensor:
        """Generates samples from the distribution. Samples can be generated in batches.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored.
            batch_size: int or None, number of samples per batch. If None, all samples are generated
                in one batch.

        Returns:
            A Tensor containing the samples, with shape [num_samples, ...] if context is None, or
            [context_size, num_samples, ...] if context is given.
        """
        if not (isinstance(num_samples, int) and num_samples > 0):
            raise TypeError("Number of samples must be a positive integer.")

        if context is not None:
            context = torch.as_tensor(context)

        if batch_size is None:
            return self._sample(num_samples, context)

        else:
            if not(isinstance(batch_size, int) and batch_size > 0):
                raise TypeError("Batch size must be a positive integer.")

            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, context) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, context))
            return torch.cat(samples, dim=0)

    def sample_and_log_prob(self, num_samples: int, context: Optional[torch.Tensor]=None)-> Tuple[torch.Tensor, torch.Tensor]:
        """Generates samples from the distribution together with their log probability.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored.

        Returns:
            A tuple of:
                * A Tensor containing the samples, with shape [num_samples, ...] if context is None,
                  or [context_size, num_samples, ...] if context is given.
                * A Tensor containing the log probabilities of the samples, with shape
                  [num_samples, ...] if context is None, or [context_size, num_samples, ...] if
                  context is given.
        """
        samples = self.sample(num_samples, context=context)

        if context is not None:
            # Merge the context dimension with sample dimension in order to call log_prob.
            samples = various.merge_leading_dims(samples, num_dims=2)
            context = various.repeat_rows(context, num_reps=num_samples)
            assert samples.shape[0] == context.shape[0]

        log_prob = self.log_prob(samples, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = various.split_leading_dim(samples, shape=[-1, num_samples])
            log_prob = various.split_leading_dim(log_prob, shape=[-1, num_samples])

        return samples, log_prob

    def log_prob(self, inputs: torch.Tensor, context: Optional[torch.Tensor]=None)-> torch.Tensor:
        """Calculate log probability under the distribution.

        Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.

        Returns:
            A Tensor of shape [input_size], the log probability of the inputs given the context.
        """
        inputs = torch.as_tensor(inputs)
        if context is not None:
            context = torch.as_tensor(context)
            if inputs.shape[0] != context.shape[0]:
                raise ValueError("Number of input items must be equal to number of context items.")
        return self._log_prob(inputs, context)


# Jit Compatible Projections
class ProjectionSplit(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_dim_total = product(input_dim)
        self.output_dim_total = product(output_dim)
        self.mode_in = "vector" if isinstance(input_dim, int) else "image"
        self.mode_out = "vector" if isinstance(input_dim, int) else "image"

        logger.debug("Set up projection from %s with dimension %s to %s with dimension %s", self.mode_in, self.input_dim, self.mode_out, self.output_dim)

        assert self.input_dim_total >= self.output_dim_total, "Input dimension has to be larger than output dimension"

    def forward(self, inputs: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode_in == "vector" and self.mode_out == "vector":
            u = inputs[:, : self.output_dim]
            rest = inputs[:, self.output_dim :]
        elif self.mode_in == "image" and self.mode_out == "vector":
            h = inputs.view(inputs.size(0), -1)
            u = h[:, : self.output_dim]
            rest = h[:, self.output_dim :]
        else:
            raise NotImplementedError("Unsuppoorted projection modes {}, {}".format(self.mode_in, self.mode_out))
        return u, rest

    def inverse(self, inputs: torch.Tensor, orthogonal_inputs: torch.Tensor=None)-> torch.Tensor:
        if orthogonal_inputs is None:
            orthogonal_inputs = torch.zeros(inputs.size(0), self.input_dim_total - self.output_dim)
        # if self.mode_in == "vector" and self.mode_out == "vector":
        x = torch.cat((inputs, orthogonal_inputs), dim=1)
        # elif self.mode_in == "image" and self.mode_out == "vector":
        #     c, h, w = self.input_dim
        #     x = torch.cat((inputs, orthogonal_inputs), dim=1)
        #     x = x.view(inputs.size(0), c, h, w)
        # else:
            # raise NotImplementedError("Unsuppoorted projection modes {}, {}".format(self.mode_in, self.mode_out))
        return x


class ManifoldFlow(nn.Module):
    """ Manifold-based flow (base class for FOM, M-flow, PIE) """

    def __init__(self, data_dim: int, latent_dim: int, outer_transform, inner_transform=None, pie_epsilon: float=1.0e-2, apply_context_to_outer: bool=True, clip_pie: bool=False):
        super(ManifoldFlow, self).__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.apply_context_to_outer = apply_context_to_outer
        self.total_data_dim = product(data_dim)
        self.total_latent_dim = product(latent_dim)

        assert self.total_latent_dim < self.total_data_dim

        self.manifold_latent_distribution = StandardNormalModified((self.total_latent_dim,))
        self.orthogonal_latent_distribution = RescaledNormalModified(
            (self.total_data_dim - self.total_latent_dim,), std=pie_epsilon, clip=None if not clip_pie else clip_pie * pie_epsilon
        )
        self.projection = ProjectionSplit(self.total_data_dim, self.total_latent_dim)

        self.outer_transform = outer_transform
        # if inner_transform is None:
        #     self.inner_transform = transforms.IdentityTransform()
        # else:
        #     print("Not None inner transform")
        self.inner_transform = inner_transform

        self._report_model_parameters()
        self.mode = None

    def set_mode(self, val: str): 
        self.mode = val

    def forward(self, x: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transforms data point to latent space, evaluates likelihood, and transforms it back to data space.

        mode can be "mf" (calculating the exact manifold density based on the full Jacobian), "pie" (calculating the density in x), "slice"
        (calculating the density on x, but projected onto the manifold), or "projection" (calculating no density at all).
        """
        assert self.mode is not None
        mode = self.mode
        # context: torch.Tensor=None
        context = None
        return_hidden: bool=False
        assert mode in ["mf", "pie", "slice", "projection", "pie-inv", "mf-fixed-manifold"]

        if mode == "mf" and not x.requires_grad:
            x.requires_grad = True

        # Encode
        u, h_manifold, h_orthogonal, log_det_outer, log_det_inner = self._encode(x, context)

        # Decode
        x_reco, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer, h_manifold_reco = self._decode(u, mode=mode, context=context)

        # Log prob
        log_prob = self._log_prob(mode, u, h_orthogonal, log_det_inner, log_det_outer, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer)

        # if return_hidden:
        #     return x_reco, log_prob, u, torch.cat((h_manifold, h_orthogonal), -1)
        return x_reco, log_prob, u

    @torch.jit.export
    def forward_for_cpp(self, x: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.mode is not None
        mode = self.mode
        context = None
        return_hidden: bool=False
        assert mode in ["mf", "pie", "slice", "projection", "pie-inv", "mf-fixed-manifold"]

        # Encode
        u, h_manifold, h_orthogonal, log_det_outer, log_det_inner = self._encode(x, context)

        # Decode
        # x_reco, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer, h_manifold_reco = self._decode_for_cpp(u)
        x_reco, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer, h_manifold_reco = self._decode(u, mode=mode, context=context)


        # Log prob
        log_prob_u, log_det_g, log_det_j = self.log_prob_for_cpp(mode, u, h_orthogonal, log_det_inner, log_det_outer, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer)
        jaco = compute_jacobian(inputs=x, output=u)
        return u, log_prob_u, log_det_g, log_det_j, jaco

    
    @torch.jit.export
    def inverse_for_cpp(self, x: torch.Tensor)-> torch.Tensor:
        assert self.mode is not None
        mode = "projection"
        context = None
        return_hidden: bool=False
        assert mode in ["mf", "pie", "slice", "projection", "pie-inv", "mf-fixed-manifold"]

        # Encode
        # u, h_manifold, h_orthogonal, log_det_outer, log_det_inner = self._encode(x, context)

        # Decode
        x_reco, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer, h_manifold_reco = self._decode(x, mode=mode, context=context)

        return x_reco


    def forward_old(self, x: torch.Tensor, mode: str="mf", context: Optional[torch.Tensor]=None, return_hidden: bool=False)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transforms data point to latent space, evaluates likelihood, and transforms it back to data space.

        mode can be "mf" (calculating the exact manifold density based on the full Jacobian), "pie" (calculating the density in x), "slice"
        (calculating the density on x, but projected onto the manifold), or "projection" (calculating no density at all).
        """

        assert mode in ["mf", "pie", "slice", "projection", "pie-inv", "mf-fixed-manifold"]

        if mode == "mf" and not x.requires_grad:
            x.requires_grad = True

        # Encode
        u, h_manifold, h_orthogonal, log_det_outer, log_det_inner = self._encode(x, context)

        # Decode
        x_reco, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer, h_manifold_reco = self._decode(u, mode=mode, context=context)

        # Log prob
        log_prob = self._log_prob(mode, u, h_orthogonal, log_det_inner, log_det_outer, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer)

        # if return_hidden:
        #     return x_reco, log_prob, u, torch.cat((h_manifold, h_orthogonal), -1)
        return x_reco, log_prob, u

    def encode(self, x: torch.Tensor, context: Optional[torch.Tensor]=None)-> torch.Tensor:
        """ Transforms data point to latent space. """

        u, _, _, _, _ = self._encode(x, context=context)
        return u

    def decode(self, u :torch.Tensor, u_orthogonal: torch.Tensor=None, context: Optional[torch.Tensor]=None)-> torch.Tensor:
        """ Decodes latent variable to data space."""

        x, _, _, _, _ = self._decode(u, mode="projection", u_orthogonal=u_orthogonal, context=context)
        return x

    def log_prob(self, x: torch.Tensor, mode: str="mf", context: Optional[torch.Tensor]=None)->torch.Tensor:
        """ Evaluates log likelihood for given data point."""

        return self.forward(x, mode, context)[1]

    def sample(self, u: torch.Tensor=None, n: int=1, context: Optional[torch.Tensor]=None, sample_orthogonal: bool=False)-> torch.Tensor:
        """
        Generates samples from model.

        Note: this is PIE / MF sampling! Cannot sample from slice of PIE efficiently.
        """

        if u is None:
            u = self.manifold_latent_distribution.sample(n, context=None)
        u_orthogonal = self.orthogonal_latent_distribution.sample(n, context=None) if sample_orthogonal else None
        x = self.decode(u, u_orthogonal=u_orthogonal, context=context)
        return x

    def _encode(self, x: torch.Tensor, context: Optional[torch.Tensor]=None)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode
        h, log_det_outer = self.outer_transform(x, full_jacobian=False, context=context if self.apply_context_to_outer else None)
        h_manifold, h_orthogonal = self.projection(h)
        u, log_det_inner = self.inner_transform(h_manifold, full_jacobian=False, context=context)

        return u, h_manifold, h_orthogonal, log_det_outer, log_det_inner

    # @torch.jit.export
    def _decode_for_cpp(self, u: torch.Tensor)-> torch.Tensor:
        # if mode == "mf" and not u.requires_grad:
        #     u.requires_grad = True

        h, inv_log_det_inner = self.inner_transform.inverse(u, full_jacobian=False, context=None)


        
        h = self.projection.inverse(h)

        # if mode in ["pie", "slice", "projection", "mf-fixed-manifold"]:
        x, inv_log_det_outer = self.outer_transform.inverse(h, full_jacobian=False, context=None)
        # inv_jacobian_outer = None
        return x


    # @torch.jit.export
    # def meta_decode_for_cpp(self, u: torch.Tensor)-> torch.Tensor:
    #     # if mode == "mf" and not u.requires_grad:
    #     #     u.requires_grad = True

    #     h, inv_log_det_inner = self.inner_transform.inverse(u, full_jacobian=False, context=None)


        
    #     h = self.projection.inverse(h)

    #     # if mode in ["pie", "slice", "projection", "mf-fixed-manifold"]:
    #     x, inv_log_det_outer = self.outer_transform.inverse(h, full_jacobian=False, context=None)
    #     # inv_jacobian_outer = None
    #     return x

    def _decode(self, u: torch.Tensor, mode: str, u_orthogonal: torch.Tensor=None, context: Optional[torch.Tensor]=None)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode == "mf" and not u.requires_grad:
            u.requires_grad = True

        h, inv_log_det_inner = self.inner_transform.inverse(u, full_jacobian=False, context=context)

        if u_orthogonal is not None:
            h = self.projection.inverse(h, orthogonal_inputs=u_orthogonal)
        else:
            h = self.projection.inverse(h)

        if mode in ["pie", "slice", "projection", "mf-fixed-manifold"]:
            x, inv_log_det_outer = self.outer_transform.inverse(h, full_jacobian=False, context=context if self.apply_context_to_outer else None)
            inv_jacobian_outer = None
        else:
            x, inv_jacobian_outer = self.outer_transform.inverse(h, full_jacobian=True, context=context if self.apply_context_to_outer else None)
            inv_log_det_outer = None

        return x, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer, h

    def _log_prob(self, mode: str, u: torch.Tensor, h_orthogonal: torch.Tensor, log_det_inner: torch.Tensor, log_det_outer: torch.Tensor, inv_log_det_inner: torch.Tensor, inv_log_det_outer: torch.Tensor, inv_jacobian_outer: torch.Tensor)-> torch.Tensor:
        if mode == "pie":
            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob + self.orthogonal_latent_distribution._log_prob(h_orthogonal, context=None)
            log_prob = log_prob + log_det_outer + log_det_inner

        elif mode == "pie-inv":
            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob + self.orthogonal_latent_distribution._log_prob(h_orthogonal, context=None)
            log_prob = log_prob - inv_log_det_outer - inv_log_det_inner

        elif mode == "slice":
            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob + self.orthogonal_latent_distribution._log_prob(torch.zeros_like(h_orthogonal), context=None)
            log_prob = log_prob - inv_log_det_outer - inv_log_det_inner

        elif mode == "mf":
            # inv_jacobian_outer is dx / du, but still need to restrict this to the manifold latents
            inv_jacobian_outer = inv_jacobian_outer[:, :, : self.latent_dim]
            # And finally calculate log det (J^T J)
            jtj = torch.bmm(torch.transpose(inv_jacobian_outer, -2, -1), inv_jacobian_outer)

            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob - 0.5 * torch.slogdet(jtj)[1] - inv_log_det_inner

        elif mode == "mf-fixed-manifold":
            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob + self.orthogonal_latent_distribution._log_prob(h_orthogonal, context=None)
            log_prob = log_prob + log_det_outer + log_det_inner

        else:
            log_prob = None

        return log_prob

    
    def log_prob_for_cpp(self, mode: str, u: torch.Tensor, h_orthogonal: torch.Tensor, log_det_inner: torch.Tensor, log_det_outer: torch.Tensor, inv_log_det_inner: torch.Tensor, inv_log_det_outer: torch.Tensor, inv_jacobian_outer: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert mode == "mf-fixed-manifold"
        log_prob_u = self.manifold_latent_distribution._log_prob(u, context=None)
        log_prob = log_prob_u + self.orthogonal_latent_distribution._log_prob(h_orthogonal, context=None)
        log_prob = log_prob + log_det_outer + log_det_inner
        return log_prob_u, log_det_outer, log_det_inner

    def _report_model_parameters(self):
        """ Reports the model size """

        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size = all_params * (32 / 8)  # Bytes
        logger.info("Model has %.1f M parameters (%.1f M trainable) with an estimated size of %.1f MB", all_params / 1e6, trainable_params / 1.0e6, size / 1.0e6)

        inner_params = sum(p.numel() for p in self.inner_transform.parameters())
        outer_params = sum(p.numel() for p in self.outer_transform.parameters())
        logger.info("  Outer transform: %.1f M parameters", outer_params / 1.0e06)
        logger.info("  Inner transform: %.1f M parameters", inner_params / 1.0e06)
