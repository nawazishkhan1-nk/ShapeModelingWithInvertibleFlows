import numpy as np
import logging

from .utils import ParticlesDataset

logger = logging.getLogger(__name__)


class CorrespondenceParticlesBaseSimulator:
    def __init__(self):
        self.latent_dim = 32
        self.data_dim = 3072

    def is_image(self):
        return False

    def data_dim(self):
        return self.data_dim

    def full_data_dim(self):
        return np.prod(self.data_dim())
    
    def latent_dim(self):
        return self.latent_dim

    def parameter_dim(self):
        raise NotImplementedError

    def log_density(self, x, parameters=None):
        raise NotImplementedError

    def load_dataset(self, dataset_dir, latent_dim=None):
        x = np.load("{}/{}.npy".format(dataset_dir))
        self.data_dim = x.shape[-1]
        if latent_dim is not None:
            self.latent_dim = latent_dim
        else:
            self.latent_dim = x.shape[0]
        return ParticlesDataset(x)

    def sample(self, n, parameters=None):
        raise NotImplementedError

    def sample_with_noise(self, n, noise, parameters=None):
        x = self.sample(n, parameters)
        x = x + np.random.normal(loc=0.0, scale=noise, size=(n, self.data_dim()))
        return x

    def sample_ood(self, n, parameters=None):
        raise NotImplementedError

    def distance_from_manifold(self, x):
        raise NotImplementedError

    def default_parameters(self, true_param_id=0):
        return np.zeros(self.parameter_dim())

    def eval_parameter_grid(self, resolution=11):
        raise NotImplementedError

    def sample_from_prior(self, n):
        raise NotImplementedError

    def evaluate_log_prior(self, parameters):
        raise NotImplementedError

    def _download(self, dataset_dir):
        raise NotImplementedError

class CorrespondenceParticlesLoader(CorrespondenceParticlesBaseSimulator):
    def __init__(self):
        super().__init__()

    def is_image(self):
        return False

    def data_dim(self):
        return self.data_dim

    def latent_dim(self):
        return self.latent_dim

    def parameter_dim(self):
        return None

    def sample(self, n, parameters=None):
        raise NotImplementedError

    def sample_ood(self, n, parameters=None):
        raise NotImplementedError

    def log_density(self, x, parameters=None):
        raise NotImplementedError

    def distance_from_manifold(self, x):
        raise NotImplementedError

    def sample_from_prior(self, n):
        raise NotImplementedError

    def evaluate_log_prior(self, parameters):
        raise NotImplementedError

