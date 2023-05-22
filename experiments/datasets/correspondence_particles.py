import numpy as np
import logging
import glob

from .utils import ParticlesDataset

logger = logging.getLogger(__name__)


class CorrespondenceParticlesBaseSimulator:
    def __init__(self):
        self.latent_dimension = 0
        self.data_dimension = 0
        self.scale_factor = 0

    def is_image(self):
        return False

    def data_dim(self):
        return self.data_dimension

    def full_data_dim(self):
        return np.prod(self.data_dim())
    
    def latent_dim(self):
        return self.latent_dimension

    def parameter_dim(self):
        raise NotImplementedError

    def log_density(self, x, parameters=None):
        raise NotImplementedError

    def load_dataset(self, dataset_dir, use_augmented_data=False, latent_dim=None, scaledata=False):
        file_ar = glob.glob(f'{dataset_dir}/*.npy') if not use_augmented_data else glob.glob(f'{dataset_dir}/aug/*.npy')
        assert len(file_ar) > 0
        x = np.load(file_ar[0])
        if scaledata:
            factor = max(np.abs(np.max(x)), np.abs(np.min(x)))
            x = (1/factor) * x
            with open(f'{dataset_dir}/scale_factor.txt', 'w') as f:
                f.write(f'Scaling factor = {factor}\n')
            print(f'Scaling done | x shape = {x.shape}')
        self.data_dimension = int(x.shape[-1])
        if latent_dim is not None:
            self.latent_dimension = latent_dim
        else:
            self.latent_dimension = x.shape[0]
        print(f'data dim = {self.data_dim()} latentdim = {self.latent_dim()} | scaling factor = {factor}')
        self.scale_factor = factor
        return ParticlesDataset(x)
    
    def scaling_factor(self):
        return self.scale_factor

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
        return self.data_dimension

    def latent_dim(self):
        return self.latent_dimension

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

