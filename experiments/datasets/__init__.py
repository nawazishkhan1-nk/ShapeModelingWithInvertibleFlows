import logging
# from .base import IntractableLikelihoodError, DatasetNotAvailableError
from .correspondence_particles import CorrespondenceParticlesLoader

logger = logging.getLogger(__name__)

SIMULATORS = ["particles"]

def load_simulator(args):
    assert args.dataset in SIMULATORS
    if args.dataset == "particles":
        simulator = CorrespondenceParticlesLoader()
    else:
        raise ValueError("Unknown dataset {}".format(args.dataset))

    # args.datadim = simulator.data_dim()
    return simulator
