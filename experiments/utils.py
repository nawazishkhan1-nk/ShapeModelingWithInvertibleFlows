import os
import logging
import numpy as np
import math
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


logger = logging.getLogger(__name__)


def create_filename(type_, label, args):
    run_label = "_run{}".format(args.i) if args.i > 0 else ""

    if type_ == "dataset":  # Fixed datasets
        filename = "{}/experiments/data/samples/{}".format(args.dir, args.dataset)

    elif type_ == "sample":  # Dynamically sampled from simulator
        filename = "{}/experiments/data/samples/{}/{}{}.npy".format(args.dir, args.dataset, label, run_label)

    elif type_ == "model":
        filename = "{}/experiments/data/models/{}.pt".format(args.dir, args.modelname)

    elif type_ == "checkpoint":
        filename = "{}/experiments/data/models/checkpoints/{}_{}_{}.pt".format(args.dir, args.modelname, "epoch" if label is None else "epoch_" + label, "{}")

    elif type_ == "resume":
        for label in ["D_", "C_", "B_", "A_", ""]:
            filename = "{}/experiments/data/models/checkpoints/{}_epoch_{}last.pt".format(args.dir, args.modelname, label, "last")
            if os.path.exists(filename):
                return filename

        raise FileNotFoundError(f"Trying to resume training from {filename}, but file does not exist")

    elif type_ == "training_plot":
        filename = "{}/experiments/figures/training/{}_{}_{}.pdf".format(args.dir, args.modelname, "epoch" if label is None else label, "{}")

    elif type_ == "learning_curve":
        filename = "{}/experiments/data/learning_curves/{}.npy".format(args.dir, args.modelname)

    elif type_ == "results":
        trueparam_name = "" if args.trueparam is None or args.trueparam == 0 else "_trueparam{}".format(args.trueparam)
        filename = "{}/experiments/data/results/{}_{}{}.npy".format(args.dir, args.modelname, label, trueparam_name)

    elif type_ == "mcmcresults":
        trueparam_name = "" if args.trueparam is None or args.trueparam == 0 else "_trueparam{}".format(args.trueparam)
        chain_name = "_chain{}".format(args.chain) if args.chain > 0 else ""
        filename = "{}/experiments/data/results/{}_{}{}{}.npy".format(args.dir, args.modelname, label, trueparam_name, chain_name)

    elif type_ == "timing":
        filename = "{}/experiments/data/timing/{}_{}_{}_{}_{}_{}{}.npy".format(
            args.dir,
            args.algorithm,
            args.outerlayers,
            args.outertransform,
            "mlp" if args.outercouplingmlp else "resnet",
            args.outercouplinglayers,
            args.outercouplinghidden,
            run_label,
        )
    elif type_ == "paramscan":
        filename = "{}/experiments/data/paramscan/{}.pickle".format(args.dir, args.paramscanstudyname)
    else:
        raise NotImplementedError

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    return filename


def create_modelname(args):
    run_label = "_run{}".format(args.i) if args.i > 0 else ""
    appendix = "" if args.modelname is None else "_" + args.modelname

    try:
        if args.truth:
            if args.dataset in ["spherical_gaussian", "conditional_spherical_gaussian"]:
                args.modelname = "truth_{}_{}_{}_{:.3f}{}{}".format(args.dataset, args.truelatentdim, args.datadim, args.epsilon, appendix, run_label)
            else:
                args.modelname = "truth_{}{}{}".format(args.dataset, appendix, run_label)
            return
    except:
        pass

    if args.dataset in ["spherical_gaussian", "conditional_spherical_gaussian"]:
        args.modelname = "{}{}_{}_{}_{}_{}_{:.3f}{}{}".format(
            args.algorithm, "_specified" if args.specified else "", args.modellatentdim, args.dataset, args.truelatentdim, args.datadim, args.epsilon, appendix, run_label,
        )
    else:
        args.modelname = "{}{}_{}_{}{}{}".format(args.algorithm, "_specified" if args.specified else "", args.modellatentdim, args.dataset, appendix, run_label)


def nat_to_bit_per_dim(dim):
    if isinstance(dim, (tuple, list, np.ndarray)):
        dim = math.prod(dim)
    logger.debug("Nat to bit per dim: factor %s", 1.0 / (math.log(2) * dim))
    return 1.0 / (math.log(2) * dim)


def sum_except_batch(x: torch.Tensor, num_batch_dims: int=1):
    reduce_dims = list(range(num_batch_dims, x.dim()))
    return torch.sum(x, dim=reduce_dims)


def array_to_image_folder(data, folder):
    for i, x in enumerate(data):
        x = np.clip(np.transpose(x, [1, 2, 0]) / 256.0, 0.0, 1.0)
        if i == 0:
            logger.debug("x: %s", x)
        plt.imsave(f"{folder}/{i}.jpg", x)


def create_dataloader(dataset, validation_split, batch_size, n_workers=4):
        logger.debug("Setting up dataloaders with %s workers", n_workers)

        if validation_split is None or validation_split <= 0.0:
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                # pin_memory=self.run_on_gpu,
                num_workers=n_workers,
            )
            val_loader = None

        else:
            assert 0.0 < validation_split < 1.0, "Wrong validation split: {}".format(validation_split)

            n_samples = len(dataset)
            indices = list(range(n_samples))
            split = int(math.floor(validation_split * n_samples))
            np.random.shuffle(indices)
            train_idx, valid_idx = indices[split:], indices[:split]

            logger.debug("Training partition indices: %s...", train_idx[:10])
            logger.debug("Validation partition indices: %s...", valid_idx[:10])

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(valid_idx)

            train_loader = DataLoader(
                dataset,
                sampler=train_sampler,
                batch_size=batch_size,
                # pin_memory=self.run_on_gpu,
                num_workers=n_workers,
            )
            val_loader = DataLoader(
                dataset,
                sampler=val_sampler,
                batch_size=batch_size,
                # pin_memory=self.run_on_gpu,
                num_workers=n_workers,
            )

        return train_loader, val_loader