#! /usr/bin/env python

""" Top-level script for training models """

import numpy as np
import logging
import sys
import torch
import configargparse
import copy
from torch import optim
import matplotlib.pyplot as plt
import os
import shutil
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

sys.path.append("../")

from training import losses, callbacks
from training import ForwardTrainer, ConditionalForwardTrainer, SCANDALForwardTrainer, AdversarialTrainer, ConditionalAdversarialTrainer, AlternatingTrainer

from datasets import load_simulator, SIMULATORS
from utils import create_filename, create_modelname, nat_to_bit_per_dim, create_dataloader
from architectures import create_model
from architectures.create_model import ALGORITHMS
from manifold_flow.transforms.normalization import ActNorm

logger = logging.getLogger(__name__)


def parse_args():
    # TODO: Clean arguments
    """ Parses command line arguments for the training """

    parser = configargparse.ArgumentParser()

    # What what what
    parser.add_argument("--modelname", type=str, default=None, help="Model name. Algorithm, latent dimension, dataset, and run are prefixed automatically.")
    parser.add_argument(
        "--algorithm", type=str, default="flow", choices=ALGORITHMS, help="Model: flow (AF), mf (FOM, M-flow), emf (Me-flow), pie (PIE), gamf (M-flow-OT), pae (PAE)...",
    )
    parser.add_argument("--dataset", type=str, default="spherical_gaussian", choices=SIMULATORS, help="Dataset: spherical_gaussian, power, lhc, lhc40d, lhc2d, and some others")
    parser.add_argument("-i", type=int, default=0, help="Run number")
    parser.add_argument("--perturb_data", action="store_true", help="Use perturbed input data")
    parser.add_argument("--scaledata", action="store_true", help="Scale input data")
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU id")

    parser.add_argument("--eval_model", action="store_true", help="Evaluate Model")
    parser.add_argument("--serialize_model", action="store_true", help="Serialize Model to TorchScript")

    # Dataset details
    parser.add_argument("--truelatentdim", type=int, default=2, help="True manifold dimensionality (for datasets where that is variable)")
    parser.add_argument("--datadim", type=int, default=3, help="True data dimensionality (for datasets where that is variable)")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Noise term (for datasets where that is variable)")

    # Model details
    parser.add_argument("--modellatentdim", type=int, default=2, help="Model manifold dimensionality")
    parser.add_argument("--specified", action="store_true", help="Prescribe manifold chart: FOM instead of M-flow")
    parser.add_argument("--outertransform", type=str, default="rq-coupling", help="Scalar base trf. for f: {affine | quadratic | rq}-{coupling | autoregressive}")
    parser.add_argument("--innertransform", type=str, default="rq-coupling", help="Scalar base trf. for h: {affine | quadratic | rq}-{coupling | autoregressive}")
    # TODO: Try without permutation as well
    parser.add_argument("--lineartransform", type=str, default="permutation", help="Scalar linear trf: linear | permutation")
    parser.add_argument("--outerlayers", type=int, default=5, help="Number of transformations in f (not counting linear transformations)")
    parser.add_argument("--innerlayers", type=int, default=5, help="Number of transformations in h (not counting linear transformations)")
    parser.add_argument("--conditionalouter", action="store_true", help="If dataset is conditional, use this to make f conditional (otherwise only h is conditional)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Use dropout")
    parser.add_argument("--pieepsilon", type=float, default=0.01, help="PIE epsilon term")
    parser.add_argument("--pieclip", type=float, default=None, help="Clip v in p(v), in multiples of epsilon")
    parser.add_argument("--encoderblocks", type=int, default=5, help="Number of blocks in Me-flow / PAE encoder")
    parser.add_argument("--encoderhidden", type=int, default=100, help="Number of hidden units in Me-flow / PAE encoder")
    parser.add_argument("--splinerange", default=3.0, type=float, help="Spline boundaries")
    parser.add_argument("--splinebins", default=8, type=int, help="Number of spline bins")
    parser.add_argument("--levels", type=int, default=3, help="Number of levels in multi-scale architectures for image data (for outer transformation f)")
    parser.add_argument("--actnorm", action="store_true", help="Use actnorm in convolutional architecture")
    parser.add_argument("--batchnorm", action="store_true", help="Use batchnorm in ResNets")
    parser.add_argument("--linlayers", type=int, default=2, help="Number of linear layers before the projection for M-flow and PIE on image data")
    parser.add_argument("--linchannelfactor", type=int, default=2, help="Determines number of channels in linear trfs before the projection for M-flow and PIE on image data")
    parser.add_argument("--intermediatensf", action="store_true", help="Use NSF rather than linear layers before projecting (for M-flows and PIE on image data)")
    parser.add_argument("--decoderblocks", type=int, default=5, help="Number of blocks in PAE encoder")
    parser.add_argument("--decoderhidden", type=int, default=100, help="Number of hidden units in PAE encoder")

    # Training
    parser.add_argument("--alternate", action="store_true", help="Use alternating M/D training algorithm")
    parser.add_argument("--sequential", action="store_true", help="Use sequential M/D training algorithm")
    parser.add_argument("--load", type=str, default=None, help="Model name to load rather than training from scratch, run is affixed automatically")
    parser.add_argument("--startepoch", type=int, default=0, help="Sets the first trained epoch for resuming partial training")
    
    # None
    parser.add_argument("--samplesize", type=int, default=None, help="If not None, number of samples used for training")
    
    # N epochs = 50
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--subsets", type=int, default=1, help="Number of subsets per epoch in an alternating training")
    parser.add_argument("--batchsize", type=int, default=20, help="Batch size for everything except OT training")
    
    # unused
    parser.add_argument("--genbatchsize", type=int, default=1000, help="Batch size for OT training")
    
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Initial learning rate")
    parser.add_argument("--msefactor", type=float, default=1000.0, help="Reco error multiplier in loss")
    parser.add_argument("--addnllfactor", type=float, default=0.1, help="Negative log likelihood multiplier in loss for M-flow-S training")
    parser.add_argument("--nllfactor", type=float, default=1.0, help="Negative log likelihood multiplier in loss (except for M-flow-S training)")
    parser.add_argument("--sinkhornfactor", type=float, default=10.0, help="Sinkhorn divergence multiplier in loss")
    parser.add_argument("--weightdecay", type=float, default=1.0e-4, help="Weight decay")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient norm clipping parameter")

    # Unused
    parser.add_argument("--nopretraining", action="store_true", help="Skip pretraining in M-flow-S training")
    parser.add_argument("--noposttraining", action="store_true", help="Skip posttraining in M-flow-S training")
    
    # TODO: Change valdiation split
    parser.add_argument("--validationsplit", type=float, default=0.25, help="Fraction of train data used for early stopping")
    parser.add_argument("--scandal", type=float, default=None, help="Activates SCANDAL training and sets prefactor of score MSE in loss")
    parser.add_argument("--l1", action="store_true", help="Use smooth L1 loss rather than L2 (MSE) for reco error")
    parser.add_argument("--uvl2reg", type=float, default=None, help="Add L2 regularization term on the latent variables after the outer flow (M-flow-M/D only)")
    parser.add_argument("--seed", type=int, default=1357, help="Random seed (--i is always added to it)")
    parser.add_argument("--resume", type=int, default=None, help="Resume training at a given epoch (overwrites --load and --startepoch)")

    # Other settings
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")
    parser.add_argument("--dir", type=str, default="/home/sci/nawazish.khan/ShapeModelingWithInvertibleFlows/", help="Base directory of repo")
    parser.add_argument("--dataset_dir", type=str, default="/home/sci/nawazish.khan/nls-data/supershapes/initial-particles-256", help="directory of data")
    parser.add_argument("--scalefactor", type=int, default=-1, help="Sacling Factor of data")
    parser.add_argument("--debug", action="store_true", help="Debug mode (more log output, additional callbacks)")

    args = parser.parse_args()
    return args


def make_training_kwargs(args, dataset):
    kwargs = {
        "dataset": dataset,
        "batch_size": args.batchsize,
        "initial_lr": args.lr,
        "scheduler": optim.lr_scheduler.CosineAnnealingLR,
        "clip_gradient": args.clip,
        "validation_split": args.validationsplit,
        "seed": args.seed + args.i,
    }
    if args.weightdecay is not None:
        kwargs["optimizer_kwargs"] = {"weight_decay": float(args.weightdecay)} #None
    scandal_loss = [losses.score_mse] if args.scandal is not None else [] # None
    scandal_label = ["score MSE"] if args.scandal is not None else [] # None
    scandal_weight = [args.scandal] if args.scandal is not None else [] # None

    return kwargs, scandal_loss, scandal_label, scandal_weight

def train_manifold_flow_alternating(args, dataset, model, simulator):
    """ M-flow-A training """

    assert not args.specified

    trainer1 = ForwardTrainer(model, gpu_id=args.gpu_id) if simulator.parameter_dim() is None else ConditionalForwardTrainer(model, gpu_id=args.gpu_id)
    trainer2 = ForwardTrainer(model, gpu_id=args.gpu_id) if simulator.parameter_dim() is None else ConditionalForwardTrainer(model, gpu_id=args.gpu_id) if args.scandal is None else SCANDALForwardTrainer(model, gpu_id=args.gpu_id)
    metatrainer = AlternatingTrainer(model, trainer1, trainer2)

    meta_kwargs = {"dataset": dataset, "initial_lr": args.lr, "scheduler": optim.lr_scheduler.CosineAnnealingLR, "validation_split": args.validationsplit}
    if args.weightdecay is not None:
        meta_kwargs["optimizer_kwargs"] = {"weight_decay": float(args.weightdecay)}
    _, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)

    phase1_kwargs = {"forward_kwargs": {"mode": "projection"}, "clip_gradient": args.clip}
    phase2_kwargs = {"forward_kwargs": {"mode": "mf-fixed-manifold"}, "clip_gradient": args.clip}

    phase1_parameters = list(model.outer_transform.parameters()) + (list(model.encoder.parameters()) if args.algorithm == "emf" else [])
    phase2_parameters = list(model.inner_transform.parameters())

    logger.info("Starting training MF, alternating between reconstruction error and log likelihood")
    learning_curves_ = metatrainer.train(
        loss_functions=[losses.smooth_l1_loss if args.l1 else losses.mse, losses.nll] + scandal_loss,
        loss_function_trainers=[0, 1] + ([1] if args.scandal is not None else []),
        loss_labels=["L1" if args.l1 else "MSE", "NLL"] + scandal_label,
        loss_weights=[args.msefactor, args.nllfactor * nat_to_bit_per_dim(args.modellatentdim)] + scandal_weight,
        epochs=args.epochs // 2,
        subsets=args.subsets,
        batch_sizes=[args.batchsize, args.batchsize],
        parameters=[phase1_parameters, phase2_parameters],
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args))],
        trainer_kwargs=[phase1_kwargs, phase2_kwargs],
        **meta_kwargs,
    )
    learning_curves = np.vstack(learning_curves_).T

    return learning_curves

def train_generative_adversarial_manifold_flow(args, dataset, model, simulator):
    """ M-flow-OT training """

    gen_trainer = AdversarialTrainer(model) if simulator.parameter_dim() is None else ConditionalAdversarialTrainer(model)
    common_kwargs, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)
    common_kwargs["batch_size"] = args.genbatchsize

    logger.info("Starting training GAMF: Sinkhorn-GAN")

    callbacks_ = [callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args))]
    if args.debug:
        callbacks_.append(callbacks.print_mf_weight_statistics())

    learning_curves_ = gen_trainer.train(
        loss_functions=[losses.make_sinkhorn_divergence()],
        loss_labels=["GED"],
        loss_weights=[args.sinkhornfactor],
        epochs=args.epochs,
        callbacks=callbacks_,
        compute_loss_variance=True,
        initial_epoch=args.startepoch,
        **common_kwargs,
    )

    learning_curves = np.vstack(learning_curves_).T
    return learning_curves


def train_generative_adversarial_manifold_flow_alternating(args, dataset, model, simulator):
    """ M-flow-OTA training """

    assert not args.specified

    gen_trainer = AdversarialTrainer(model) if simulator.parameter_dim() is None else ConditionalAdversarialTrainer(model)
    likelihood_trainer = ForwardTrainer(model) if simulator.parameter_dim() is None else ConditionalForwardTrainer(model) if args.scandal is None else SCANDALForwardTrainer(model)
    metatrainer = AlternatingTrainer(model, gen_trainer, likelihood_trainer)

    meta_kwargs = {"dataset": dataset, "initial_lr": args.lr, "scheduler": optim.lr_scheduler.CosineAnnealingLR, "validation_split": args.validationsplit}
    if args.weightdecay is not None:
        meta_kwargs["optimizer_kwargs"] = {"weight_decay": float(args.weightdecay)}
    _, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)

    phase1_kwargs = {"clip_gradient": args.clip}
    phase2_kwargs = {"forward_kwargs": {"mode": "mf-fixed-manifold"}, "clip_gradient": args.clip}

    phase1_parameters = list(model.parameters())
    phase2_parameters = list(model.inner_transform.parameters())

    logger.info("Starting training GAMF, alternating between Sinkhorn divergence and log likelihood")
    learning_curves_ = metatrainer.train(
        loss_functions=[losses.make_sinkhorn_divergence(), losses.nll] + scandal_loss,
        loss_function_trainers=[0, 1] + [1] if args.scandal is not None else [],
        loss_labels=["GED", "NLL"] + scandal_label,
        loss_weights=[args.sinkhornfactor, args.nllfactor * nat_to_bit_per_dim(args.modellatentdim)] + scandal_weight,
        batch_sizes=[args.genbatchsize, args.batchsize],
        epochs=args.epochs // 2,
        parameters=[phase1_parameters, phase2_parameters],
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args))],
        trainer_kwargs=[phase1_kwargs, phase2_kwargs],
        subsets=args.subsets,
        subset_callbacks=[callbacks.print_mf_weight_statistics()] if args.debug else None,
        **meta_kwargs,
    )
    learning_curves = np.vstack(learning_curves_).T

    return learning_curves

def train_manifold_flow_sequential(args, dataset, model, simulator):
    """ Sequential M-flow-M/D training """

    assert not args.specified

    if simulator.parameter_dim() is None:
        trainer1 = ForwardTrainer(model, gpu_id=args.gpu_id)
        trainer2 = ForwardTrainer(model, gpu_id=args.gpu_id)
    else:
        trainer1 = ConditionalForwardTrainer(model, gpu_id=args.gpu_id)
        if args.scandal is None:
            trainer2 = ConditionalForwardTrainer(model, gpu_id=args.gpu_id)
        else:
            trainer2 = SCANDALForwardTrainer(model, gpu_id=args.gpu_id)

    common_kwargs, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)

    callbacks1 = [callbacks.save_model_after_every_epoch(create_filename("checkpoint", "A", args)), callbacks.print_mf_latent_statistics(), callbacks.print_mf_weight_statistics()]
    callbacks2 = [callbacks.save_model_after_every_epoch(create_filename("checkpoint", "B", args)), callbacks.print_mf_latent_statistics(), callbacks.print_mf_weight_statistics()]
    
    # Not applicable
    if simulator.is_image():
        callbacks1.append(
            callbacks.plot_sample_images(
                create_filename("training_plot", "sample_epoch_A", args), context=None if simulator.parameter_dim() is None else torch.zeros(30, simulator.parameter_dim()),
            )
        )
        callbacks2.append(
            callbacks.plot_sample_images(
                create_filename("training_plot", "sample_epoch_B", args), context=None if simulator.parameter_dim() is None else torch.zeros(30, simulator.parameter_dim()),
            )
        )
        callbacks1.append(callbacks.plot_reco_images(create_filename("training_plot", "reco_epoch_A", args)))
        callbacks2.append(callbacks.plot_reco_images(create_filename("training_plot", "reco_epoch_B", args)))

    logger.info("Starting training MF, phase 1: manifold training")
    learning_curves = trainer1.train(
        loss_functions=[losses.smooth_l1_loss if args.l1 else losses.mse] + ([] if args.uvl2reg is None else [losses.hiddenl2reg]),
        loss_labels=["L1" if args.l1 else "MSE"] + ([] if args.uvl2reg is None else ["L2_lat"]),
        loss_weights=[args.msefactor] + ([] if args.uvl2reg is None else [args.uvl2reg]),
        epochs=args.epochs // 2,
        parameters=list(model.outer_transform.parameters()) + list(model.encoder.parameters()) if args.algorithm == "emf" else list(model.outer_transform.parameters()),
        callbacks=callbacks1,
        forward_kwargs={"mode": "projection", "return_hidden": args.uvl2reg is not None},
        initial_epoch=args.startepoch,
        **common_kwargs,
    )
    learning_curves = np.vstack(learning_curves).T

    logger.info("Starting training MF, phase 2: density training")
    learning_curves_ = trainer2.train(
        loss_functions=[losses.nll] + scandal_loss,
        loss_labels=["NLL"] + scandal_label,
        loss_weights=[args.nllfactor * nat_to_bit_per_dim(args.modellatentdim)] + scandal_weight,
        epochs=args.epochs - (args.epochs // 2),
        parameters=list(model.inner_transform.parameters()),
        callbacks=callbacks2,
        forward_kwargs={"mode": "mf-fixed-manifold"},
        initial_epoch=args.startepoch - args.epochs // 2,
        **common_kwargs,
    )
    learning_curves = np.vstack((learning_curves, np.vstack(learning_curves_).T))

    return learning_curves

def train_model(args, dataset, model, simulator):
    """ Starts appropriate training """
    if args.algorithm in ["mf", "emf"]:
        if args.alternate:
            learning_curves = train_manifold_flow_alternating(args, dataset, model, simulator)
        elif args.sequential:
            learning_curves = train_manifold_flow_sequential(args, dataset, model, simulator)
    elif args.algorithm == "gamf":
        if args.alternate:
            learning_curves = train_generative_adversarial_manifold_flow_alternating(args, dataset, model, simulator)
        else:
            learning_curves = train_generative_adversarial_manifold_flow(args, dataset, model, simulator)
    else:
        raise ValueError("Unknown algorithm %s", args.algorithm)

    return learning_curves

def fix_act_norm_issue(model):
    if isinstance(model, ActNorm):
        logger.debug("Fixing initialization state of actnorm layer")
        model.initialized = True

    for _, submodel in model._modules.items():
        fix_act_norm_issue(submodel)

def run_evaluation(model, dataset, device, eval_results_dir, args):
    train_loader, val_loader = create_dataloader(dataset, args.validationsplit, args.batchsize)
    model.eval()
    model.to(device)
    
    # 2. Generalization
    print(f'Running Generalization')
    recon_dir = f"{eval_results_dir}/inputs_and_reconstructions/"
    os.makedirs(recon_dir, exist_ok=True)
    generalization_errors = []
    for i_batch, batch_data in enumerate(val_loader):
        x = batch_data[0].to(device, torch.float)
        print(f' input vector shape {x.shape} device = {x.get_device()}')
        x_recon, log_prob, u = model(x, mode="mf-fixed-manifold")
        print(f' Reconstructed vector shape {x_recon.shape} | log_prob shape {log_prob.shape} | u shape {u.shape}')
        gen = torch.linalg.norm((x_recon-x)).item()
        generalization_errors.append(gen)
        print(f'GEN = {gen}')
        np.save(f"{recon_dir}/x_batch_{i_batch}.npy", x.detach().numpy())
        np.save(f"{recon_dir}/x_recon_batch_{i_batch}.npy", x_recon.detach().numpy())
        np.save(f"{recon_dir}/log_prob_batch_{i_batch}.npy", log_prob.detach().numpy())
        np.save(f"{recon_dir}/u_batch_{i_batch}.npy", u.detach().numpy())


    gen_val = np.mean(np.array(generalization_errors))
    print(f"mean gen = {gen_val}")

    # 1. Specificity
    print(f'Running Specificty')
    samples = model.sample(n=100, sample_orthogonal=False)
    samples_ = model.sample(n=100, sample_orthogonal=True)
    # SCALE_FACTOR = 399.18096923828125
    SCALE_FACTOR = args.scalefactor
    assert SCALE_FACTOR > 0
    plot_reconstructions( SCALE_FACTOR * samples, f"{eval_results_dir}/without_orthogonal/")
    plot_reconstructions( SCALE_FACTOR * samples_, f"{eval_results_dir}/with_orthogonal/")

    x_ar = []
    for i_batch, batch_data in enumerate(train_loader):
        x_batch = batch_data[0]
        x_ar.append(x_batch)

    x_all = torch.vstack(x_ar)
    N = x_all.shape[0]
    print("Sampling")
    # samples_for_kde = model.sample(n=N, sample_orthogonal=False)
    # np.save(f'{eval_results_dir}/samples.npy', samples_for_kde.detach().numpy())
    # np.save(f'{eval_results_dir}/input_z.npy', x_all.detach().numpy())

    # plot_kde_plots(samples_for_kde, x_all, eval_results_dir)

    specificity_errors = []
    for i in range(100):
        x_sampled = samples[i][None, ...]
        x_sampled = x_sampled.repeat_interleave(N, dim=0)
        errors = torch.mean(torch.sqrt((x_all - x_sampled)**2), dim=0)
        spec = torch.min(errors).item()
        print(f' SPEC without orthogonal = {spec}')
        specificity_errors.append(spec)
    spec_val1 = SCALE_FACTOR * np.mean(np.array(specificity_errors))

    specificity_errors_ = []
    for i in range(100):
        x_sampled = samples_[i][None, ...]
        x_sampled = x_sampled.repeat_interleave(N, dim=0)
        errors = torch.mean(torch.sqrt((x_all - x_sampled)**2), dim=0)
        spec = torch.min(errors).item()
        print(f' SPEC with orthogonal = {spec}')
        specificity_errors_.append(spec)
    spec_val2 = SCALE_FACTOR * np.mean(np.array(specificity_errors_))

    with open(f'{eval_results_dir}/metrics.txt', 'w') as f:
        f.write(f'Generalization = {gen_val} \n Sepecificity w/o orthogonal sampling = {spec_val1} \n Specifictity with orthogonal sampling = {spec_val2} \n Scale Factor = {SCALE_FACTOR}')
    return gen_val, (spec_val1, spec_val2)


def plot_kde_plots(sampled_x, x, out_dir):
    N = sampled_x.shape[0]
    assert sampled_x.shape[0] == x.shape[0] and sampled_x.shape[1] == x.shape[1]
    # PCA
    sampled_x_reduced = PCA(n_components=2).fit(sampled_x.detach().numpy())
    sampled_x_reduced = sampled_x_reduced.transform(sampled_x.detach().numpy())

    x_reduced = PCA(n_components=2).fit(x.detach().numpy())
    x_reduced = x_reduced.transform(x.detach().numpy())

    plt.clf()
    fig, axes = plt.subplots()
    sns.kdeplot(x=x_reduced[:, 0], y=x_reduced[:, 1], cmap='Reds', shade=True, ax = axes)
    # axes.scatter(x_reduced[:, 0], x_reduced[:, 1], edgecolor='k', alpha=0.4)
    axes.set_title(r"Input $Z$ ----> ")
    plt.savefig(f'{out_dir}/plt_densities_input_Z.png')

    plt.clf()
    fig, axes = plt.subplots()
    sns.kdeplot(x=sampled_x_reduced[:, 0], y=sampled_x_reduced[:, 1], shade=True, ax = axes)
    # axes.scatter(sampled_x_reduced[:, 0], sampled_x_reduced[:, 1], edgecolor='k', alpha=0.4)
    axes.set_title(r"Sampled $Z$")
    plt.savefig(f'{out_dir}/plt_densities_sampled_Z.png')


def plot_reconstructions(samples, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    particles_dir = f'{out_dir}/particles/'
    os.makedirs(particles_dir, exist_ok=True)
    samples = samples.detach().cpu().numpy()
    N = samples.shape[0]
    for i in range(0, N, 2):
        # print(f'Sample {i}.... ')
        plt.clf()
        fig = plt.figure(figsize=(25, 20))
        fig.suptitle('Particle Systems')
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_title(f"Sample {i%N}")
        z0_sample = samples[i, :]
        z0_sample = z0_sample.reshape((-1, 3))
        ax.scatter3D(z0_sample[:, 0], z0_sample[: , 1], z0_sample[:, 2], color = "green")
        np.savetxt(f'{particles_dir}/sampling_z0_{i}.particles', z0_sample)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_title(f"Sample {(i+1)%N}")
        z_new_sample = samples[(i+1)%N, :]
        z_new_sample = z_new_sample.reshape((-1, 3))
        ax.scatter3D(z_new_sample[:, 0], z_new_sample[: , 1], z_new_sample[:, 2], color = "green")
        np.savetxt(f'{particles_dir}/sampling_z_{i}.particles', z_new_sample)
        plt.savefig(f'{out_dir}/plt_{i}_sampling.png')

if __name__ == "__main__":
    # Logger
    train_done = False
    args = parse_args()
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO)
    logger.info("Hi!")
    logger.debug("Starting train.py with arguments %s", args)

    create_modelname(args)

    if args.resume is not None:
        resume_filename = create_filename("resume", None, args)
        args.startepoch = args.resume
        logger.info("Resuming training. Loading file %s and continuing with epoch %s.", resume_filename, args.resume + 1)
    elif args.load is None:
        logger.info("Training model %s with algorithm %s on data set %s", args.modelname, args.algorithm, args.dataset)
    else:
        logger.info("Loading model %s and training it as %s with algorithm %s on data set %s", args.load, args.modelname, args.algorithm, args.dataset)

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Data
    simulator = load_simulator(args)
    dataset = simulator.load_dataset(dataset_dir=args.dataset_dir, use_augmented_data=args.perturb_data, latent_dim=args.modellatentdim, scaledata=args.scaledata)
    args.datadim = simulator.data_dim()
    args.scalefactor = simulator.scaling_factor()
    # Model
    model = create_model(args, simulator)
    # Maybe load pretrained model
    if args.resume is not None:
        model.load_state_dict(torch.load(resume_filename, map_location=torch.device("cpu")))
        fix_act_norm_issue(model)
    elif args.load is not None:
        args_ = copy.deepcopy(args)
        args_.modelname = args.load
        if args_.i > 0:
            args_.modelname += "_run{}".format(args_.i)
        logger.info("Loading model %s", args_.modelname)
        model.load_state_dict(torch.load(create_filename("model", None, args_), map_location=torch.device("cpu")))
        fix_act_norm_issue(model)

    if (not args.eval_model) and (not args.serialize_model):
        # Train and save
        logger.info("Start Training model")
        learning_curves = train_model(args, dataset, model, simulator)

        # Save
        logger.info("Saving model")
        torch.save(model.state_dict(), create_filename("model", None, args))
        np.save(create_filename("learning_curve", None, args), learning_curves)
        train_done = True
        logger.info("All Training done! Have a nice day!")
    if args.eval_model:
        eval_dev = torch.device('cpu')
        if args.eval_model:
            model_fn = create_filename("model", None, args)
            if not os.path.exists(model_fn):
                model_fn = create_filename("resume", None, args)
            model.load_state_dict(torch.load(model_fn, map_location=eval_dev))
        eval_results_dir = "{}/experiments/data/eval_results/{}/".format(args.dir, args.modelname)
        os.makedirs(eval_results_dir, exist_ok=True)
        shutil.copy2(args.c, eval_results_dir)
        logger.info("Evaluating model now !!!!!")
        run_evaluation(model, dataset, eval_dev,eval_results_dir, args )
        logger.info("All Eval done! Have a nice day!")

    if args.serialize_model:
        eval_dev = torch.device('cpu')

        model_fn = create_filename("model", None, args)
        if not os.path.exists(model_fn):
            model_fn = create_filename("resume", None, args)
        model.load_state_dict(torch.load(model_fn, map_location=eval_dev))
        logger.info("Model Loaded, Running Serialization !!!!!")
        eval_results_dir = "{}/experiments/data/eval_results/{}/".format(args.dir, args.modelname)
        os.makedirs(eval_results_dir, exist_ok=True)
        model.eval()
        sm = torch.jit.script(model)
        serialized_model_path = f"{eval_results_dir}/serialized_model.pt"
        torch.jit.save(sm, serialized_model_path)
        logger.info(f'******************** Serialized Module saved ************************')
        logger.info("Serialization done! Have a nice day!")