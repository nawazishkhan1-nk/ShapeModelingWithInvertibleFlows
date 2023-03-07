import logging

from .image_transforms import create_image_transform, create_image_encoder
from .vector_transforms import create_vector_encoder, create_vector_transform, create_vector_decoder
from manifold_flow import transforms
# from manifold_flow.flows import Flow, EncoderManifoldFlow, VariableDimensionManifoldFlow, ManifoldFlow, ProbabilisticAutoEncoder
from manifold_flow.flows import ManifoldFlow


logger = logging.getLogger(__name__)


ALGORITHMS = ["flow", "pie", "mf", "gamf", "hybrid", "emf", "pae"]  # , "dough"


def create_model(args, simulator):
    assert args.algorithm in ALGORITHMS

    if simulator.is_image():
        c, h, w = simulator.data_dim()
    else:
        c, h, w = None, None, None

    # FOM for vector data
    if args.algorithm in ["mf", "gamf", "pie"] and args.specified and not simulator.is_image():
        model = create_vector_specified_flow(args, simulator)

    # FOM for image data
    elif args.algorithm in ["mf", "gamf", "pie"] and args.specified and simulator.is_image():
        raise NotImplementedError

    # M-flow or PIE for vector data
    elif args.algorithm in ["mf", "gamf", "pie"] and not args.specified and not simulator.is_image():
        print("Creating Vector MF")
        model = create_vector_mf(args, simulator)


    else:
        raise ValueError(f"Don't know how to construct model for algorithm {args.algorithm} and image flag {simulator.is_image()}")

    return model


def create_vector_mf(args, simulator):
    print("Main Model Creation")
    logger.info(
        "Creating manifold flow for vector data with %s latent dimensions, %s + %s layers, transforms %s / %s, %s context features",
        args.modellatentdim,
        args.outerlayers,
        args.innerlayers,
        args.outertransform,
        args.innertransform,
        simulator.parameter_dim(),
    )
    outer_transform = create_vector_transform(
        args.datadim,
        args.outerlayers,
        linear_transform_type=args.lineartransform,
        base_transform_type=args.outertransform,
        context_features=simulator.parameter_dim() if args.conditionalouter else None,
        dropout_probability=args.dropout,
        tail_bound=args.splinerange,
        num_bins=args.splinebins,
        use_batch_norm=args.batchnorm,
    )
    inner_transform = create_vector_transform(
        args.modellatentdim,
        args.innerlayers,
        linear_transform_type=args.lineartransform,
        base_transform_type=args.innertransform,
        context_features=simulator.parameter_dim(),
        dropout_probability=args.dropout,
        tail_bound=args.splinerange,
        num_bins=args.splinebins,
        use_batch_norm=args.batchnorm,
    )
    model = ManifoldFlow(
        data_dim=args.datadim,
        latent_dim=args.modellatentdim,
        outer_transform=outer_transform,
        inner_transform=inner_transform,
        apply_context_to_outer=args.conditionalouter,
        pie_epsilon=args.pieepsilon,
        clip_pie=args.pieclip,
    )
    return model


def create_vector_specified_flow(args, simulator):
    logger.info(
        "Creating manifold flow for vector data with %s latent dimensions, specified outer transformation + %s inner layers, transform %s, %s context features",
        args.modellatentdim,
        args.innerlayers,
        args.innertransform,
        simulator.parameter_dim(),
    )
    if args.dataset in ["spherical_gaussian", "conditional_spherical_gaussian"]:
        outer_transform = transforms.SphericalCoordinates(n=args.modellatentdim, r0=1.0)
    else:
        raise NotImplementedError("Specified outer transformation not yet implemented for dataset {}".format(args.dataset))
    inner_transform = create_vector_transform(
        args.modellatentdim,
        args.innerlayers,
        linear_transform_type=args.lineartransform,
        base_transform_type=args.innertransform,
        context_features=simulator.parameter_dim(),
        dropout_probability=args.dropout,
        tail_bound=args.splinerange,
        num_bins=args.splinebins,
        use_batch_norm=args.batchnorm,
    )
    model = ManifoldFlow(
        data_dim=args.datadim,
        latent_dim=args.modellatentdim,
        outer_transform=outer_transform,
        inner_transform=inner_transform,
        apply_context_to_outer=args.conditionalouter,
        pie_epsilon=args.pieepsilon,
        clip_pie=args.pieclip,
    )
    return model

