import argparse
import sys

from optimize_pygs.datasets import DATASET_REGISTRY, try_import_dataset
from optimize_pygs.models import MODEL_REGISTRY, try_import_model
from optimize_pygs.loaders import SAMPLER_REGISTRY, try_import_sampler

def get_parser():
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    # fmt: off
    parser.add_argument('--seed', default=[1], type=int, nargs='+', metavar='N',
                        help='pseudo random number generator seed')
    parser.add_argument('--max-epoch', default=500, type=int)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of CUDA')
    parser.add_argument('--device-id', default=[0], type=int, nargs='+',
                        help='which GPU to use')
    # fmt: on
    return parser


def add_dataset_args(parser):
    group = parser.add_argument_group("Dataset and data loading")
    # fmt: off
    group.add_argument('--dataset', '-dt', metavar='DATASET', nargs='+', required=True,
                       help='Dataset')
    # fmt: on
    return group


def add_model_args(parser):
    group = parser.add_argument_group("Model configuration")
    # fmt: off
    group.add_argument('--model', '-m', metavar='MODEL', nargs='+', required=True,
                       help='Model Architecture')
    # fmt: on
    return group


def add_sampler_args(parser):
    # fmt: off
    parser.add_argument('--sampler', '-s', type=str, required=False, default="graphsage", 
                       help='Loader: sampler')
    # fmt: on


def add_trainer_args(parser):
    group = parser.add_argument_group("Trainer configuration")
    # fmt: off
    group.add_argument('--trainer', metavar='TRAINER', nargs='+', required=False,
                       help='Trainer')
    # fmt: on
    return group

def get_training_parser():
    parser = get_parser()
    add_dataset_args(parser)
    add_model_args(parser)
    add_sampler_args(parser)
    # add_trainer_args(parser)
    return parser


def get_default_args(dataset, model, sampler, **kwargs): 
    if not isinstance(dataset, list):
        dataset = [dataset]
    if not isinstance(model, list):
        model = [model]
    sys.argv = [sys.argv[0], "-m"] + model + ["-dt"] + dataset + ["-s" + sampler] 
    parser = get_training_parser()
    args, _ = parser.parse_known_args()
    args = parse_args_and_arch(parser, args)
    for key, value in kwargs.items():
        args.__setattr__(key, value)
    return args


def parse_args_and_arch(parser, args): 
    """The parser doesn't know about model-specific args, so we parse twice."""
    for model in args.model:
        if try_import_model(model):
            MODEL_REGISTRY[model].add_args(parser)
    for dataset in args.dataset:
        if try_import_dataset(dataset):
            if hasattr(DATASET_REGISTRY[dataset], "add_args"):
                DATASET_REGISTRY[dataset].add_args(parser)
    if try_import_sampler(args.sampler):
        if hasattr(SAMPLER_REGISTRY[args.sampler], "add_args"):
            SAMPLER_REGISTRY[args.sampler].add_args(parser)
    # if "trainer" in args and args.trainer is not None:
    #     for trainer in args.trainer:
    #         if try_import_universal_trainer(trainer):
    #             if hasattr(UNIVERSAL_TRAINER_REGISTRY[trainer], "add_args"):
    #                 UNIVERSAL_TRAINER_REGISTRY[trainer].add_args(parser)
    # Parse a second time.
    args = parser.parse_args()

    return args