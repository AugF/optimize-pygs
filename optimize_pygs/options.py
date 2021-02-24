import argparse
import sys

from optimize_pygs.datasets import DATASET_REGISTRY, try_import_dataset
from optimize_pygs.models import MODEL_REGISTRY, try_import_model


def get_parser():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--max-epoch', default=500, type=int)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of CUDA')
    parser.add_argument('--device-id', default=1)
    
    # analysis
    parser.add_argument('--nvtx_flag', type=bool, required=False, default=False)
    parser.add_argument('--memory_flag', type=bool, required=False, default=False)
    parser.add_argument('--infer_flag', type=bool, required=False, default=False)
    
    # special
    parser.add_argument('--infer_layer', type=bool, required=False, default=False)
    # fmt: on
    return parser


def add_sampler_args(parser):
    # fmt: off
    parser.add_argument('--sampler', type=str, required=False, default="graphsage", 
                       help='Loader: sampler')
    parser.add_argument('--infer_sampler', type=str, required=False, default="graphsage", 
                       help='Val/Test Loader: infer_sampler')
    parser.add_argument('--num_parts', type=int, required=False, default=1500)
    parser.add_argument('--batch_partitions', type=int, required=False, default=150)
    parser.add_argument('--batch_size', type=int, required=False, default=1024)
    parser.add_argument('--infer_batch_size', type=int, required=False, default=1024)
    parser.add_argument('--num_workers', type=int, required=False, default=12)
    # fmt: on


def get_training_parser():
    parser = get_parser()
    parser.add_argument('--model', '-m', default='pyg15_ggnn', type=str)
    parser.add_argument('--dataset', '-dt', default='cora', type=str)
    add_sampler_args(parser)
    parser.add_argument('--criterion', '-c', default='no_stopping_with_acc', type=str, help="early stopping criterion")
    parser.add_argument('--trainer', '-tr', default='graph', type=str, help="match sampler")
    return parser


def get_default_args(model, dataset, **kwargs): # 函数调用的优先级最高!
    parser = get_training_parser()
    args = parser.parse_args()
    args.model, args.dataset = model, dataset
    args = parse_args_and_arch(parser, args)
    args.model, args.dataset = model, dataset # 需要赋值两次

    for key, value in kwargs.items():
        args.__setattr__(key, value)
    if args.sampler == "cluster":
        args.trainer = "graph"
    else:
        args.trainer = "layer"
    return args


def parse_args_and_arch(parser, args): 
    """The parser doesn't know about model-specific args, so we parse twice."""
    if try_import_model(args.model):
        MODEL_REGISTRY[args.model].add_args(parser)
    if try_import_dataset(args.dataset):
        if hasattr(DATASET_REGISTRY[args.dataset], "add_args"):
            DATASET_REGISTRY[args.dataset].add_args(parser)
    args = parser.parse_args()
    return args

