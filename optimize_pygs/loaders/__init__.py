import importlib

from optimize_pygs.loaders.base_sampler import BaseSampler

SAMPLER_REGISTRY = {}

def register_sampler(name):
    """
    New sampler types can be added with the :func:`register_sampler`
    function decorator.
    
    For example:
        @register_sampler('gat')
        class GAT(BaseSampler):
            (...)
    """
    def register_sampler_cls(cls):
        if name in SAMPLER_REGISTRY:
            raise ValueError(f"Cannot register duplicate Sampler ({name})")
        if not issubclass(cls, BaseSampler):
            raise ValueError(f"Sampler ({name}: {cls.__name__}) must extend BaseSampler")
        SAMPLER_REGISTRY[name] = cls
        cls.sampler_name = name
        return cls 
    return register_sampler_cls


def try_import_sampler(sampler):
    if sampler not in SAMPLER_REGISTRY:
        if sampler in SUPPORED_SAMPLERS:
            importlib.import_module(SUPPORED_SAMPLERS[sampler])
        else:
            print(f"Failed to import {sampler} sampler.")
            return False
    return True


def build_sampler(args): 
    if not try_import_sampler(args.sampler):
        exit(1)
    return SAMPLER_REGISTRY[args.sampler].build_sampler_from_args(args)


def build_sampler_from_name(sampler, **args): 
    if not try_import_sampler(sampler):
        exit(1)
    return SAMPLER_REGISTRY[sampler](**args)


SUPPORED_SAMPLERS = {
    "cluster": "optimize_pygs.loaders.graph_sampler",
    "graphsage": "optimize_pygs.loaders.layer_sampler",
}