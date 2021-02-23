import importlib

from optimize_pygs.trainers.base_trainer import BaseTrainer

TRAINER_REGISTRY = {}

def register_trainer(name):
    """
    New trainer types can be added with the :func:`register_trainer`
    function decorator.
    
    For example:
        @register_trainer('gat')
        class GAT(BaseTrainer):
            (...)
    """
    def register_trainer_cls(cls):
        if name in TRAINER_REGISTRY:
            raise ValueError(f"Cannot register duplicate trainer ({name})")
        if not issubclass(cls, BaseTrainer):
            raise ValueError(f"trainer ({name}: {cls.__name__}) must extend BaseTrainer")
        TRAINER_REGISTRY[name] = cls
        cls.trainer_name = name
        return cls 
    return register_trainer_cls


def try_import_trainer(trainer):
    if trainer not in TRAINER_REGISTRY:
        if trainer in SUPPORED_TRAINERS:
            importlib.import_module(SUPPORED_TRAINERS[trainer])
        else:
            print(f"Failed to import {trainer} trainer.")
            return False
    return True


def build_trainer(args): 
    if not try_import_trainer(args.trainer):
        exit(1)
    return TRAINER_REGISTRY[args.trainer].build_trainer_from_args(args)


def build_trainer_from_name(trainer, **args): 
    if not try_import_trainer(trainer):
        exit(1)
    return TRAINER_REGISTRY[trainer](**args)


SUPPORED_TRAINERS = {
    "full": "optimize_pygs.trainers.full_batch_trainer",
    "graph": "optimize_pygs.trainers.graph_sampler_trainer",
    "layer": "optimize_pygs.trainers.layer_sampler_trainer",
}