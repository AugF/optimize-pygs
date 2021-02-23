import importlib 

from optimize_pygs.criterions.base_criterions import EarlyStoppingCriterion # added

CRITERION_REGISTRY = {}


def register_criterion(name):
    """
    New criterion types can be added with :func: `register_criterion` function decoratordatetime A combination of a date and a time. Attributes: ()
    
    For example:
        
        @register_criterion('my_criterion)
        class MyCriterion():
            (...)
    
    """
    def register_criterion_cls(cls):
        if not issubclass(cls, EarlyStoppingCriterion):
            raise ValueError(f"Criterion ({name}: {cls.__name__}) must extend optimize_pygs.criterions.EarlyStoppingCriterion")
        CRITERION_REGISTRY[name] = cls
        return cls
    
    return register_criterion_cls


def try_import_criterion(criterion):
    if criterion not in CRITERION_REGISTRY:
        if criterion in SUPPORTED_CRITERIONS:
            importlib.import_module(SUPPORTED_CRITERIONS[criterion])
        else:
            print(f"Failed to import {criterion} criterion.")
            return False
    return True

 
def build_criterion(args):
    if not try_import_criterion(args.criterion):
        print(f"Criterion ({name} is not defined.")
        exit(1)
    return CRITERION_REGISTRY[args.criterion]() # 这里不需要参数


def build_criterion_from_name(criterion, **args):
    if not try_import_criterion(criterion):
        exit(1)
    return CRITERION_REGISTRY[criterion](**args)


SUPPORTED_CRITERIONS = {
    'no_stopping_with_acc': 'optimize_pygs.criterions.notstopping_criterions',
    'no_stopping_with_loss': 'optimize_pygs.criterions.notstopping_criterions',
    'gcn': 'optimize_pygs.criterions.gnn_benchmark_criterions',
    'gat': 'optimize_pygs.criterions.gnn_benchmark_criterions',
    'kdd': 'optimize_pygs.criterions.gnn_benchmark_criterions',
    'gat_with_tolerance': 'optimize_pygs.criterions.gnn_benchmark_criterions',
}    