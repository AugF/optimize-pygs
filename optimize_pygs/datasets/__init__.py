import importlib 

from optimize_pygs.datasets.custom_dataset import CustomDataset # added

try:
    import torch_geometric
except ImportError:
    pyg = False
else:
    pyg = True

DATASET_REGISTRY = {}


def register_dataset(name):
    """
    New dataset types can be added with :func: `register_dataset` function decoratordatetime A combination of a date and a time. Attributes: ()
    
    For example:
        
        @register_dataset('my_dataset')
        class MyDataset():
            (...)
    
    Args:
        name(str): the name of the dataset
    """
    def register_dataset_cls(cls):
        if not issubclass(cls, CustomDataset) and (pyg and not issubclass(cls, torch_geometric.data.Dataset)):
            raise ValueError(f"Dataset ({name}: {cls.__name__}) must extend optimize_pygs.datasets.Dataset")
        DATASET_REGISTRY[name] = cls
        return cls
    
    return register_dataset_cls


def try_import_dataset(dataset):
    if dataset not in DATASET_REGISTRY:
        if dataset in SUPPORTED_DATASETS:
            importlib.import_module(SUPPORTED_DATASETS[dataset])
        else:
            print(f"Failed to import {dataset} dataset.")
            return False
    return True
        
def build_dataset(args):
    if not try_import_dataset(args.dataset):
        # assert hasattr(args, "task")
        # dataset = build_dataset_from_path(args.dataset, args.task)
        # if dataset is not None:
        #     return dataset
        exit(1)
    return DATASET_REGISTRY[args.dataset]()


def build_dataset_from_name(dataset):
    if not try_import_dataset(dataset):
        exit(1)
    return DATASET_REGISTRY[dataset]()


def build_dataset_from_path(data_path, task):
    if "node_classification" in task:
        pass
    elif "graph_classification" in task:
        pass
    else:
        return None


SUPPORTED_DATASETS = {
    'ppi': 'optimize_pygs.datasets.graphsaint_data',
    'ppi-large': 'optimize_pygs.datasets.graphsaint_data',
    'flickr': 'optimize_pygs.datasets.graphsaint_data',
    'reddit': 'optimize_pygs.datasets.graphsaint_data',
    'yelp': 'optimize_pygs.datasets.graphsaint_data',
    'amazon': 'optimize_pygs.datasets.graphsaint_data',
    'cora': 'optimize_pygs.datasets.neuroc_data',
    'pubmed': 'optimize_pygs.datasets.neuroc_data',
    'amazon-computers': 'optimize_pygs.datasets.neuroc_data',
    'amazon-photo': 'optimize_pygs.datasets.neuroc_data',
    'coauthor-physics': 'optimize_pygs.datasets.neuroc_data',
    'com-amazon': 'optimize_pygs.datasets.neuroc_data',
}    