import os.path as osp

from torch_geometric.datasets.planetoid import Planetoid
from torch_geometric.datasets.amazon import Amazon
from torch_geometric.datasets.coauthor import Coauthor

from optimize_pygs.global_configs import dataset_root, neuroc_dataset_root
from optimize_pygs.utils import download_url, accuracy, cross_entropy_loss
from optimize_pygs.datasets import CustomDataset, register_dataset


@register_dataset("cora")
class PPIDataset(Planetoid):
    def __init__(self):
        dataset = "cora"
        super(PPIDataset, self).__init__(neuroc_dataset_root, dataset, split="full")

    def get_evaluator(self):
        return accuracy

    def get_loss_fn(self):
        return cross_entropy_loss


@register_dataset("pubmed")
class PubmedDataset(Planetoid):
    def __init__(self):
        dataset = "pubmed"
        super(PubmedDataset, self).__init__(neuroc_dataset_root, dataset, split="full")

    def get_evaluator(self):
        return accuracy

    def get_loss_fn(self):
        return cross_entropy_loss
    

@register_dataset("amazon-computers")
class AmazonComputersDataset(Amazon):
    def __init__(self):
        dataset = "computers"
        path = osp.join(neuroc_dataset_root, "amazon-computers")
        super(AmazonComputersDataset, self).__init__(path, dataset)

    def get_evaluator(self):
        return accuracy

    def get_loss_fn(self):
        return cross_entropy_loss


@register_dataset("amazon-photo")
class AmazonPhotoDataset(Amazon):
    def __init__(self):
        dataset = "photo"
        path = osp.join(neuroc_dataset_root, "amazon-photo")
        super(AmazonPhotoDataset, self).__init__(path, dataset)

    def get_evaluator(self):
        return accuracy

    def get_loss_fn(self):
        return cross_entropy_loss


@register_dataset("coauthor-physics")
class CoauthorPhysicsDataset(Coauthor):
    def __init__(self):
        dataset = "physics"
        path = osp.join(neuroc_dataset_root, "coauthor-physics")
        super(CoauthorPhysicsDataset, self).__init__(path, dataset)

    def get_evaluator(self):
        return accuracy

    def get_loss_fn(self):
        return cross_entropy_loss
    

@register_dataset("com-amazon")
class ComAmazonDataset(CustomDataset):
    def __init__(self):
        dataset = "com-amazon"
        super(ComAmazonDataset, self).__init__(dataset_root, dataset)

    def get_evaluator(self):
        return accuracy

    def get_loss_fn(self):
        return cross_entropy_loss