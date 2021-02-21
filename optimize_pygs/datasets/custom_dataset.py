import json
import os.path as osp

import torch
import numpy as np
import scipy.sparse as sp

from optimize_pygs.datasets.in_memory_dataset import InMemoryDataset
from optimize_pygs.data.data import Data
from optimize_pygs.global_configs import dataset_root as root
from optimize_pygs.utils import get_evaluator, get_loss_fn

class CustomDataset(InMemoryDataset):
    r"""
    首先，按照https://github.com/GraphSAINT/GraphSAINT中介绍的将数据集data预处理成adj_full.npz, adj_train.npz, role.json, class_map,json, feats.json
    这里借鉴的是PyG对https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/flickr.html#Flickr
    将数据集变为PyG的数据集的格式，并后续采用它的预处理方法
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.epochs.Data` object and returns a transformed
            version. The epochs object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.epochs.Data` object and returns a
            transformed version. The epochs object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root, name, metric="accuracy", transform=None, pre_transform=None):
        root = osp.join(root, name)
        super(CustomDataset, self).__init__(root, transform, pre_transform)
        self.metric = metric
        self.data, self.slices = torch.load(self.processed_paths[0]) # 直接这里返回就可以

    @property
    def raw_file_names(self):
        return ['adj_full.npz', 'feats.npy', 'class_map.json', 'role.json']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        f = np.load(osp.join(self.raw_dir, 'adj_full.npz'))
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        x = np.load(osp.join(self.raw_dir, 'feats.npy'))
        x = torch.from_numpy(x).to(torch.float)

        ys = [0] * x.size(0) # com-amazon中.cmty_txt没有包含全部
        with open(osp.join(self.raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = torch.tensor(ys)

        with open(osp.join(self.raw_dir, 'role.json')) as f:
            role = json.load(f)

        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        train_mask[torch.tensor(role['tr'])] = True

        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask[torch.tensor(role['va'])] = True

        test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask[torch.tensor(role['te'])] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        data = data if self.pre_transform is None else self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def get_evaluator(self):
        evaluator = get_evaluator(self.metric)
        if evaluator == None:
            return NotImplementedError
        return evaluator

    def get_loss_fn(self):
        loss_fn = get_loss_fn(self.metric)
        if loss_fn == None:
            return NotImplementedError
        return loss_fn
    
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


"""examples
@register_dataset("ppi")
class PPIDataset(CustomDataset):
    def __init__(self):
        dataset = "ppi"
        metric = "accuracy"
        super(PPIDataset, self).__init__(dataset_root, dataset, metric)
"""


def test_custom_dataset():
    for name in ['ppi', 'flickr', 'reddit', 'yelp', 'amazon']: # len(dataset)=1
        dataset = CustomDataset(root, name)
        data = dataset[0]
        print(len(dataset), name, data)


if __name__ == "__main__":
    test_custom_dataset()