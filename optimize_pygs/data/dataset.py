# https://github.com/rusty1s/pytorch_geometric/commit/f0c66093dd8bcd0e4ae81af0c5edf0a7e1b3db7e
import re
import copy
import logging
import errno
import os.path as osp

import torch.utils.data
from optimize_pygs.utils import makedirs, accuracy, cross_entropy_loss


def to_list(x):
    if not isinstance(x, (tuple, list)):
        x = [x]
    return x


def files_exist(files):
    return len(files) != 0 and all(osp.exists(f) for f in files)


def __repr__(obj):
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())


class Dataset(torch.utils.data.Dataset):
    r"""Dataset base class for creating graph datasets.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_dataset.html>`__ for the accompanying tutorial.
    Args:
        root (string, optional): Root directory where the dataset should be
            saved. (optional: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    @staticmethod
    def add_args(parser):
        """Add dataset-specific argument to the parser."""
        pass
    
    @property
    def raw_file_names(self):
        r"""
        """
        raise NotImplementedError
    
    @property
    def processed_file_names(self):
        r""""""
        raise NotImplementedError
    
    def download(self):
        raise NotImplementedError
    
    def process(self):
        raise NotImplementedError
    
    def __len__(self):
        r""""""
        raise NotImplementedError
    
    def get(self, idx):
        raise NotImplementedError
    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(Dataset, self).__init__()
        
        self.root = osp.expanduser((osp.normpath(root)))
        self.raw_dir = osp.join(self.root, "raw")
        self.processed_dir = osp.join(self.root, "processed")
        self.transfrom = transform 
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        
        self._download()
        self._process()
        
    @property 
    def num_features(self):
        return self[0].num_features

    @property
    def raw_paths(self):
        files = to_list(self.raw_file_names)
        return [osp.join(self.raw_dir, f) for i in files]

    @property
    def processed_paths(self):
        files = to_list(self.processed_file_names)
        return [osp.join(self.processed_dir, f) for f in files]

    def _download(self):
        if files_exist(self.raw_paths):
            return 
        
        makedirs(self.raw_dir)
        self.download()
    
    def _process(self):
        if files_exist(self.processed_paths):
            return
        
        print("Processing")
        
        makedirs(self.processed_dir)
        self.process()
        
        print("Done!")
    
    def get_evaluator(self):
        return accuracy
    
    def get_loss_fn(self):
        return cross_entropy_loss
    
    def __getitem__(self, idx):
        data = self.get(idx)
        data = data if self.transform is None else self.transform(data)
        return data
    
    @property
    def num_classes(self):
        y = self.data.y
        return y.max().item() + 1 if y.dim() == 1 else y.size(1)
    

class MultiGraphDataset(Dataset):
    """
    """
    pass
        
    