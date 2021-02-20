import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from . import BaseModel

class ModelName(BaseModel):
    """
    [Model Name] [Link to paper]
    """
    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--batch-size", type=int, default=20)
        # 划分这里可以归到由数据集本身提供
        parser.add_argument("--train-ratio", type=float, default=0.7)
        parser.add_argument("--test-ratio", type=float, default=0.1)
        parser.add_argument("--lr", type=float, default=0.001)
    
    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
        )
    
    def __init__(self, in_feats, hidden_dim, out_feats, k=20, dropout=0.5):
        # 这里k是什么意思
        super(ModelName, self).__init__
        
    def forward(self, x, adj):
        return 0, 0
    
    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )
    
    def predict(self, data):
        return self.forward(data.x, data.edge_index)