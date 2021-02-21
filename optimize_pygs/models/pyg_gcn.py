import torch
import torch.nn as nn 
import torch.nn.functional as F 

from torch_geometric.nn.conv import GCNConv
from . import BaseModel, register_model


@register_model("pyg_gcn")
class GCN(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser, for future
        """
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.5)
    
    @classmethod
    def build_model_from_args(cls, args):
        return cls( 
            args.num_features,
            args.num_classes,
            args.hidden_size,
            dropout=args.dropout, # 可以乱序
            num_layers=args.num_layers,
        )
    
    def __init__(self, num_features, num_classes, hidden_size, num_layers, dropout):
        super(GCN, self).__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        shapes = [num_features] + [hidden_size] * (num_layers - 1) + [num_classes]
        self.convs = nn.ModuleList(
            [GCNConv(shapes[layer], shapes[layer + 1], cached=False) for layer in range(num_layers)]
        )
    
    def forward(self, x, edge_index, weight=None):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index, weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, weight)
        return F.log_softmax(x, dim=1)

    def node_classification_loss(self, data): # 这里是log_softmax + F.nll_loss, 所以重写了
        return F.nll_loss(
            self.forward(data.x, data.edge_index, None if "norm_aggr" not in data else data.norm_aggr)[data.train_mask],
            data.y[data.train_mask],
        )

    def predict(self, data):
        return self.forward(data.x, data.edge_index, None if "norm_aggr" not in data else data.norm_aggr)

