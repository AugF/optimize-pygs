import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch_geometric.nn.conv import SAGEConv

from . import BaseModel, register_model


@register_model("pyg_sage")
class SAGE(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--hidden-size", type=int, default=8)
        parser.add_argument("--dropout", type=float, default=0.6)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.dropout,
        )

    def __init__(self, num_features, hidden_size, num_classes, num_layers, dropout):
        super(SAGE, self).__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        shapes = [num_features] + [hidden_size] * (num_layers - 1) + [num_classes]
        self.convs = nn.ModuleList(
            [SAGEConv(shapes[layer], shapes[layer + 1]) for layer in range(num_layers)]
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=-1)

    def predict(self, data):
        return self.forward(data.x, data.edge_index)
    
    def node_classification_loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )
        
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(self.device)
                x = x_all[n_id].to(self.device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all