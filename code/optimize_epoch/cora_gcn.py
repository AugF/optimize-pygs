import os.path as osp
import copy
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

dataset = 'cora'
path = osp.join('/home/wangzhaokang/wangyunpan/gnns-project/datasets')
transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]
data.adj_t = gcn_norm(data.adj_t) 

class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0, 
                 weight_decay=5e-4, lr=0.01):
        super(Net, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout
        self.lr, self.weight_decay = lr, weight_decay
        self.optimizer = torch.optim.Adam([
                    dict(params=self.convs.parameters(), weight_decay=0.01),
                    dict(params=self.lins.parameters(), weight_decay=self.weight_decay)
                ], lr=self.lr)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.optimizer = torch.optim.Adam([
                    dict(params=self.convs.parameters(), weight_decay=0.01),
                    dict(params=self.lins.parameters(), weight_decay=self.weight_decay)
                ], lr=self.lr)

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)
    
    def train_step(self, data):
        self.train()
        self.optimizer.zero_grad()
        out = self.forward(data.x, data.adj_t)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return float(loss)
    
    def eval_step(self, data):
        self.eval()
        with torch.no_grad():
            pred, accs = self(data.x, data.adj_t).argmax(dim=-1), []
            for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
            return accs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5,
            shared_weights=True, dropout=0.6, 
            weight_decay=5e-4, lr=0.01)

