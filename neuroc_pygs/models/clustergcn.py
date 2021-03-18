import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class ClusterGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, device):
        super(ClusterGCN, self).__init__()
        self.inProj = torch.nn.Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.linear = torch.nn.Linear(hidden_channels, out_channels)
        self.weights = torch.nn.Parameter(torch.randn((len(self.convs))))
        self.device = device
        self.dropout = dropout
        self.loss_fn, self.evaluator = None, None

    def reset_parameters(self):
        self.inProj.reset_parameters()
        self.linear.reset_parameters()
        torch.nn.init.normal_(self.weights)
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.inProj(x)
        inp = x
        x = F.relu(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x + 0.2*inp
 
        return torch.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, df=None):
        device = self.device
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')
        
        x_all = self.inProj(x_all.to(device))
        x_all = x_all.cpu()
        inp = x_all
        x_all = F.relu(x_all)
        out = []
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                xs.append(x.cpu())

                pbar.update(batch_size)

                if i == 0:
                    if df is not None:
                        node, edge = size[0], edge_index.shape[1]
                        memory = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]
                        df['nodes'].append(node)
                        df['edges'].append(edge)
                        df['memory'].append(memory)
                        print(f'nodes={node}, edge={edge}, real: {memory}')
                    torch.cuda.reset_max_memory_allocated(device)
                    torch.cuda.empty_cache()
                    
            x_all = torch.cat(xs, dim=0)
            if i != len(self.convs) - 1:
                x_all = x_all + 0.2*inp
        pbar.close()

        return x_all
    
    def inference_cuda(self, x_all, subgraph_loader):
        device = self.device        
        x_all = self.inProj(x_all.to(device))
        x_all = x_all.cpu()
        inp = x_all
        x_all = F.relu(x_all)
        out = []
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)
            if i != len(self.convs) - 1:
                x_all = x_all + 0.2*inp
        return x_all
    
    
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
        
    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def get_hyper_paras(self):
        return { #in_channels, hidden_channels, out_channels
            'layers': self.num_layers,
            'n_features': self.in_channels,
            'n_classses': self.out_channels,
            'hidden_dims': self.hidden_channels,
        }