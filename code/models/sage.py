import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from code.globals import *

class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, cpu_eval=False):
        super(SAGE, self).__init__()

        self.use_cuda = (args_global.gpu >= 0)
        if cpu_eval:
            self.use_cuda = False        
        
        # params
        self.lr = train_paras['lr']
        self.weight_decay = train_paras['weight_decay']
        self.dropout = train_paras['dropout']  
        
        # model
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)
    
    
    def train_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph):
        """
        Forward and backward propagation
        """
        self.train()
        self.optimizer.zero_grad()
        preds, labels, labels_converted = self(node_subgraph, adj_subgraph)
        loss = self._loss(preds, labels_converted, norm_loss_subgraph)
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 5) # 梯度裁剪
        self.optimizer.step()
        return loss, self.predict(preds), labels


    def inference(self, x_all, subgraph_loader, device):
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)
        return x_all
    
    def eval_step(self, x, y, adj):
        """
        Forward propagation only
        """
        self.eval()
        with torch.no_grad():
            out = self(x, adj)
            loss = F.nll_loss(out, y)
        return loss
        
