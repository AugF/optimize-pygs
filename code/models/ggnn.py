import torch

import torch.nn.functional as F
from torch.nn import Parameter, Module

from code.models.ggnn_layer
from code.utils.inits import glorot

class GGNN(Module):
    def __init__(self, layers, n_features, n_classes, hidden_dims, gpu=False, device=None, flag=False, infer_flag=False):
        super(GGNN, self).__init__()
        self.n_features, self.n_classes = n_features, n_classes
        self.layers, self.hidden_dims = layers, hidden_dims

        self.weight_in = Parameter(torch.Tensor(n_features, hidden_dims))
        self.weight_out = Parameter(torch.Tensor(hidden_dims, n_classes))
        self.convs = torch.nn.ModuleList([GatedGraphConv(out_channels=hidden_dims, num_layers=layers)])
        glorot(self.weight_in)
        glorot(self.weight_out)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, adjs):
        device = torch.device(self.device)
        x = torch.matmul(x, self.weight_in)

        x = self.convs[0](x, adjs)
        x = torch.matmul(x, self.weight_out)
        return x

    def inference(self, x_all, subgraph_loader):
        device = torch.device(self.device)
        x_all = torch.matmul(x_all.to(device), self.weight_in) # 尽最大可能第键槽内存
        x_all = self.convs[0].inference(x_all.cpu(), subgraph_loader)
        x_all = torch.matmul(x_all.to(device), self.weight_out)
        return x_all.cpu()
    

