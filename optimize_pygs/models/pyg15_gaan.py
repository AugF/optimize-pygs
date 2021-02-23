import torch
import time

import torch.nn.functional as F
from torch.nn import Parameter, Module
from tqdm import tqdm

from optimize_pygs.layers import GaANConv
from optimize_pygs.utils.pyg15_utils import gcn_cluster_norm
from optimize_pygs.utils.pyg15_utils import nvtx_push, nvtx_pop, log_memory
from . import BaseModel, register_model


@register_model("pyg15_gaan")
class GaAN(BaseModel):
    """
    GaAN model
    dropout, negative_slop set: GaAN: Gated attention networks for learning on large and spatiotemporal graphs 5.3
    """
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser.
        """
        # fmt: off
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--heads", type=int, default=8)
        parser.add_argument("--d_v", type=int, default=8)
        parser.add_argument("--d_a", type=int, default=8)
        parser.add_argument("--d_m", type=int, default=64)
        # fmt: on
        
    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_layers,
            args.num_classes,
            args.heads,
            args.d_v,
            args.d_a,
            args.d_m,
            device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id,
            gpu=args.nvtx_flag,
            flag=args.memory_flag,
            infer_flag=args.infer_flag
        )
        
    def __init__(self, num_features, hidden_size, num_classes, num_layers, 
                 heads, d_v, d_a, d_m, dropout=0.1, negative_slop=0.1,
                 gpu=False, device="cpu", flag=False, infer_flag=False):
        super(GaAN, self).__init__()
        self.num_features, self.num_classes = num_features, num_classes
        self.num_layers, self.hidden_size, self.heads = num_layers, hidden_size, heads
        self.dropout, self.negative_slop = dropout, negative_slop
        self.d_v, self.d_a, self.d_m = d_v, d_a, d_m
        self.gpu = gpu
        self.device = device
        self.flag, self.infer_flag = flag, infer_flag

        shapes = [num_features] + [hidden_size] * (num_layers - 1) + [num_classes]
        self.convs = torch.nn.ModuleList(
            [
                GaANConv(in_channels=shapes[layer], out_channels=shapes[layer + 1],
                         d_a=d_a, d_m=d_m, d_v=d_v, heads=heads, gpu=gpu)
                for layer in range(num_layers)
            ]
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, adjs):
        device = torch.device(self.device)
        
        if isinstance(adjs, list):
            for i, (edge_index, _, size) in enumerate(adjs):
                nvtx_push(self.gpu, "layer" + str(i))
                x = self.convs[i](x, edge_index, size=size[1])
                if i != self.num_layers - 1:
                    x = F.leaky_relu(x, self.negative_slop)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                nvtx_pop(self.gpu)
                log_memory(self.flag, device, 'layer' + str(i))
        else:
            for i in range(self.num_layers):
                nvtx_push(self.gpu, "layer" + str(i))
                x = self.convs[i](x, adjs)
                if i != self.num_layers - 1:
                    x = F.leaky_relu(x, self.negative_slop)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                nvtx_pop(self.gpu)
                log_memory(self.flag, device, 'layer' + str(i))
                
        return x

    def inference(self, x_all, subgraph_loader):
        device = torch.device(self.device)
        flag = self.infer_flag
        
        sampling_time, to_time, train_time = 0.0, 0.0, 0.0
        total_batches = len(subgraph_loader)
        
        log_memory(flag, device, 'inference start') 
        for i in range(self.layers):
            log_memory(flag, device, f'layer{i} start')

            xs = []
            loader_iter = iter(subgraph_loader)
            while True:
                try:
                    et0 = time.time()
                    batch_size, n_id, adj = next(loader_iter)
                    log_memory(flag, device, 'batch start') 
                    
                    et1 = time.time()
                    edge_index, _, size = adj.to(device)
                    x = x_all[n_id].to(device)
                    log_memory(flag, device, 'to end') 
                    
                    et2 = time.time()
                    x = self.convs[i](x, edge_index, size=size[1])
                    if i != self.layers - 1:
                        x = F.leaky_relu(x, self.negative_slop)
                        x = F.dropout(x, p=self.dropout, training=self.training)
                    xs.append(x.cpu())
                    log_memory(flag, device, 'batch end') 
                    
                    sampling_time += et1 - et0
                    to_time += et2 - et1
                    train_time += time.time() - et2
                except StopIteration:
                    break
            
            x_all = torch.cat(xs, dim=0)
             
        sampling_time /= total_batches
        to_time /= total_batches
        train_time /= total_batches
        
        log_memory(flag, device, 'inference end') 
        print(f"avg_batch_train_time: {train_time}, avg_batch_sampling_time:{sampling_time}, avg_batch_to_time: {to_time}")
        return x_all
    
    def __repr__(self):
        return '{}(layers={}, num_features={}, num_classes={}, hidden_size={}, heads={},' \
               'd_v={}, d_a={}, d_m={}, dropout={}, negative_slop={}, gpu={})'.format(
            self.__class__.__name__, self.num_layers, self.num_features, self.num_classes,
            self.hidden_size, self.heads, self.d_v, self.d_a, self.d_m, self.dropout,
            self.negative_slop, self.gpu) + '\nLayer(conv->leaky_relu->dropout)\n' + str(self.convs[0])







