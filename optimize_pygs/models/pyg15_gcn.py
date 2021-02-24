import sys
import torch
import time
import torch.nn.functional as F

from tqdm import tqdm

from optimize_pygs.layers import GCNConv
from optimize_pygs.utils.pyg15_utils import gcn_cluster_norm
from optimize_pygs.utils.pyg15_utils import nvtx_push, nvtx_pop, log_memory
from . import BaseModel, register_model


@register_model("pyg15_gcn")
class GCN(BaseModel):
    """
    GCN layer
    dropout set: https://github.com/tkipf/pygcn/blob/master/pygcn/train.py
    """
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser.
        """
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--norm", type=torch.Tensor)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--num-layers", type=int, default=2)
        # fmt: on
        
    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            norm=args.norm,
            device = "cpu" if not torch.cuda.is_available() or args.cpu else f'cuda:{args.device_id}',
            nvtx_flag=args.nvtx_flag,
            memory_flag=args.memory_flag,
            infer_flag=args.infer_flag
        )
        
    def __init__(self, num_features, hidden_size, num_classes, num_layers, norm=None, dropout=0.5,
                 nvtx_flag=False, device="cpu", memory_flag=False, infer_flag=False, cluster_flag=False,
                 cached_flag=False):
        """
        device: 其值表示内存数据统计中指定的GPU设备
        nvtx_flag: True表示启动torch.cuda.nvtx运行时间统计
        memory_flag: True表示启动Training阶段GPU内存数据统计
        infer_flag: True表示启动inference阶段GPU内存数据统计
        cluster_flag: True表示训练时采用的是cluster sampler, 需要特殊处理
        """
        super(GCN, self).__init__()
        self.num_features, self.num_classes = num_features, num_classes
        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.dropout = dropout
        self.nvtx_flag = nvtx_flag
        self.memory_flag, self.infer_flag = memory_flag, infer_flag
        self.device = device

        shapes = [num_features] + [hidden_size] * (num_layers - 1) + [num_classes]
        self.convs = torch.nn.ModuleList(
            [
                GCNConv(in_channels=shapes[layer], out_channels=shapes[layer + 1], gpu=nvtx_flag, device=device, cached=cached_flag)
                for layer in range(num_layers)
            ]
        )
        if norm is not None:
            self.norm = norm.to(device)
        self.cluster_flag = cluster_flag
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            
    def forward(self, x, adjs):
        """
        修改意见：https://github.com/THUDM/cogdl/blob/master/cogdl/models/nn/pyg_gcn.py
        :param x: [num_nodes, num_features]
        :param edge_index: [num_nodes, num_features]
        :return: x
        """
        device = self.device
        if isinstance(adjs, list):
            for i, (edge_index, e_id, size) in enumerate(adjs):
                nvtx_push(self.nvtx_flag, "layer" + str(i))
                x = self.convs[i](x, edge_index, size=size[1], norm=self.norm[e_id])
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                nvtx_pop(self.nvtx_flag)
                log_memory(self.memory_flag, device, 'layer' + str(i))
        else:
            if self.cluster_flag:
                norm = gcn_cluster_norm(adjs, x.size(0), None, False, x.dtype)
            else:
                norm = None
            for i in range(self.num_layers):
                nvtx_push(self.nvtx_flag, "layer" + str(i))
                x = self.convs[i](x, adjs, norm=norm)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                nvtx_pop(self.nvtx_flag)
                log_memory(self.memory_flag, device, 'layer' + str(i))
                
        return x # loss使用默认的交叉熵

    def inference(self, x_all, subgraph_loader):
        device = torch.device(self.device)
        flag = self.infer_flag
        
        sampling_time, to_time, train_time = 0.0, 0.0, 0.0
        total_batches = len(subgraph_loader)

        log_memory(flag, device, 'inference start')       
        for i in range(self.num_layers):
            log_memory(flag, device, f'layer{i} start')

            xs = []
            loader_iter = iter(subgraph_loader)
            while True:
                try:
                    et0 = time.time()      
                    batch_size, n_id, adj = next(loader_iter)
                    log_memory(flag, device, 'batch start')                    
                    
                    et1 = time.time()      
                    edge_index, e_id, size = adj.to(device)
                    x = x_all[n_id].to(device)
                    log_memory(flag, device, 'to end') 
                    
                    et2 = time.time()
                    x = self.convs[i](x, edge_index, size=size[1], norm=self.norm[e_id])
                    if i != self.num_layers - 1:
                        x = F.relu(x)
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
        return '{}(num_layers={}, num_features={}, num_classes={}, hidden_size={}, dropout={})'.format(
            self.__class__.__name__, self.num_layers, self.num_features, self.num_classes, self.hidden_size,
            self.dropout) + '\nLayer(conv->relu->dropout)\n' + str(self.convs)




