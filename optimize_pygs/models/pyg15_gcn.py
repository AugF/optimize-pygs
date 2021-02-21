import sys
import torch
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
            args.num_layers,
            args.num_features,
            args.num_classes,
            args.hidden_size,
            norm=args.norm,
        )
        
    def __init__(self, layers, n_features, n_classes, hidden_dims, norm=None, dropout=0.5,
                 gpu=False, device="cpu", flag=False, infer_flag=False, cluster_flag=False,
                 cached_flag=True):
        """
        gpu: True表示启动torch.cuda.nvtx运行时间统计
        device: 其值表示内存数据统计中指定的GPU设备
        flag: True表示启动Training阶段GPU内存数据统计
        infer_flag: True表示启动inference阶段GPU内存数据统计
        cluster_flag: True表示训练时采用的是cluster sampler, 需要特殊处理
        """
        super(GCN, self).__init__()
        self.n_features, self.n_classes = n_features, n_classes
        self.layers, self.hidden_dims = layers, hidden_dims
        self.dropout = dropout
        self.gpu = gpu
        self.flag, self.infer_flag = flag, infer_flag
        self.device = device

        shapes = [n_features] + [hidden_dims] * (layers - 1) + [n_classes]
        self.convs = torch.nn.ModuleList(
            [
                GCNConv(in_channels=shapes[layer], out_channels=shapes[layer + 1], gpu=gpu, device=device, cached=cached_flag)
                for layer in range(layers)
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
                nvtx_push(self.gpu, "layer" + str(i))
                x = self.convs[i](x, edge_index, size=size[1], norm=self.norm[e_id])
                if i != self.layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                nvtx_pop(self.gpu)
                log_memory(self.flag, device, 'layer' + str(i))
        else:
            if self.cluster_flag:
                norm = gcn_cluster_norm(adjs, x.size(0), None, False, x.dtype)
            else:
                norm = None
            for i in range(self.layers):
                nvtx_push(self.gpu, "layer" + str(i))
                x = self.convs[i](x, adjs, norm=norm)
                if i != self.layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                nvtx_pop(self.gpu)
                log_memory(self.flag, device, 'layer' + str(i))
                
        return x # loss使用默认的交叉熵

    def inference(self, x_all, subgraph_loader):        
        pbar = tqdm(total=x_all.size(0) * self.layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, e_id, size = adj.to(self.device)
                x = x_all[n_id].to(self.device)
                x = self.convs[i](x, edge_index, size=size[1], norm=self.norm[e_id])
                if i != self.layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all




