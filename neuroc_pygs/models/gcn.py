import torch
import time
import sys

import torch.nn.functional as F
from torch.nn import Module

from neuroc_pygs.models.gcn_layers import GCNConv
from neuroc_pygs.utils import glorot, zeros, nvtx_push, nvtx_pop, log_memory, gcn_cluster_norm


class GCN(Module):
    """
    GCN layer
    dropout set: https://github.com/tkipf/pygcn/blob/master/pygcn/train.py
    """
    def __init__(self, layers, n_features, n_classes, hidden_dims, norm=None, dropout=0.5,
                 gpu=False, device="cpu", flag=False, infer_flag=False, cluster_flag=False,
                 cached_flag=True): # add adj
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
        self.loss_fn, self.evaluator = None, None
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            
    def forward(self, x, adjs):
        """
        修改意见：https://github.com/THUDM/cogdl/blob/master/cogdl/models/nn/pyg_gcn.py
        :param x:
        :param edge_index:
        :return:
        """
        device = torch.device(self.device)
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
                
        return x

    def inference(self, x_all, subgraph_loader, log_batch=False, opt_loader=False):
        device = torch.device(self.device)
        flag = self.infer_flag
        
        sampling_time, to_time, train_time = 0.0, 0.0, 0.0
        total_batches = len(subgraph_loader)

        log_memory(flag, device, 'inference start')       
        for i in range(self.layers):
            log_memory(flag, device, f'layer{i} start')

            xs = []
            if opt_loader:
                loader_iter = BackgroundGenerator(iter(subgraph_loader))
            else:
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
                    if i != self.layers - 1:
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
        if log_batch:
            print(f"avg_batch_train_time: {train_time}, avg_batch_sampling_time:{sampling_time}, avg_batch_to_time: {to_time}")
        return x_all

    def inference_base(self, x_all, loader_iter, loader_num):
        device = torch.device(self.device)

        for i in range(self.layers):
            for _ in range(loader_num):
                batch_size, n_id, adj = next(loader_iter)

                edge_index, e_id, size = adj.to(device)
                x = x_all[n_id].to(device)

                x = self.convs[i](x, edge_index, size=size[1], norm=self.norm[e_id])
                if i != self.layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
        return x_all

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
    
    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def __repr__(self):
        return '{}(layers={}, n_features={}, n_classes={}, hidden_dims={}, dropout={}, gpu={})'.format(
            self.__class__.__name__, self.layers, self.n_features, self.n_classes, self.hidden_dims,
            self.dropout, self.gpu) + '\nLayer(conv->relu->dropout)\n' + str(self.convs)

    def get_hyper_paras(self):
        return {
            'layers': self.layers,
            'n_features': self.n_features,
            'n_classses': self.n_classes,
            'hidden_dims': self.hidden_dims,
        }

