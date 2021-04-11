import torch
import time

import torch.nn.functional as F
from torch.nn import Parameter, Module
from tqdm import tqdm

from neuroc_pygs.models.gaan_layers import GaANConv
from neuroc_pygs.utils import glorot, zeros, nvtx_push, nvtx_pop, log_memory
from neuroc_pygs.samplers.prefetch_generator import BackgroundGenerator


class GaAN(Module):
    """
    GaAN model
    dropout, negative_slop set: GaAN: Gated attention networks for learning on large and spatiotemporal graphs 5.3
    """
    def __init__(self, layers, n_features, n_classes, hidden_dims,
                 heads, d_v, d_a, d_m, dropout=0.1, negative_slop=0.1,
                 gpu=False, device="cpu", flag=False, infer_flag=False):
        super(GaAN, self).__init__()
        self.n_features, self.n_classes = n_features, n_classes
        self.layers, self.hidden_dims, self.heads = layers, hidden_dims, heads
        self.dropout, self.negative_slop = dropout, negative_slop
        self.d_v, self.d_a, self.d_m = d_v, d_a, d_m
        self.gpu = gpu
        self.device = device
        self.flag, self.infer_flag = flag, infer_flag

        shapes = [n_features] + [hidden_dims] * (layers - 1) + [n_classes]
        self.convs = torch.nn.ModuleList(
            [
                GaANConv(in_channels=shapes[layer], out_channels=shapes[layer + 1],
                         d_a=d_a, d_m=d_m, d_v=d_v, heads=heads, gpu=gpu)
                for layer in range(layers)
            ]
        )
        self.loss_fn, self.evaluator = None, None

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, adjs):
        device = torch.device(self.device)
        
        if isinstance(adjs, list):
            for i, (edge_index, _, size) in enumerate(adjs):
                nvtx_push(self.gpu, "layer" + str(i))
                x = self.convs[i](x, edge_index, size=size[1])
                if i != self.layers - 1:
                    x = F.leaky_relu(x, self.negative_slop)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                nvtx_pop(self.gpu)
                log_memory(self.flag, device, 'layer' + str(i))
        else:
            for i in range(self.layers):
                nvtx_push(self.gpu, "layer" + str(i))
                x = self.convs[i](x, adjs)
                if i != self.layers - 1:
                    x = F.leaky_relu(x, self.negative_slop)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                nvtx_pop(self.gpu)
                log_memory(self.flag, device, 'layer' + str(i))
                
        return x

    def inference(self, x_all, subgraph_loader, log_batch=False, opt_loader=False, df_time=None):
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

                    if i == 0 and df_time is not None:
                        df_time['sample'].append(sampling_time)
                        df_time['move'].append(to_time)
                        df_time['cal'].append(train_time)
                        df_time['cnt'][0] += 1
                        print(f"Batch:{df_time['cnt'][0]}, sample: {sampling_time}, move: {to_time}, cal: {train_time}")
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


    def inference_base(self, x_all, subgraph_loader, df):
        device = torch.device(self.device)
        
        loader_num = len(subgraph_loader)
        for i in range(self.layers):
            xs = []
            # 
            loader_iter = iter(subgraph_loader)
            for _ in range(loader_num):
                t0 = time.time()
                batch = next(loader_iter)
                batch_size, n_id, adj = batch
                t1 = time.time()
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                t2 = time.time()
                
                x = self.convs[i](x, edge_index, size=size[1])
                if i != self.layers - 1:
                    x = F.leaky_relu(x, self.negative_slop)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                xs.append(x.cpu())
                t3 = time.time()
                df.append([t1 - t0, t2 - t1, t3 - t2])
            x_all = torch.cat(xs, dim=0)
        return x_all


    def inference_cuda(self, x_all, subgraph_loader):
        device = torch.device(self.device)
        
        for i in range(self.layers):
            xs = []
            # 
            for batch in subgraph_loader:
                batch_size, n_id, adj = batch
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)

                x = self.convs[i](x, edge_index, size=size[1])
                if i != self.layers - 1:
                    x = F.leaky_relu(x, self.negative_slop)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
        return x_all

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
    
    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def __repr__(self):
        return '{}(layers={}, n_features={}, n_classes={}, hidden_dims={}, heads={},' \
               'd_v={}, d_a={}, d_m={}, dropout={}, negative_slop={}, gpu={})'.format(
            self.__class__.__name__, self.layers, self.n_features, self.n_classes,
            self.hidden_dims, self.heads, self.d_v, self.d_a, self.d_m, self.dropout,
            self.negative_slop, self.gpu) + '\nLayer(conv->leaky_relu->dropout)\n' + str(self.convs[0])

    def get_hyper_paras(self):
        return {
            'layers': self.layers,
            'n_features': self.n_features,
            'n_classses': self.n_classes,
            'hidden_dims': self.hidden_dims,
            'heads': self.heads,
            'd_v': self.d_v,
            'd_a': self.d_a,
            'd_m': self.d_m
        }





