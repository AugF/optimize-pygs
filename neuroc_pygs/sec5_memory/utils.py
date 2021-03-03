import os
import copy
import torch
import pandas as pd
from typing import List, Optional, Tuple, NamedTuple, Union, Callable
from torch import Tensor
from torch_sparse import SparseTensor
from neuroc_pygs.configs import PROJECT_PATH


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)

# 合并文件
def get_datasets(model_name='gat'):
    dir_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'sec5_2_memory_log')
    df = pd.read_csv(dir_path + '/{model_name}_datasets.csv', index_col=0)
    y = df['memory'].values
    del df['memory']
    X = df.values
    return X, y


def get_dataset_info():
    args = get_args()
    tab_data = []
    for exp_data in EXP_DATASET:
        args.dataset = exp_data
        data = build_dataset(args)
        feats, classes, nodes, edges = args.num_features, args.num_classes, data.num_nodes, data.num_edges 
        tab_data.append([args.dataset, nodes, edges, edges/nodes, feats, classes])

    print(tabulate(tab_data, headers=['name', 'nodes', 'edges', 'avg_degrees', 'feats', 'classes'], tablefmt="github"))


def get_adj(data, batch_size): # get same device
    N, E = data.num_nodes, data.num_edges
    adj = SparseTensor(
        row=data.edge_index[0], col=data.edge_index[1],
        value=torch.arange(E, device=data.edge_index.device),
        sparse_sizes=(N, N))
    print(adj)
    def sample_node(batch_size, sampler=''):
        if sampler == 'node':
            edge_sample = torch.randint(0, E, (batch_size, batch_size),
                                        dtype=torch.long)
            return adj.storage.row()[edge_sample]
        elif sampler == 'edge':
            row, col, _ = adj.coo()
            deg_in = 1. / adj.storage.colcount()
            deg_out = 1. / adj.storage.rowcount()
            prob = (1. / deg_in[row]) + (1. / deg_out[col])
            source_node_sample = col[edge_sample]
            target_node_sample = row[edge_sample]
            return torch.cat([source_node_sample, target_node_sample], -1)
        else:
            start = torch.randint(0, N, (batch_size, ), dtype=torch.long)
            node_idx = adj.random_walk(start.flatten(), walk_length=2)
            return node_idx.view(-1)

    node_idx = sample_node(batch_size).unique()
    adj, _ = adj.saint_subgraph(node_idx)
    print('new', adj)


def get_neighbor_sampler(data, size='1'):
    N, E = data.num_nodes, data.num_edges
    adj = SparseTensor(
        row=data.edge_index[0], col=data.edge_index[1],
        value=torch.arange(E, device=data.edge_index.device),
        sparse_sizes=(N, N))
    adj_t, n_id = adj_t.sample_adj(n_id, size, replace=False)
    e_id = adj_t.storage.value()
    size = adj_t.sparse_sizes()[::-1]    
    # get edge_index
    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)
    return EdgeIndex(edge_index, e_id, size)


from neuroc_pygs.options import get_args, build_dataset
data = build_dataset(get_args())
get_adj(data, 10)