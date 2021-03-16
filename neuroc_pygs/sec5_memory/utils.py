import os
import copy
import torch
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, NamedTuple, Union, Callable
from torch import Tensor
from torch_sparse import SparseTensor
from sklearn.metrics import mean_squared_error
from neuroc_pygs.configs import PROJECT_PATH, EXP_DATASET


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


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


def get_metrics(y_pred, y_test):
    mse = mean_squared_error(y_pred, y_test)
    num = len(y_pred)
    bad_exps = 0
    max_bias = 0
    max_bias_percent = 0
    for i in range(num):
        if y_pred[i] >= y_test[i]:
            max_bias_percent = max(max_bias_percent, (y_pred[i] - y_test[i]) / y_test[i])
            max_bias = max(max_bias, y_pred[i] - y_test[i])
            bad_exps += 1
    return mse, bad_exps / num, max_bias, max_bias_percent


def get_automl_datasets(model='gcn'):
    # 之后再灵活变化来做
    train_datasets = ['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'reddit']
    test_datasets = ['flickr', 'yelp']

    x_train, y_train = [], []
    for data in train_datasets:
        file_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_motivation_datasets', f'{model}_{data}_automl_datasets.csv')
        df = pd.read_csv(file_path, index_col=0)
        y_train.extend(df['memory'].values / (1024 * 1024))
        del df['memory']
        x_train.extend(df.values.tolist())

    x_test, y_test = [], []
    for data in test_datasets:
        file_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_motivation_datasets', f'{model}_{data}_automl_datasets.csv')
        df = pd.read_csv(file_path, index_col=0)
        y_test.extend(df['memory'].values / (1024 * 1024))
        del df['memory']
        x_test.extend(df.values.tolist())     
    return x_train, y_train, x_test, y_test


def test():
    for model in ['gcn', 'gat']:
        print(model)
        x_train, y_train, x_test, y_test = get_automl_datasets(model)
        print(len(y_train), len(y_test))
        print(x_train[:2], y_train[:2])
    # gcn: 15100, 3940
    # gat: 11220 1700
    

if __name__ == '__main__':
    test()