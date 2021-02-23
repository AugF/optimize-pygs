from typing import List

import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.utils.data


class Sampler:
    r"""
    Base sampler class. 
    Constructs a sampler with data (`torch.geometric.data.Data`), which indicates Graph to be sampled, 
    and args_params (Dictionary) with respresents args parameters by the sampler.
    """
    def __init__(self, data, args_params):
        self.data = data.clone()
        self.num_nodes = self.data.x.size()[0]
        self.num_edges = (
            self.data.edge_index_train.size()[1]
            if hasattr(self.data, "edge_index_train")
            else self.data.edge_index.size()[1]
        )
    
    def sample(self):
        pass


class LayerSamper(Sampler):
    def __init__(self, data, model, params_args):
        super().__init__(data, params_args)
        self.model = model
        # 这里与model相关
        self.sample_one_layer = self.model.__sample_one_layer
        self.sample = self.model.sampling
    
    def get_batches(self, train_nodes, train_labels, batch_size=64, shuffle=True):
        if shuffle:
            random.shuffle(train_nodes)
        total = train_nodes.shape[0]
        for i in range(0, total, batch_size):
            if i + batch_size <= total:
                cur_nodes = train_nodes[i : i + batch_size]
                cur_labels = train_labels[cur_nodes]
                yield cur_nodes, cur_labels


class FastGCNSampler(LayerSampler):
    def __init__(self, data, params_args):
        super().__init__(data, params_args)
    def generate_adj(self, sample1, sample2):
        edgelist = []
        mapping = {}
        for i in range(len(sample1)):
            mapping[sample1[i]] = i
        for i in range(len(sample2)):
            nodes = self.adj[sample2[i]]
            for node in nodes:
                if node in mapping:
                    edgelist.append([mapping[node], i])
        edgetensor = torch.LongTensor(edgelist)
        valuetensor = torch.ones(edgetensor.shape[0]).float()
        t = torch.sparse_coo_tensor(
            edgetensor.t(), valuetensor, (len(sample1), len(sample2))
        )
        return t
    def sample_one_layer(self, sampled, sample_size):
        total = []
        for node in sampled:
            total.extend(self.adj[node])
        total = list(set(total))
        if sample_size < len(total):
            total = random.sample(total, sample_size)
        return total
    def sample(self, x, v, num_layers):
        all_support = [[] for _ in range(num_layers)]
        sampled = v.detach().cpu().numpy()
        for i in range(num_layers - 1, -1, -1):
            cur_sampled = self.sample_one_layer(sampled, self.sample_size[i])
            all_support[i] = self.generate_adj(sampled, cur_sampled).to(x.device)
            sampled = cur_sampled
        return x[torch.LongTensor(sampled).to(x.device)], all_support, 0


class ASGCNSampler(LayerSampler):
    def __init__(self, data, params_args):
        super().__init__(data, params_args)
    def set_w(self, w_s0, w_s1):
        self.w_s0 = w_s0
        self.w_s1 = w_s1
    def set_adj(self, edge_index, num_nodes):
        self.sparse_adj = sp.coo_matrix(
            (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        self.num_nodes = num_nodes
        self.adj = self.compute_adjlist(self.sparse_adj)
        self.adj = torch.tensor(self.adj)
    def compute_adjlist(self, sp_adj, max_degree=32):
        num_data = sp_adj.shape[0]
        adj = num_data + np.zeros((num_data+1, max_degree), dtype=np.int32)
        for v in range(num_data):
            neighbors = np.nonzero(sp_adj[v, :])[1]
            len_neighbors = len(neighbors)
            if len_neighbors > max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=False)
                adj[v] = neighbors
            else:
                adj[v, :len_neighbors] = neighbors
        return adj
    def from_adjlist(self, adj):
        u_sampled, index = torch.unique(torch.flatten(adj), return_inverse=True)
        row = (torch.range(0, index.shape[0]-1) / adj.shape[1]).long().to(adj.device)
        col = index
        values = torch.ones(index.shape[0]).float().to(adj.device)
        indices = torch.cat([row.unsqueeze(1), col.unsqueeze(1)], axis=1).t()
        dense_shape = (adj.shape[0], u_sampled.shape[0])
        support = torch.sparse_coo_tensor(indices, values, dense_shape)
        return support, u_sampled.long()
    def _sample_one_layer(self, x, adj, v, sample_size):
        support, u = self.from_adjlist(adj)
        h_v = torch.sum(torch.matmul(x[v], self.w_s1))
        h_u = torch.matmul(x[u], self.w_s0)
        attention = (F.relu(h_v + h_u) + 1) * (1.0 / sample_size)
        g_u = F.relu(h_u) + 1
        p1 = attention * g_u
        p1 = p1.cpu()
        if self.num_nodes in u:
            p1[u == self.num_nodes] = 0
        p1 = p1 / torch.sum(p1)
        samples = torch.multinomial(p1, sample_size, False)
        u_sampled = u[samples]
        support_sampled = torch.index_select(support, 1, samples)
        return u_sampled, support_sampled
    def sample(self, x, v, num_layers):
        all_support = [[] for _ in range(num_layers)]
        sampled = v
        x = torch.cat((x, torch.zeros(1, x.shape[1]).to(x.device)), dim=0)
        for i in range(num_layers - 1, -1, -1):
            cur_sampled, cur_support = self.sample_one_layer(x, self.adj[sampled], sampled, self.sample_size[i])
            all_support[i] = cur_support.to(x.device)
            sampled = cur_sampled
        return x[sampled.to(x.device)], all_support, 0
