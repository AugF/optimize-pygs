import copy
import random
import torch
import time
import sys

from collections import defaultdict
from tabulate import tabulate
from torch_sparse import cat

from ogb.nodeproppred import PygNodePropPredDataset
from optimize_pygs.data.cluster import ClusterData, ClusterLoader
from optimize_pygs.datasets import build_dataset
from optimize_pygs.global_configs import dataset_root as root
from optimize_pygs.utils.utils import tabulate_results, build_args_from_dict

class ClusterOptimizerLoader(torch.utils.data.DataLoader):
    def __init__(self, cluster_data, **kwargs):
        self.cluster_data = cluster_data

        super(ClusterOptimizerLoader,
              self).__init__(range(len(cluster_data)),
                             collate_fn=self.__collate__, **kwargs)

    def __collate__(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        N = self.cluster_data.data.num_nodes
        E = self.cluster_data.data.num_edges

        start = self.cluster_data.partptr[batch].tolist()
        end = self.cluster_data.partptr[batch + 1].tolist()
        node_idx = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])

        data = copy.copy(self.cluster_data.data)
        if hasattr(data, '__num_nodes__'):
            del data.__num_nodes__
        adj, data.adj = self.cluster_data.data.adj, None

        adj = adj.index_select(0, node_idx).index_select(1, node_idx)
        row, col, edge_idx = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)

        for key, item in data:
            if isinstance(item, torch.Tensor) and item.size(0) == N:
                data[key] = item[node_idx]
            elif isinstance(item, torch.Tensor) and item.size(0) == E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        return data


def test_correct():
    print("test correctness")
    dataset = PygNodePropPredDataset(name='ogbn-products', root=root)
    data = dataset[0]
    cluster_data = ClusterData(data, num_parts=15000, recursive=False, save_dir=dataset.processed_dir)
    adj = cluster_data.data.adj
    
    tab_data = []
    for batch_size in [10, 50, 100, 200]: 
        batch = random.sample(range(0, len(cluster_data.partptr) - 1), batch_size)
        batch = torch.tensor(batch)
        # 1. ClusterLoader
        start = cluster_data.partptr[batch].tolist()
        end = cluster_data.partptr[batch + 1].tolist()
        node_idx = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])
        out1 = cat([adj.narrow(0, s, e - s) for s, e in zip(start, end)], dim=0)
        out1 = out1.coo()
        
        # 2. ClusterOptimizerLoader
        out2 = adj.index_select(0, node_idx)
        out2 = out2.coo()
        
        res = [batch_size]
        for o1, o2 in zip(out1, out2):
            res.append(o1.equal(o2))
        tab_data.append(res)
    print(tabulate(tab_data, headers=["Batch Size", "Row", "Col", "EdgeIndex"], tablefmt="github"))


def test_efficiency(datasets, batch_size=20):
    print(f"{sys._getframe().f_code.co_name}, batch_size={batch_size}")
    results_dict = defaultdict(list)
    for name in datasets:
        dataset = build_dataset(build_args_from_dict({'dataset': name}))
        data = dataset[0]
        nodes, edges = data.num_nodes, data.num_edges
        cluster_data = ClusterData(data, num_parts=1500, recursive=False, save_dir=dataset.processed_dir)
        adj = cluster_data.data.adj
        print(name, nodes, edges)

        for _ in range(5):
            # prepare
            batch = random.sample(range(0, len(cluster_data.partptr) - 1), batch_size)
            batch = torch.tensor(batch)
            start = cluster_data.partptr[batch].tolist()
            end = cluster_data.partptr[batch + 1].tolist()
            node_idx = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])
            
            # test
            t1 = time.time()
            out1 = cat([adj.narrow(0, s, e - s) for s, e in zip(start, end)], dim=0)
            t2 = time.time()
            out2 = adj.index_select(0, node_idx)
            t3 = time.time()
            base_time, optimize_time = t2 - t1, t3 - t2
            ratio = 100 * (base_time - optimize_time) / base_time
            results_dict[(name, nodes, edges)].append({'Base Time(s)': base_time, 'Optmize Time(s)': optimize_time, 'Ratio(%)': ratio})
    
    tab_data = tabulate_results(results_dict)
    print(tabulate(tab_data, headers=["(Dataset, nodes, edges)", "Base Time(s)", "Optmize Time(s)", "Ratio(%)"], tablefmt="github"))


def test_loader_efficiency(datasets, batch_size=20, num_workers=0):    
    print(f"{sys._getframe().f_code.co_name}, batch_size={batch_size}, num_workers={num_workers}")
    results_dict = defaultdict(list)
    for name in datasets:
        dataset = build_dataset(build_args_from_dict({'dataset': name}))
        data = dataset[0]
        nodes, edges = data.num_nodes, data.num_edges
        cluster_data = ClusterData(data, num_parts=1500, recursive=False, save_dir=dataset.processed_dir)
        print(name, nodes, edges)
        for _ in range(5):
            # default下，num_workers为0，表示只保留在主线程
            # 单线程测试
            loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True, num_workers=num_workers) 
            optimizer_loader = ClusterOptimizerLoader(cluster_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            t1 = time.time()
            for _ in loader: pass
            t2 = time.time()
            for _ in optimizer_loader: pass
            t3 = time.time()
            base_time, optimize_time = t2 - t1, t3 - t2
            ratio = 100 * (base_time - optimize_time) / base_time
            results_dict[(name, nodes, edges)].append({'Base Time(s)': base_time, 'Optmize Time(s)': optimize_time, 'Ratio(%)': ratio})
    
    tab_data = tabulate_results(results_dict)
    print(tabulate(tab_data, headers=["(Dataset, nodes, edges)", "Base Time(s)", "Optmize Time(s)", "Ratio(%)"], tablefmt="github"))
    

if __name__ == "__main__":    
    graphsaint_data = ["ppi", "flickr", "reddit", "yelp", "amazon"]
    neuroc_data = ['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics']
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-dt', type=str, default='graphsaint')
    args = parser.parse_args()
    
    datasets = graphsaint_data if args.dataset == 'graphsaint' else neuroc_data
    print(datasets)
    print("测试Change模块")
    cluster_batchs = [15, 45, 90, 150, 375, 750]
    for batch_size in cluster_batchs:
        test_efficiency(datasets, batch_size=batch_size)
    
    print("测试Loader模块")
    test_loader_efficiency(datasets, num_workers=0)
    test_loader_efficiency(datasets, num_workers=12)
    for batch_size in cluster_batchs:
        test_loader_efficiency(datasets, batch_size=batch_size)
    
    
    