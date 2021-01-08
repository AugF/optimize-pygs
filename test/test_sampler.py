"""
Sampler背景下，精度与BatchSize变化的文件, fix_time
"""
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler

import sys
import time
import json
import argparse
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import os.path as osp
from gaan.models import GaAN
from ggnn.models import GGNN
from gat.models import GAT
from gcn.models import GCN
from logger import Logger

from utils import get_dataset, gcn_norm, normalize, get_split_by_file, nvtx_push, nvtx_pop, log_memory, small_datasets

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help="dataset: [cora, flickr, com-amazon, reddit, com-lj,"
                                                                    "amazon-computers, amazon-photo, coauthor-physics, pubmed]")

parser.add_argument('--model', type=str, default='gcn', help="gnn models: [gcn, ggnn, gat, gaan]")
parser.add_argument('--runs', type=int, default=1, help="total runs")
parser.add_argument('--epochs', type=int, default=100000, help="epochs for training")
parser.add_argument('--layers', type=int, default=2, help="layers for hidden layer")
parser.add_argument('--hidden_dims', type=int, default=64, help="hidden layer output dims")
parser.add_argument('--heads', type=int, default=8, help="gat or gaan model: heads")
parser.add_argument('--head_dims', type=int, default=8, help="gat model: head dims") # head_dims * heads = hidden_dims
parser.add_argument('--d_v', type=int, default=8, help="gaan model: vertex's dim") # d_v * heads = hidden_dims?
parser.add_argument('--d_a', type=int, default=8, help="gaan model: each vertex's dim in edge attention") # d_a = head_dims
parser.add_argument('--d_m', type=int, default=64, help="gaan model: gate: max aggregator's dim, default=64")
parser.add_argument('--x_sparse', action='store_true', default=False, help="whether to use data.x sparse version")

parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--device', type=str, default='cuda:0', help='[cpu, cuda:id]')
parser.add_argument('--cpu', action='store_true', default=False, help='use cpu, not use gpu')
parser.add_argument('--lr', type=float, default=0.01, help="adam's learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0005, help="adam's weight decay")
parser.add_argument('--no_record_shapes', action='store_false', default=True, help="nvtx or autograd's profile to record shape")
parser.add_argument('--json_path', type=str, default='', help="json file path for memory")
parser.add_argument('--mode', type=str, default='cluster', help='sampling: [cluster, graphsage]')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--batch_partitions', type=int, default=20, help='number of cluster partitions per batch')
parser.add_argument('--cluster_partitions', type=int, default=1500, help='number of cluster partitions')
parser.add_argument('--num_workers', type=int, default=40, help='number of Data Loader partitions')
parser.add_argument('--fix_time', type=int, default=800, help='fix use time')
args = parser.parse_args()
gpu = not args.cpu and torch.cuda.is_available()
flag = not args.json_path == ''

print(args)

# 0. set manual seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if gpu:
    torch.cuda.manual_seed(args.seed)

device = torch.device(args.device if gpu else 'cpu')

# 1. set datasets
dataset_info = args.dataset.split('_')
if dataset_info[0] in small_datasets and len(dataset_info) > 1:
    args.dataset = dataset_info[0]

dataset = get_dataset(args.dataset, normalize_features=True)
data = dataset[0]

# add train, val, test split
if args.dataset in ['amazon-computers', 'amazon-photo', 'coauthor-physics']:
    file_path = osp.join('/home/wangzhaokang/wangyunpan/gnns-project/datasets', args.dataset + "/raw/role.json")
    data.train_mask, data.val_mask, data.test_mask = get_split_by_file(file_path, data.num_nodes)

num_features = dataset.num_features
if dataset_info[0] in small_datasets and len(dataset_info) > 1:
    file_path = osp.join('/home/wangzhaokang/wangyunpan/gnns-project/datasets', "data/feats_x/" + '_'.join(dataset_info) + '_feats.npy')
    if osp.exists(file_path):
        data.x = torch.from_numpy(np.load(file_path)).to(torch.float) # 因为这里是随机生成的，不考虑normal features
        num_features = data.x.size(1)
    
# 2. set sampling
# 2.1 test_data
subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=1024,
                                  shuffle=False, num_workers=args.num_workers)


cluster_data = ClusterData(data, num_parts=args.cluster_partitions, recursive=False,
                        save_dir=dataset.processed_dir)
cluster_loader = ClusterLoader(cluster_data, batch_size=args.batch_partitions, shuffle=True,
                            num_workers=args.num_workers)

neighbor_loader = NeighborSampler(data.edge_index, node_idx=None,
                            sizes=[25, 10], batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers)

for run in range(3):
    print("run: ", run)
    # test trainloader
    names = ["cluster_sampler", "graphsage_sampler"]
    for i, loader in enumerate([cluster_loader, neighbor_loader]):
        print(names[i])
        for batch in loader:
            if i == 0:
                nodes, edges = batch.x.shape[0], batch.edge_index.shape[1]
            else:
                batch_size, n_id, adjs = batch
                nodes, edges = adjs[0][2][0], adjs[0][0].shape[1]
            print(nodes, edges)

    print("subgraph_loader")
    # test eval_loader
    for batch in subgraph_loader:
        batch_size, n_id, adjs = batch
        nodes, edges = adjs[0][2][0], adjs[0][0].shape[1]
        print(nodes, edges)