import time

import torch
import torch.nn.functional as F
import numpy as np

from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from code.models.sage import SAGE
from code.optimize_batch.utils import get_args, train_base, train_thread_2
from code.optimize_batch.train_thread_3 import train_thread_3

# ---- begin ----
# step1. get args
args = get_args(description="ogbn_products_sage_cluster")
print(args)
device = f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu'
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if 'cuda' in device:
    torch.cuda.manual_seed(args.seed)

device = torch.device(device)
dataset = PygNodePropPredDataset(name='ogbn-products', root="/home/wangzhaokang/wangyunpan/gnns-project/datasets")
split_idx = dataset.get_idx_split()
data = dataset[0]

for key, idx in split_idx.items():
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[idx] = True
    data[f'{key}_mask'] = mask


model = SAGE(data.x.size(-1), args.hidden_channels, dataset.num_classes,
                args.num_layers, args.dropout).to(device)

cluster_data = ClusterData(data, num_parts=args.num_partitions,
                            recursive=False, save_dir=dataset.processed_dir)

loader = ClusterLoader(cluster_data, batch_size=1500,
                        shuffle=True, num_workers=args.num_workers)

model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
t1 = time.time()
train_thread_3(model, loader, optimizer, device)
t2 = time.time()
print(f"use time: {t2 - t1}s")


def begin_test():
    # 开始测试性能: 
    print(f"ogbn_products: node, edge = {data.num_nodes}, {data.num_edges}")
    cluster_batchs = [150, 450, 900, 1500, 3750, 750]
    xticklabels = ['1%', '3%', '6%', '10%', '25%', '50%']

    method_names = ['train_base', 'train_thread_3', 'train_thread_2']

    for i, bs in enumerate(cluster_batchs):
    # if True:
        cluster_data = ClusterData(data, num_parts=args.num_partitions,
                                    recursive=False, save_dir=dataset.processed_dir)

        # loader = ClusterLoader(cluster_data, batch_size=args.batch_size,
        #                         shuffle=True, num_workers=args.num_workers)
        # cluster_data = ClusterData(data, num_parts=1500,
        #                             recursive=False, save_dir=dataset.processed_dir)

        loader = ClusterLoader(cluster_data, batch_size=bs,
                                shuffle=True, num_workers=args.num_workers)
        
        for j, train_method in enumerate([train_base, train_thread_3, train_thread_2]):
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            st1 = time.time()
            train_method(model, loader, optimizer, device)
            st2 = time.time()
            # print(f"method={method_names[j]}, use time: {st2 - st1}s")
            print(f"method={method_names[j]}, relative batch size={xticklabels[i]}, use time: {st2 - st1}s")
