import os.path as osp
import torch
import json
import torch.cuda.nvtx as nvtx

import torch_geometric.transforms as T
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops

from torch_geometric.datasets.planetoid import Planetoid
from torch_geometric.datasets.amazon import Amazon
from torch_geometric.datasets.coauthor import Coauthor
from optimize_pygs.datasets.custom_dataset import CustomDataset

df = {}
memory_labels = ["allocated_bytes.all.current", "allocated_bytes.all.peak",
                 "reserved_bytes.all.current", "reserved_bytes.all.peak"]


def nvtx_push(flag, info):  # ?是否会指定到不需要的flag上
    if flag:
        nvtx.range_push(info)


def nvtx_pop(flag):
    if flag:
        nvtx.range_pop()


def log_memory(flag, device, label):
    if flag:
        res = torch.cuda.memory_stats(device)
        torch.cuda.reset_max_memory_allocated(device)
        # print(res["allocated_bytes.all.current"])
        if label not in df.keys():
            df[label] = [[res[i] for i in memory_labels]]
        else:
            df[label].append([res[i] for i in memory_labels])


def gcn_norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)  # ? todo

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def gcn_cluster_norm(edge_index, num_nodes, edge_weight=None, improved=False,
                     dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    #fill_value = 1 if not improved else 2
    # edge_index, edge_weight = add_remaining_self_loops(
    #    edge_index, edge_weight, fill_value, num_nodes) # ? todo

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def get_datasets(name, normalize_features=False, transform=None):  # pyg15
    dataset_root = '/mnt/data/wangzhaokang/wangyunpan/datasets'
    if name in ["cora", "pubmed"]:
        path = osp.join(dataset_root)
        dataset = Planetoid(path, name, split='full')
    else:
        path = osp.join(dataset_root, name)
        if name in ["amazon-computers", "amazon-photo"]:
            dataset = Amazon(path, name[7:])
        elif name == "coauthor-physics":
            dataset = Coauthor(path, name[9:])
        else:
            dataset = CustomDataset(root=dataset_root, name=name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset

def get_split_by_file(file_path, nodes): # 通过读取roles.json文件来获取train, val, test mask
    with open(file_path) as f:
        role = json.load(f)

    train_mask = torch.zeros(nodes, dtype=torch.bool)
    train_mask[torch.tensor(role['tr'])] = True

    val_mask = torch.zeros(nodes, dtype=torch.bool)
    val_mask[torch.tensor(role['va'])] = True

    test_mask = torch.zeros(nodes, dtype=torch.bool)
    test_mask[torch.tensor(role['te'])] = True
    return train_mask, val_mask, test_mask