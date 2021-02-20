import torch
import torch.cuda.nvtx as nvtx

from torch_scatter import scatter_add

df = {}
memory_labels = ["allocated_bytes.all.current", "allocated_bytes.all.peak", "reserved_bytes.all.current", "reserved_bytes.all.peak"]


def nvtx_push(flag, info): # ?是否会指定到不需要的flag上
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
        edge_index, edge_weight, fill_value, num_nodes) # ? todo 

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
    #edge_index, edge_weight = add_remaining_self_loops(
    #    edge_index, edge_weight, fill_value, num_nodes) # ? todo 

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    