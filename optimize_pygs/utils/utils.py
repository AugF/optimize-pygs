import os
import itertools
import errno

import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F


class ArgClass(object):
    def __init__(self):
        pass


def build_args_from_dict(dic):
    args = ArgClass()
    for key, value in dic.items():
        args.__setattr__(key, value)
    return args


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e
        
def get_extra_args(args):
    redundancy = {
        "checkpoint": False,
        "load_emb_path": None,
    }
    args = {**args, **redundancy}
    return args


def add_remaining_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1]).to(device)
    if num_nodes is None:
        num_nodes = torch.max(edge_index) + 1
    if fill_value is None:
        fill_value = 1

    N = num_nodes
    row, col = edge_index[0], edge_index[1]
    mask = row != col

    loop_index = torch.arange(0, N, dtype=edge_index.dtype, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    inv_mask = ~mask

    loop_weight = torch.full((N,), fill_value, dtype=edge_weight.dtype, device=edge_weight.device)
    remaining_edge_weight = edge_weight[inv_mask]
    if remaining_edge_weight.numel() > 0:
        loop_weight[row[inv_mask]] = remaining_edge_weight
    edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)

    return edge_index, edge_weight


def mul_edge_softmax(indices, values, shape):
    """
    Args:
        indices: Tensor, shape=(2, E)
        values: Tensor, shape=(E, d)
        shape: tuple(int, int)
    Returns:
        Softmax values of multi-dimension edge values for nodes
    """
    device = values.device
    values = torch.exp(values)
    output = torch.zeros(shape[0], values.shape[1]).to(device)
    output = output.scatter_add_(0, indices[0].unsqueeze(-1).expand_as(values), values)
    softmax_values = values / (output[indices[0]] + 1e-8)
    softmax_values[torch.isnan(softmax_values)] = 0
    return softmax_values


def spmm(indices, values, b):
    r"""
    Args:
        indices : Tensor, shape=(2, E)
        values : Tensor, shape=(E,)
        shape : tuple(int ,int)
        b : Tensor, shape=(N, )
    """
    output = b.index_select(0, indices[1]) * values.unsqueeze(-1)
    output = torch.zeros_like(b).scatter_add_(0, indices[0].unsqueeze(-1).expand_as(output), output)
    return output


def get_degrees(indices, num_nodes=None):
    device = indices.device
    values = torch.ones(indices.shape[1]).to(device)
    if num_nodes is None:
        num_nodes = torch.max(values) + 1
    b = torch.ones((num_nodes, 1)).to(device)
    degrees = spmm(indices, values, b).view(-1)
    return degrees


def get_activation(act: str):
    if act == "relu":
        return F.relu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "gelu":
        return F.gelu
    elif act == "prelu":
        return F.prelu
    elif act == "identity":
        return lambda x: x
    else:
        return F.relu

def tabulate_results(results_dict):
    """
    results_dict = {
        'variant1': [1, 2, 3], 
        'variant2': [2, 3, 4]
    }
    """
    # Average for different seeds
    tab_data = []
    for variant in results_dict:
        results = np.array([list(res.values()) for res in results_dict[variant]])
        tab_data.append(
            [variant]
            + list(
                itertools.starmap(
                    lambda x, y: f"{x:.4f}±{y:.4f}",
                    zip(
                        np.mean(results, axis=0).tolist(),
                        np.std(results, axis=0).tolist(),
                    ),
                )
            )
        )
    return tab_data


def test_tabulate_results():
    results_dict = {
        ('cora', 'gcn'): [{'Acc': 0.818, 'ValAcc': 0.798},
                          {'Acc': 0.820, 'ValAcc': 0.788}],
        ('cora', 'gat'): [{'Acc': 0.818, 'ValAcc': 0.798},
                          {'Acc': 0.820, 'ValAcc': 0.788}],
        }
    print(tabulate_results(results_dict))

if __name__ == "__main__":
    test_tabulate_results()
    