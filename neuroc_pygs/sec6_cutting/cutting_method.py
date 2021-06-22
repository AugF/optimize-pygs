import copy
import math
import torch
import numpy as np
from scipy import sparse
from fast_pagerank import pagerank
from collections import defaultdict


def get_degree(edge_index):
    degrees = defaultdict(int)
    for i in range(edge_index.shape[1]):
        degrees[int(edge_index[0][i])] += 1
        degrees[int(edge_index[1][i])] += 1
    return degrees


def get_pagerank(edge_index):
    # weights = np.arange(edge_index.shape[1])
    np.random.seed(1)
    weights = np.random.random(edge_index.shape[1])
    nodes = int(torch.max(edge_index)) + 1
    G = sparse.csr_matrix(
        (weights, (edge_index[0, :], edge_index[1, :])), shape=(nodes, nodes))
    return pagerank(G, p=0.85)


def cut_by_random(edge_index, cutting_nums, seed=2):
    np.random.seed(seed)
    outliers = np.random.choice(edge_index.shape[1], cutting_nums, replace=False)
    mask = list(set(range(edge_index.shape[1])) - set(outliers))
    return edge_index[:, mask]


def do_method(d1, d2, name='way1'):
    if name == 'way1':
        return 1 / (d1 + d2)
    elif name == 'way2':
        return 1 / (d1 * d2)


def get_importance(v, name='degree', **args):
    if name == 'degree':
        return args['degree'][v]
    else:
        return args['pr'][v]


def cut_by_importance(edge_index, cutting_nums, method='degree', name='way1'):
    if method == 'degree':
        degrees, pr = get_degree(edge_index), None
    else:
        degrees, pr = None, get_pagerank(edge_index)

    importance = []
    for i in range(edge_index.shape[1]):
        v, w = int(edge_index[0][i]), int(edge_index[1][i])
        iv, iw = get_importance(v, name=method, degree=degrees, pr=pr), get_importance(
            w, name=method, degree=degrees, pr=pr)
        importance.append((do_method(iv, iw, name=name), i))
    importance.sort()

    outliers = [importance[i][1] for i in range(cutting_nums)]
    mask = list(set(range(edge_index.shape[1])) - set(outliers))
    print(outliers)
    return edge_index[:, mask]


def cut_by_importance_method(edge_index, cutting_nums, method='degree', name='way1', degree=None, pr=None):
    importance = []
    for i in range(edge_index.shape[1]):
        v, w = int(edge_index[0][i]), int(edge_index[1][i])
        iv, iw = get_importance(v, name=method, degree=degree, pr=pr), get_importance(
            w, name=method, degree=degree, pr=pr)
        importance.append((do_method(iv, iw, name=name), i))
    importance.sort()

    outliers = [importance[i][1] for i in range(1, cutting_nums+1)]
    mask = list(set(range(edge_index.shape[1])) - set(outliers))
    return edge_index[:, mask]


def run_overhead(): # 计算额外开销，对应于大论文图5-19
    import time
    import pandas as pd
    from neuroc_pygs.options import get_args, build_dataset

    # 实验1
    rs = [0.01, 0.03, 0.06, 0.1, 0.2, 0.5]
    args = get_args()
    args.dataset = 'random_10k_100k'
    data = build_dataset(args)
    for r in rs:
        cutting_nums = int(100000 * r)
        t1 = time.time()
        cut_by_random(data.edge_index, cutting_nums)
        t2 = time.time()
        cut_by_importance(data.edge_index, cutting_nums, method='degree', name='way1')
        t3 = time.time()
        cut_by_importance(data.edge_index, cutting_nums, method='pagerank', name='way2')
        t4 = time.time()
        df[r] = [t2 - t1, t3 - t2, t4 - t3]
        print(r, df[r])

    # 实验2
    edges = [10, 15, 30, 50, 70, 90]
    args = get_args()
    for e in edges:
        args.dataset = f'random_10k_{e}k'
        cutting_nums = 10
        data = build_dataset(args)
        t1 = time.time()
        cut_by_random(data.edge_index, cutting_nums)
        t2 = time.time()
        cut_by_importance(data.edge_index, cutting_nums, method='degree', name='way1')
        t3 = time.time()
        cut_by_importance(data.edge_index, cutting_nums, method='pagerank', name='way2')
        t4 = time.time()
        df[e] = [t2 - t1, t3 - t2, t4 - t3]
        print(e, df[e])


if __name__ == '__main__':
    run_overhead()