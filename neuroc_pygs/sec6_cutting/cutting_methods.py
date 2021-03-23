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
    weights = np.arange(edge_index.shape[1])
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
        return 1 / d1 + 1/d2
    elif name == 'way2':
        return 1 / math.sqrt(d1 * d2)
    elif name == 'way3':
        return d1 + d2
    elif name == 'way4':
        return d1 * d2
    elif name == 'way5':
        return d1 * d2 / (d1 + d2)


def get_importance(v, name='degree', **args):
    if name == 'degree':
        return args['degree'][v]
    elif name == 'pagerank':
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

    outliers = [importance[-i][1] for i in range(1, cutting_nums+1)]
    mask = list(set(range(edge_index.shape[1])) - set(outliers))
    return edge_index[:, mask]


def cut_by_importance_method(edge_index, cutting_nums, method='degree', name='way1', degree=None, pr=None):
    importance = []
    for i in range(edge_index.shape[1]):
        v, w = int(edge_index[0][i]), int(edge_index[1][i])
        iv, iw = get_importance(v, name=method, degree=degree, pr=pr), get_importance(
            w, name=method, degree=degree, pr=pr)
        importance.append((do_method(iv, iw, name=name), i))
    importance.sort()

    outliers = [importance[-i][1] for i in range(1, cutting_nums+1)]
    mask = list(set(range(edge_index.shape[1])) - set(outliers))
    return edge_index[:, mask]



def test():
    edge_index = np.array([[0, 1], [0, 2], [1, 2], [2, 0], [3, 2]]).T
    print(edge_index)
    random_ = cut_by_random(edge_index, cutting_nums=2)
    degree1 = cut_by_importance(
        edge_index, cutting_nums=2, method='degree', name='way1')
    degree2 = cut_by_importance(
        edge_index, cutting_nums=2, method='degree', name='way2')
    pr1 = cut_by_importance(edge_index, cutting_nums=2,
                            method='pagerank', name='way1')
    pr2 = cut_by_importance(edge_index, cutting_nums=2,
                            method='pagerank', name='way2')

    # print(random_, degree1, degree2, pr1, pr2, sep='\n')

import time
import pandas as pd
from neuroc_pygs.options import get_args, build_dataset

dir_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec6_cutting/exp_cutting_res'
# 实验1
# rs = [0.01, 0.03, 0.06, 0.1, 0.2, 0.5]
# args = get_args()
# args.dataset = 'random_10k_100k'
# data = build_dataset(args)
# df = {}
# for r in rs:
#     cutting_nums = int(100000 * r)
#     t1 = time.time()
#     cut_by_random(data.edge_index, cutting_nums)
#     t2 = time.time()
#     cut_by_importance(data.edge_index, cutting_nums, method='degree', name='way3')
#     t3 = time.time()
#     cut_by_importance(data.edge_index, cutting_nums, method='pagerank', name='way4')
#     t4 = time.time()
#     df[r] = [t2 - t1, t3 - t2, t4 - t3]
#     print(r, df[r])
# pd.DataFrame(df, columns=['random', 'degree3', 'pr4']).to_csv(dir_path + f'/cutting_method_fix_edges_use_time.csv')

# 实验2
df = {}
edges = [10, 15, 30, 50, 70, 90]
args = get_args()
for e in edges:
    args.dataset = f'random_10k_{e}k'
    cutting_nums = 5000
    data = build_dataset(args)
    t1 = time.time()
    cut_by_random(data.edge_index, cutting_nums)
    t2 = time.time()
    cut_by_importance(data.edge_index, cutting_nums, method='degree', name='way3')
    t3 = time.time()
    cut_by_importance(data.edge_index, cutting_nums, method='pagerank', name='way4')
    t4 = time.time()
    df[e] = [t2 - t1, t3 - t2, t4 - t3]
    print(e, df[e])

pd.DataFrame(df, columns=['random', 'degree3', 'pr4']).to_csv(dir_path + f'/cutting_method_fix_cutting_nums_use_time.csv')