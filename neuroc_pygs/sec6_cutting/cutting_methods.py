import copy
import math
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
    nodes = np.max(edge_index) + 1
    G = sparse.csr_matrix(
        (weights, (edge_index[0, :], edge_index[1, :])), shape=(nodes, nodes))
    return pagerank(G, p=0.85)


def cut_by_random(edge_index, cutting_nums):
    outliers = np.random.choice(edge_index.shape[1], cutting_nums)
    mask = list(set(range(edge_index.shape[1])) - set(outliers))
    return edge_index[:, mask]


def do_method(d1, d2, name='way1'):
    if name == 'way1':
        return 1 / d1 + 1/d2
    elif name == 'way2':
        return 1 / math.sqrt(d1 * d2)


def get_importance(v, name='degree', **args):
    if name == 'degree':
        return args['degree'][v]
    elif name == 'pagerank':
        return args['pr'][v]


def cut_by_importance(edge_index, cutting_nums, method='degree', name='way1'):
    if method = 'degree':
        degrees = get_degree(edge_index)
    pr = get_pagerank(edge_index)

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


