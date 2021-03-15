import copy
import numpy as np
from collections import defaultdict

# 二分法确定需要删除多少边
class BSearch(object):
    def __init__(self, clf, ratio, memory_limit):
        # clf: 预测模型; ratio: 偏差比例
        self.clf, self.ratio, self.memory_limit = clf, ratio, memory_limit
    
    def get_proper_edges(self, nodes, edges):
        l, r = 0, edges
        while (l < r):
            mid = (l + r + 1) // 2
            if (self.clf.predict(nodes, mid) * (1 + self.ratio) <= self.memory_limit):
                l = mid
            else:
                r = mid - 1
        return l


def bsearch(f, ratio, edges, memory): 
    # 寻找小于等于的最大右端点
    # f->内存预测函数
    # edges->实际的边
    # ratio->内存预测函数的偏差
    # memory->内存使用上限
    l, r = 0, edges
    while (l < r):
        mid = (l + r + 1) // 2
        if (f(mid) * (1 + ratio) <= memory):
            l = mid
        else:
            r = mid - 1
    return l

def test_bsearch():
    a = [4, 5, 6, 6, 8, 7, 9]
    print(bsearch(f=lambda x: a[x], ratio=0, edges=len(a), memory=6))


# 删边的做法
# - input: batch
# batch_size, n_id, adjs = batch
# adjs = [Adj(edge_index;  e_id; sizes),]
# size=[4343, 1024] 
# 删除edge_index中的点时，同时考虑删去e_id对应的坐标
# 如果某个点都被删完了，考虑删去n_id中的点，并修改sizes中的大小
def get_degree(edge_index):
    degrees = defaultdict(int)
    for i in range(edge_index.shape[1]):
        degrees[int(edge_index[0][i])] += 1
        degrees[int(edge_index[1][i])] += 1
    return degrees


class CuttingStrategies(object):
    def __init__(self, batch):
        self.batch_size, self.n_id, self.adj = batch
        self.edge_index, self.e_id, self.sizes = self.adj
        self.degrees = get_degree(self.edge_index)
        self.nodes, self.edges = len(self.n_id), len(self.e_id)

    def random_cut(self, cutting_nums):
        pass

    def enlighten1_cut(self, cutting_nums):
        pass
    
    def enlighten2_cut(self, cutting_nums):
        pass

    def check_nodes(self, degrees):
        new_id = []
        for v in range(self.nodes):
            # print(v)
            if degrees[v] > 0:
                new_id.append(self.n_id[v])
        return new_id, [len(new_id), self.batch_size]
    
    def getting_batch(self, method, cutting_nums):
        if method == 'random':
            degrees = copy.deepcopy(self.degrees)
            outliers = np.random.choice(self.edges, cutting_nums)
            mask = list(set(range(self.edges)) - set(outliers))
            for idx in outliers:
                degrees[int(self.edge_index[0][idx])] -= 1
                degrees[int(self.edge_index[1][idx])] -= 1
            print(max(mask), self.edge_index.shape, self.e_id.shape)
            edge_index, e_id = self.edge_index[:,mask], self.e_id[mask]
            n_id, sizes = self.check_nodes(degrees)
            return edge_index, e_id, n_id, sizes # 简单版
            # batch_size, n_id, adjs 
            # [Adj(edge_index;  e_id; sizes),]
            # return [self.batch_size, n_id, Adj(edge_index, e_id, sizes)]
            # e_id这里不需要


from neuroc_pygs.options import build_subgraphloader, get_args, build_dataset
args = get_args()
data = build_dataset(args)
subgraphloader = build_subgraphloader(args, data)

batch = next(iter(subgraphloader))
print(batch)

cs = CuttingStrategies(batch)
edge_index = cs.edge_index
cnt = 0
# for i in range(edge_index.shape[1]):
#     v, w = edge_index[:,i]
#     # print(v, w)
#     if v not in cs.n_id or w not in cs.n_id:
#         cnt+=1 

# print(edge_index.max(), edge_index.min(), edge_index.shape)
# 边没有了, 删除对应的边就可以!!!
edge_index, e_id, n_id, sizes = cs.getting_batch('random', 10)
print(edge_index.shape, e_id.shape, len(n_id), sizes)