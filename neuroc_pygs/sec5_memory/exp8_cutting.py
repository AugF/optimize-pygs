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
    for v, w in edge_index:
        degrees[v] += 1
        degrees[w] += 1
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
        for v in self.n_id:
            if degrees[v]:
                new_id.append(v)
        return new_id, [len(new_id), self.batch_size]
    
    def getting_batch(self, method, cutting_nums):
        if method == 'random':
            degrees = copy.deepcopy(self.degrees)
            outliers = np.random.choice(self.edges, cutting_nums)
            mask = list(set(range(self.edges)) - set(outliers))
            for idx in outliers:
                degrees[self.edge_index[idx][0]] -= 1
                degrees[self.edge_index[idx][1]] -= 1
            edge_index, e_id = self.edge_index[mask], self.e_id[mask]
            n_id, sizes = self.check_nodes(degrees)
            # batch_size, n_id, adjs 
            # [Adj(edge_index;  e_id; sizes),]
            #return [self.batch_size, n_id, Adj(edge_index, e_id, sizes)]
            # e_id这里不需要


edges = 10
cutting_nums = 3
outliers = np.random.choice(edges, cutting_nums)
mask = list(set(range(edges)) - set(outliers))
print(outliers, mask)