
class BSearch(object):
    def __init__(self, clf, memory_limit, ratio=0.0):
        # clf: 预测模型; ratio: 偏差比例
        self.clf, self.ratio, self.memory_limit = clf, ratio, memory_limit
    
    def get_cutting_nums(self, nodes, edges, current_memory=0):
        l, r = 0, edges
        while (l < r):
            mid = (l + r + 1) // 2
            if ((self.clf.predict([[nodes, mid]]) / (1 - self.ratio) + current_memory) <= self.memory_limit):
                l = mid
            else:
                r = mid - 1
        return edges - l


def test():
    from neuroc_pygs.options import get_args, build_dataset, build_subgraphloader
    from joblib import load

    dir_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec6_cutting/exp_diff_res'
    args = get_args()
    data = build_dataset(args)
    subgraphloader = build_subgraphloader(args, data)

    model = 'cluster_gcn'
    batch = iter(subgraphloader).next()
    batch_size, _, adj = batch
    edge_index, _, size = adj 

    node, edge = size[0], edge_index.shape[1]
    reg = load(dir_path + f'/{model}_linear_model_v0.pth')
    res = reg.predict([[node, edge]])
    memory_limit = 4000000

    bs = BSearch(reg, memory_limit)
    cutting_nums = bs.get_cutting_nums(node, edge)
    print(cutting_nums, reg.predict([[node, edge - cutting_nums]]))