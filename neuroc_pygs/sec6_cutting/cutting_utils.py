
class BSearch(object):
    # 基于二分的超限子图规模预测方法
    def __init__(self, clf, memory_limit):
        # clf: 预测模型; ratio: 偏差比例
        self.clf, self.memory_limit = clf, memory_limit
    
    def get_cutting_nums(self, nodes, edges, ratio, cur_memory):
        l, r = 0, edges
        while (l < r):
            mid = (l + r + 1) // 2
            if (self.clf.predict([[nodes, mid]])[0] / (1 - ratio) <= self.memory_limit - cur_memory):
                l = mid
            else:
                r = mid - 1
        return edges - l
