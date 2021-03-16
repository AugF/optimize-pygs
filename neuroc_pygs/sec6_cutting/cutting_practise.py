# 二分法确定需要删除多少边
class BSearch(object):
    def __init__(self, clf, memory_limit, ratio=0.0):
        # clf: 预测模型; ratio: 偏差比例
        self.clf, self.ratio, self.memory_limit = clf, ratio, memory_limit
    
    def get_proper_edges(self, nodes, edges):
        l, r = 0, edges
        while (l < r):
            mid = (l + r + 1) // 2
            if (self.clf.predict([[nodes, mid]]) * (1 + self.ratio) <= self.memory_limit):
                l = mid
            else:
                r = mid - 1
        return l


