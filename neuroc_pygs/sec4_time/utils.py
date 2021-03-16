import math

datasets_maps = {
    'amazon-photo': 'amp',
    'pubmed': 'pub',
    'amazon-computers': 'amc',
    'coauthor-physics': 'cph',
    'flickr': 'fli',
    'com-amazon': 'cam'
}

algorithms = {
    'gcn': 'GCN',
    'ggnn': 'GGNN',
    'gat': 'GAT',
    'gaan': 'GaAN'
}

sampling_modes = {
    'graphsage': 'GraphSAGE',
    'cluster': 'Cluster-GCN'
}


def turkey(total_times):
    total_times.sort()
    n = len(total_times)
    x, y = (n + 1) * 0.25, (n + 1) * 0.75
    tx, ty = math.floor(x), math.floor(y)
    Q1 = total_times[tx - 1] * (x - tx) + total_times[tx] * (1 - x + tx)
    Q3 = total_times[ty - 1] * (y - ty) + total_times[ty] * (1 - y + ty)
    min_val, max_val = Q1 - 1.5 * (Q3 - Q1), Q3 + 1.5 * (Q3 - Q1)
    inliers = []
    for i, x in enumerate(total_times):
        if x >= min_val and x <= max_val:
            inliers.append(i)
    return inliers



