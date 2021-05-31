import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from neuroc_pygs.sec4_time.utils import datasets_maps, algorithms, sampling_modes
from matplotlib.font_manager import _rebuild
_rebuild()
config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)


def float_x(x):
    return [float(i) for i in x]


base_size = 12
plt.style.use("grayscale")

plt.rcParams["font.size"] = base_size


datasets_maps = {
    'amazon-photo': 'amp',
    'pubmed': 'pub',
    'amazon-computers': 'amc',
    'coauthor-physics': 'cph',
    'flickr': 'fli',
    'com-amazon': 'cam'
}

file_names = ['gcn', 'gat']
# vs = ['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
datasets = ['pub', 'amc', 'fli', 'red']
modes = ['graphsage', 'cluster']
titles = {
    'gcn': 'GCN', 'ggnn': 'GGNN', 'gat': 'GAT', 'gaan': 'GaAN'
}
MODES = ['邻居采样', '聚类采样']
algs = ['gcn', 'gaan']
linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (5, 5)), (0, (3, 10, 1, 10))]

for i, mode in enumerate(['cluster']):
    df = {}
    for alg in algs:
        df[alg] = {'Baseline': [], 'Optimize': [], 'real_ratio': [],
                   'exp_ratio': [], 'r1': [], 'y': [], 'z': []}

    for data in datasets:
        df_data = pd.read_csv(
            f'out_batch_csv/models_{mode}_{data}.csv', index_col=0)
        for alg in algs:
            df[alg]['Baseline'].append(float(df_data['baseline'][alg]))
            df[alg]['Optimize'].append(float(df_data['opt'][alg]))
            df[alg]['real_ratio'].append(1/float(df_data['real_ratio'][alg]))
            df[alg]['exp_ratio'].append(1/float(df_data['exp_ratio'][alg]))
            df[alg]['r1'].append(1/float(df_data['r1'][alg]))
            df[alg]['y'].append(100 * float(df_data['y'][alg]))
            df[alg]['z'].append(100 * float(df_data['z'][alg]))

    base_size = 14
    for alg in algs:
        tab_data = df[alg]

        xs = datasets
        x = np.arange(len(xs))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
        ax.set_title(titles[alg], fontsize=base_size+2)
        ax.set_ylabel('比值', fontsize=base_size+2)
        ax.set_xlabel('数据集', fontsize=base_size+2)
        line3, = ax.plot(x, tab_data['r1'], '^r', label='优化效果', linestyle=linestyles[0])

        ax2 = ax.twinx()
        ax2.set_ylabel("50批次训练时间 (ms)", fontsize=base_size + 2)
        
        line4, = ax2.plot(x, [1000*t for t in tab_data['Baseline']], 'ob', linestyle=linestyles[1],
                          label='优化前耗时')
        line5, = ax2.plot(x, [1000*t for t in tab_data['Optimize']], 'Dg', linestyle=linestyles[2],
                           label='优化后耗时')
        plt.legend(handles=[line3,
                            line4, line5])
        plt.xticks(ticks=x, labels=xs, fontsize=base_size + 2)
        plt.yticks(fontsize=base_size + 2)
        fig.tight_layout()  # 防止重叠

        fig.savefig(
            f'exp4_thesis_ppt_figs/batch_figs/exp_batch_datasets_{mode}_else_{alg}.png', dpi=400)
