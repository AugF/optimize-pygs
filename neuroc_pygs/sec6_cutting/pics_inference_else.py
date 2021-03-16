# BatchSize下的内存使用的箱线图和时间,精度的图
# https://matplotlib.org/2.0.2/examples/pylab_examples/boxplot_demo2.html
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.options import get_args
from matplotlib.font_manager import _rebuild
_rebuild() 

plt.style.use("grayscale")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams["font.size"] = 14

colors = plt.get_cmap('Greys')(np.linspace(0.15, 0.85, 2))
index_dict = {
    'reddit_sage': [1, 1, 1, 2, 3],
    'cluster_gcn': [1, 1, 1, 1, 1]
}
titles = {'reddit_sage': 'reddit SAGE', 
          'cluster_gcn': 'ogbn-products GCN'}

for model in ['reddit_sage', 'cluster_gcn']:
    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    ax.set_title(titles[model])
    ax.set_ylabel('运行时间 (秒)', fontsize=16)
    ax.set_xlabel('批规模', fontsize=16)

    box_data = []

    batch_sizes = [1024, 2048, 4096, 8192, 16384]
    
    file_name = '_'.join([model, 'acc'])
    real_path = os.path.join(PROJECT_PATH, 'sec6_cutting/exp_res', file_name) + '.csv'
    print(real_path)
    df = pd.read_csv(real_path, index_col=0)
    df.index = df['0']
    del df['0']
    accs, use_times = [], []
    for i, bs in enumerate(batch_sizes):
        tmp = df.loc[bs]
        bs_len = index_dict[model][i]
        acc = np.mean([tmp[str(j)] for j in range(1, bs_len + 1)])
        use_time = np.mean([tmp[str(bs_len + j)] for j in range(1, bs_len + 1)])
        accs.append(acc)
        use_times.append(use_time)
    print(use_times)
    print(accs)
    x = np.arange(len(batch_sizes))
    ax.bar(x, use_times, edgecolor='black', hatch="///", color=colors[0])
    ax2 = ax.twinx()
    ax2.set_ylim(0, 1)
    ax2.plot(np.arange(len(batch_sizes)), accs, marker='o', markersize=10)
    ax2.set_ylabel('精度', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
        
    fig.savefig(os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_figs', f'{model}_inference_else.png'))

