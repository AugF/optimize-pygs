# BatchSize下的内存使用的箱线图和时间
# https://matplotlib.org/2.0.2/examples/pylab_examples/boxplot_demo2.html
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.options import get_args
from matplotlib.font_manager import _rebuild
_rebuild()
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimHei'],
}

base_size = 16
plt.style.use("grayscale")
plt.rcParams["font.size"] = base_size

colors = plt.get_cmap('Greys')(np.linspace(0.15, 0.85, 2))
mode = 'cluster'

titles = {'reddit_sage': 'SAGE reddit', 
          'cluster_gcn': 'ClusterGCN ogbn-products'}

fig, axes = plt.subplots(
    nrows=1, ncols=2, figsize=(7, 7/2), tight_layout=True)
   
for i, model in enumerate(['reddit_sage', 'cluster_gcn']):
    ax = axes[i]
    ax.set_title(titles[model], fontsize=base_size + 2)
    ax.set_ylabel('峰值内存 (GB)', fontsize=base_size + 2)
    ax.set_xlabel('批规模', fontsize=base_size + 2)

    box_data = []

    if model == 'reddit_sage':
        batch_sizes = [8700, 8800, 8900]
    else:
        batch_sizes = [9000, 9100, 9200]
    
    for bs in batch_sizes:
        # read file
        file_name = '_'.join([model, str(bs)])
        real_path = os.path.join(PROJECT_PATH, 'sec6_cutting/out_motivation_data', file_name) + '.csv'
        print(real_path)
        if os.path.exists(real_path):
            res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')['memory']
        else:
            res = []
        box_data.append(list(map(lambda x: x/(1024*1024*1024), res)))
    bp = ax.boxplot(box_data, labels=batch_sizes)
    ax.set_xticklabels(batch_sizes, fontsize=base_size + 2)

    xlim = ax.get_xlim()
    line = 3 if model == 'reddit_sage' else 2
    ax.plot(xlim, [line] * len(xlim), linestyle='dashed', color='b', linewidth=1.5, label='GPU内存上限')
    ax.legend()

fig.savefig(f'exp6_thesis_figs/exp_memory_inference_motivation.png', dpi=400)



