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
plt.rcParams.update(config)

base_size = 16
plt.style.use("grayscale")
plt.rcParams["font.size"] = base_size

mode = 'cluster'
for model in ['gcn', 'gat']:
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(7, 7/2), tight_layout=True)
    for i, data in enumerate(['reddit', 'yelp']):
        ax = axes[i]

        ax.set_title(data, fontsize=base_size + 2)
        ax.set_ylabel('边的数目 (k)', fontsize=base_size + 2)
        ax.set_xlabel('批规模', fontsize=base_size + 2)

        box_data = []

        if data == 'reddit' and model == 'gat':
            batch_sizes = [170, 175, 180]
        else:
            batch_sizes = [175, 180, 185]

        for bs in batch_sizes:
            # read file
            file_name = '_'.join([data, model, str(bs), mode, 'v2'])
            real_path = os.path.join(
                PROJECT_PATH, 'sec5_memory/exp_motivation_diff', file_name) + '.csv'
            print(real_path)
            if os.path.exists(real_path):
                res = pd.read_csv(real_path, index_col=0).to_dict(
                    orient='list')['edges']
                print(file_name, res)
            else:
                res = []
            box_data.append([r / 1000 for r in res])
        bp = ax.boxplot(box_data, labels=batch_sizes)
        ax.set_xticklabels(batch_sizes, fontsize=base_size+2)

    fig.savefig(f'exp5_thesis_figs/motivation/exp_memory_training_{model}_{mode}_motivation_edges_diff.png', dpi=400)
