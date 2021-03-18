import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.options import get_args
from matplotlib.font_manager import _rebuild
_rebuild()

plt.style.use("grayscale")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams["font.size"] = 14

mode = 'cluster'
for model in ['gcn', 'gat']:
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(7, 7/2), tight_layout=True)
    for i, data in enumerate(['reddit', 'yelp']):
        ax = axes[i]

        ax.set_title(data.capitalize())
        ax.set_ylabel('峰值内存 (GB)', fontsize=14)
        ax.set_xlabel('批规模', fontsize=14)

        box_data = []

        if data == 'reddit' and model == 'gat':
            batch_sizes = [170, 175, 180]
        else:
            batch_sizes = [175, 180, 185]

        for bs in batch_sizes:
            # read file
            file_name = '_'.join([data, model, str(bs), mode, 'v2'])
            real_path = os.path.join(
                PROJECT_PATH, 'sec5_memory/exp_motivation_final', file_name) + '.csv'
            print(real_path)
            if os.path.exists(real_path):
                res = pd.read_csv(real_path, index_col=0).to_dict(
                    orient='list')['memory']
                print(file_name, res)
            else:
                res = []
            box_data.append(list(map(lambda x: x/(1024*1024*1024), res)))
        ax.boxplot(box_data, labels=batch_sizes)

        if model == 'gcn':
            xlim = ax.get_xlim()
            ax.set_ylim(0, 7.6)
            ax.set_yticks([2, 4, 6, 7])
            ax.plot(xlim, [6.5] * len(xlim), linestyle='dashed', color='b', linewidth=1.5, label='GPU内存上限')
        elif model == 'gat':
            xlim = ax.get_xlim()
            ax.set_ylim(0, 9.6)
            ax.set_yticks([2, 4, 6, 8, 9])
            ax.plot(xlim, [8] * len(xlim), linestyle='dashed', color='b', linewidth=1.5, label='GPU内存上限')
        ax.legend(fontsize=12)
    fig.savefig(os.path.join(PROJECT_PATH, 'sec5_memory',
                             'exp_figs', f'{model}_{mode}_motivation_final.png'))
