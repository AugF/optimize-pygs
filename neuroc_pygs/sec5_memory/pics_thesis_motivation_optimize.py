import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
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

plt.style.use("grayscale")
plt.rcParams["font.size"] = 12

colors = ['black', 'white']
      
def run(predict_model='linear_model', bias=0.001):
    for model in ['gat', 'gcn']:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 7/2), tight_layout=True)
        for i, data in enumerate(['reddit', 'yelp']):
            ax = axes[i]
        
            ax.set_title(data, fontsize=14)
            ax.set_ylabel('峰值内存 (GB)', fontsize=14)
            ax.set_xlabel('批规模', fontsize=14)

            box_data = []
            
            if data == 'reddit' and model == 'gat':
                batch_sizes = [170, 175, 180]
            else:
                batch_sizes = [175, 180, 185]
            
            for bs in batch_sizes:
                for var in ['cluster', predict_model]:
                    if var == 'cluster':
                        real_path = f'out_motivation_data/{data}_{model}_{bs}_cluster.csv'
                    else:
                        real_path = f'out_{predict_model}_res/{data}_{model}_{bs}_{predict_model}.csv'
                    print(real_path)
                    if os.path.exists(real_path):
                        res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')['memory']
                    else:
                        res = []
                    box_data.append(list(map(lambda x: x/(1024*1024*1024), res)))
            bp = ax.boxplot(box_data, patch_artist=True)

            numBoxes = len(batch_sizes) * 2
            for i in range(numBoxes):
                if i % 2 == 1:
                    plt.setp(bp['medians'][i], color='red')
                    plt.setp(bp['boxes'][i], color='red')
                    plt.setp(bp['boxes'][i], facecolor=colors[1])
                    plt.setp(bp['fliers'][i], markeredgecolor='red')
                else:
                    plt.setp(bp['boxes'][i], facecolor=colors[0])
            
            ax.set_xticks([1.5, 3.5, 5.5])
            ax.set_xticklabels(batch_sizes, fontsize=14)

            if model == 'gcn':
                xlim = ax.get_xlim()
                ax.set_ylim(0, 7.6)
                ax.set_yticks([2, 4, 6, 7])
                line, = ax.plot(xlim, [6.5] * len(xlim), linestyle='dashed', color='b', linewidth=1.5, label='GPU内存上限')
            elif model == 'gat':
                xlim = ax.get_xlim()
                ax.set_ylim(0, 9.6)
                ax.set_yticks([2, 4, 6, 8, 9])
                line, = ax.plot(xlim, [8] * len(xlim), linestyle='dashed', color='b', linewidth=1.5, label='GPU内存上限')

            legend_colors = [Patch(facecolor=colors[0], edgecolor='black'), Patch(facecolor=colors[1], edgecolor='red')]
            ax.legend(legend_colors + [line], ['优化前', '优化后', 'GPU内存限制'], fontsize=10)

        fig.savefig(f'exp5_thesis_figs/exp_memory_training_{model}_cluster_motivation_{predict_model}.png', dpi=400)


if __name__ == '__main__':
    for predict_model in ['linear_model', 'random_forest']:
        run(predict_model)

