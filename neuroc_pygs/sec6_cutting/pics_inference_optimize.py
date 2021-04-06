# BatchSize下的内存使用的箱线图和时间
# https://matplotlib.org/2.0.2/examples/pylab_examples/boxplot_demo2.html
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.options import get_args
from matplotlib.font_manager import _rebuild
_rebuild() 

plt.style.use("grayscale")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams["font.size"] = 14

colors = plt.get_cmap('Greys')(np.linspace(0.15, 0.85, 2))
colors = [colors[-1], colors[0]]
mode = 'cluster'

titles = {'reddit_sage': 'SAGE Reddit', 
          'cluster_gcn': 'ClusterGCN Ogbn-Products'}

fig, axes = plt.subplots(
    nrows=1, ncols=2, figsize=(7, 7/2), tight_layout=True)


for i, model in enumerate(['reddit_sage', 'cluster_gcn']):
# for i, model in enumerate(['reddit_sage']):
    ax = axes[i]
    ax.set_title(titles[model])
    ax.set_ylabel('峰值内存 (GB)', fontsize=14)
    ax.set_xlabel('批规模', fontsize=14)

    box_data = []

    if model == 'reddit_sage':
        batch_sizes = [8700, 8800, 8900]
    else:
        batch_sizes = [9000, 9100, 9200]
    
    for bs in batch_sizes:
        # read file
        for var in ['_', '_opt_random_0_']:
            file_name = '_'.join([model, str(bs)])
            if i == 0 and var == '_opt_random_0_':
                k = 'final_v3'
            else: k = 'v0'
            real_path = os.path.join(PROJECT_PATH, 'sec6_cutting/exp_diff_res', file_name) + var + f'{k}.csv'
            print(real_path)
            if os.path.exists(real_path):
                res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')['memory']
                print(file_name, res)
            else:
                res = []
            box_data.append(list(map(lambda x: x/(1024*1024*1024), res)))
    bp = ax.boxplot(box_data)

    numBoxes = len(batch_sizes) * 2
    for i in range(numBoxes):
        if i % 2 == 1:
            plt.setp(bp['medians'][i], color='red')
            plt.setp(bp['boxes'][i], color='red')
            # plt.setp(bp['caps'][i], color='red')
            plt.setp(bp['fliers'][i], markeredgecolor='red')
            # https://matplotlib.org/stable/gallery/statistics/boxplot.html#sphx-glr-gallery-statistics-boxplot-py
    medians = list(range(numBoxes))
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(len(batch_sizes)):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        # Alternate between Dark Khaki and Royal Blue
        k = i % 2
        boxPolygon = Polygon(boxCoords, facecolor=colors[k])
        ax.add_patch(boxPolygon)

    ax.set_xticks([1.5, 3.5, 5.5])
    ax.set_xticklabels(batch_sizes, fontsize=14)

    xlim = ax.get_xlim()
    line = 3 if model == 'reddit_sage' else 2
    line, = ax.plot(xlim, [line] * len(xlim), linestyle='dashed', color='b', linewidth=1.5, label='GPU内存上限')

    legend_colors = [Patch(facecolor=colors[0], edgecolor='black'), Patch(facecolor=colors[1], edgecolor='red')]
    ax.legend(legend_colors + [line], ['优化前', '优化后', 'GPU内存限制'], fontsize=10)

fig.savefig(os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_figs', f'exp_memory_inference_motivation_optimize.png'))

