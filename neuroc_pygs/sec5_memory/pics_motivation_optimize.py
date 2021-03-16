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
plt.rcParams["font.size"] = 12

colors = plt.get_cmap('Greys')(np.linspace(0.15, 0.85, 2))
colors = [colors[-1], colors[0]]

predict_model = 'linear_model'
batch_sizes = [175, 180, 185]
labels = []
for i in batch_sizes:
    labels.extend(['优化前\n' + str(i), '优化后\n' + str(i)])
                  
for model in ['gat', 'gcn']:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 7/2), tight_layout=True)
    for i, data in enumerate(['reddit', 'yelp']):
        ax = axes[i]
    
        ax.set_title(data.capitalize(), fontsize=14)
        ax.set_ylabel('峰值内存 (GB)', fontsize=14)
        ax.set_xlabel('批规模', fontsize=14)

        box_data = []
        
        for bs in batch_sizes:
            # read file
            for var in ['cluster', predict_model]:
                file_name = '_'.join([data, model, str(bs), var, 'v2'])
                real_path = os.path.join(PROJECT_PATH, 'sec5_memory/exp_motivation', file_name) + '.csv'
                print(real_path)
                if os.path.exists(real_path):
                    res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')['memory']
                    print(file_name, res)
                else:
                    res = []
                box_data.append(list(map(lambda x: x/(1024*1024*1024), res)))
        bp = ax.boxplot(box_data)
        
        # color
        numBoxes = len(labels)
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
        
        ax.set_xticks([1.5, 3.5, 5.5, 7.5])
        ax.set_xticklabels(batch_sizes, fontsize=14)
        legend_colors = [Patch(facecolor=c, edgecolor='black') for c in colors]
        ax.legend(legend_colors, ['优化前', '优化后'])

    fig.savefig(os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_figs', f'{model}_cluster_motivation_{predict_model}.png'))



