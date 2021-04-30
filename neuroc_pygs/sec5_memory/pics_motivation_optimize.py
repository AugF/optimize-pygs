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

root_path = os.path.join(PROJECT_PATH, 'sec5_memory/exp_automl_datasets_diff')
ratio_dict = pd.read_csv(root_path + '/regression_mape_res.csv', index_col=0)
linear_ratio_dict = pd.read_csv(root_path + '/regression_linear_mape_res.csv', index_col=0)
dir_path = os.path.join(PROJECT_PATH, 'sec5_memory/exp_motivation_diff')
dir_out = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp5_thesis_figs')

# colors = [colors[-1], colors[0]]
colors = ['black', 'white']
      
def run(predict_model='linear_model', bias=0.001):
    for model in ['gat', 'gcn']:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 7/2), tight_layout=True)
        for i, data in enumerate(['reddit', 'yelp']):
            if predict_model == 'linear_model':
                memory_ratio = linear_ratio_dict[model][data] + bias
            else:
                memory_ratio = ratio_dict[model][predict_model] + bias
            
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
                # read file
                for var in ['cluster', predict_model]:
                    if var == 'cluster':
                        file_name = '_'.join([data, model, str(bs), var, 'v2'])
                    else:
                        file_name = '_'.join([data, model, str(bs), var, str(int(100*memory_ratio)), 'mape_diff_v3'])
                    real_path =  dir_path + '/' + file_name + '.csv'
                    print(file_name)
                    if os.path.exists(real_path):
                        res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')['memory']
                        print(file_name, res)
                    else:
                        res = []
                    box_data.append(list(map(lambda x: x/(1024*1024*1024), res)))
            bp = ax.boxplot(box_data, patch_artist=True)

            # for key in ['medians', 'boxes', 'caps', 'fliers']:
            #     print(bp[key])
            #     print(len(bp[key]))
            # color
            numBoxes = len(batch_sizes) * 2
            for i in range(numBoxes):
                if i % 2 == 1:
                    plt.setp(bp['medians'][i], color='red')
                    plt.setp(bp['boxes'][i], color='red')
                    plt.setp(bp['boxes'][i], facecolor=colors[1])
                    plt.setp(bp['fliers'][i], markeredgecolor='red')
                    # https://matplotlib.org/stable/gallery/statistics/boxplot.html#sphx-glr-gallery-statistics-boxplot-py
                else:
                    plt.setp(bp['boxes'][i], facecolor=colors[0])

            # medians = list(range(numBoxes))
            # for i in range(numBoxes):
            #     box = bp['boxes'][i]
            #     boxX = []
            #     boxY = []
            #     for j in range(len(batch_sizes)):
            #         boxX.append(box.get_xdata()[j])
            #         boxY.append(box.get_ydata()[j])
            #     boxCoords = list(zip(boxX, boxY))
            #     # Alternate between Dark Khaki and Royal Blue
            #     k = i % 2
            #     boxPolygon = Polygon(boxCoords, facecolor=colors[k])
            #     ax.add_patch(boxPolygon)
            
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

        fig.savefig(dir_out + f'/exp_memory_training_{model}_cluster_motivation_{predict_model}_mape_diff_v3.png', dpi=400)


if __name__ == '__main__':
    for predict_model in ['automl', 'linear_model']:
        run(predict_model)

