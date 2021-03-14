import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import font_manager
import matplotlib.pyplot as plt
from neuroc_pygs.sec4_time.utils import datasets_maps, algorithms, sampling_modes
from matplotlib.font_manager import _rebuild
# print(_rebuild())
_rebuild() 

def float_x(x):
    return [float(i) for i in x]

plt.style.use("grayscale")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams["font.size"] = 14

root_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time'
real_path = root_path + '/exp_res/batch/datasets.csv'
df = pd.read_csv(real_path, index_col=0)
print(df)

file_names = ['gcn', 'gat']
vs = ['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
modes = ['graphsage', 'cluster']
MODES = ['GraphSAGE Sampler', 'Cluster Sampler']

for k, mode in enumerate(modes):
    for file in file_names:
        base_times, opt_times, base_error, opt_error = [], [], [], []
        if mode == 'clsuter':
            xs = vs
        else:
            xs = vs[:-1]
        for v in xs:
            index = f'{v}_{file}_None_{mode}_pin_memory_False_num_workers_0_non_blocking_False'
            tmp_data = df.loc[index]
            base_times.append(float_x([tmp_data['Base Sampling'], tmp_data['Base Transfer'], tmp_data['Base Training']]))
            opt_times.append(float_x([tmp_data['Opt Sampling'], tmp_data['Opt Transfer'], tmp_data['Opt Training']]))
            base_error.append(float_x([tmp_data['Base max'], tmp_data['Base min']]))
            opt_error.append(float_x([tmp_data['Opt max'], tmp_data['Opt min']]))
        
        base_times, opt_times, base_error, opt_error = np.cumsum(np.array(base_times).T, axis=0) * 1000, np.cumsum(np.array(opt_times).T, axis=0)*1000, np.array(base_error).T * 1000, np.array(opt_error).T * 1000 # 单位ms

        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        xticklabels = [datasets_maps[i] for i in xs]
        x = np.arange(len(xticklabels))

        locations = [-1, 1]
        colors = plt.get_cmap('Greys')(np.linspace(0.15, 0.85, 2))
        # colors = ['blue', 'cyan']
        width = 0.35
        errors_bar = [base_error, opt_error]
        for i, times in enumerate([base_times, opt_times]):
            ax.bar(x + locations[i] * width / 2, times[0], width, color=colors[1-i], edgecolor='black', hatch="///")
            ax.bar(x + locations[i] * width / 2, times[1], width, color=colors[1-i], edgecolor='black', bottom=times[0], hatch='...')
            ax.bar(x + locations[i] * width / 2, times[2], width, yerr=[errors_bar[i][1], errors_bar[i][0]], color=colors[1-i], edgecolor='black', bottom=times[1], hatch='xxx')
            # 待做: error bar
        if mode == 'cluster':
            ax.set_ylim(0, 120)
        ax.set_title(MODES[k])
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)

        legend_colors = [Line2D([0], [0], color=c, lw=4) for c in colors]
        legend_hatchs = [Patch(facecolor='white', edgecolor='black', hatch='xxxx'), Patch(facecolor='white',edgecolor='black', hatch='....'), Patch(facecolor='white', edgecolor='black', hatch='////')]
        ax.legend(legend_hatchs + legend_colors, ['训练', '数据传输', '采样'] + ['优化前', '优化后'], ncol=1, loc='upper center')
        ax.set_ylabel('每轮训练时间 (毫秒)')
        ax.set_xlabel('数据集')
        fig.savefig(root_path + f'/exp_figs/exp_batch_datasets_{file}_{mode}.png')
        plt.close()

