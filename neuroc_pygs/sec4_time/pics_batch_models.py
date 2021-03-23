import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from neuroc_pygs.sec4_time.utils import datasets_maps, algorithms, sampling_modes
from matplotlib.font_manager import _rebuild
# print(_rebuild())
_rebuild() 

def float_x(x):
    return [float(i) for i in x]

base_size = 12
plt.style.use("grayscale")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams["font.size"] = base_size

root_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time'
real_path = root_path + '/exp_res/sampling_training_models.txt'
headers = None
mydata = []
with open(real_path) as f:
    for line in f.readlines():
        if headers == None:
            headers = [l.strip() for l in line.split('|')][1:]
        else:
            mydata.append([l.strip() for l in line.split('|')][1:])

df = pd.DataFrame(mydata, columns=headers)
df.index = df['Name']
del df['Name']
print(df)


file_names = ['amazon-computers', 'flickr']
titles = ['Amazon-computers', 'Flickr']
xs = ['gcn', 'ggnn', 'gat', 'gaan']
modes = ['graphsage', 'cluster']
MODES = ['GraphSAGE Sampler', 'Cluster Sampler']

y_lims = {
    'cluster_amazon-computers': 100,
    'graphsage_amazon-computers': 700,
    'cluster_flickr': 100,
    'graphsage_flickr': 800
}

for i, mode in enumerate(['cluster']):
    for k, file in enumerate(file_names):
        base_times, opt_times, base_error, opt_error = [], [], [], []
        for v in xs:
            index = f'{file}_{v}_None_{mode}_pin_memory_False_num_workers_0_non_blocking_False'
            tmp_data = df.loc[index]
            base_times.append(float_x([tmp_data['Base Sampling'], tmp_data['Base Transfer'], tmp_data['Base Training']]))
            opt_times.append(float_x([tmp_data['Opt Sampling'], tmp_data['Opt Transfer'], tmp_data['Opt Training']]))
            base_error.append(float_x([tmp_data['Base max'], tmp_data['Base min']]))
            opt_error.append(float_x([tmp_data['Opt max'], tmp_data['Opt min']]))
        
        base_times, opt_times, base_error, opt_error = np.cumsum(np.array(base_times).T, axis=0) * 1000, np.cumsum(np.array(opt_times).T, axis=0)*1000, np.array(base_error).T * 1000, np.array(opt_error).T * 1000 # 单位ms

        fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
        xticklabels = [algorithms[i] for i in xs]
        x = np.arange(len(xticklabels))

        locations = [-1, 1]
        colors = plt.get_cmap('Greys')(np.linspace(0.15, 0.75, 2))
        colors = [colors[-1], colors[0]]
        width = 0.35
        errors_bar = [base_error, opt_error]
        for i, times in enumerate([base_times, opt_times]):
            ax.bar(x + locations[i] * width / 2, times[0], width, color=colors[i], edgecolor='black', hatch="///")
            ax.bar(x + locations[i] * width / 2, times[1] - times[0], width, color=colors[i], edgecolor='black', bottom=times[0], hatch='...')
            ax.bar(x + locations[i] * width / 2, times[2] - times[1], width, yerr=[errors_bar[i][1], errors_bar[i][0]], color=colors[i], edgecolor='black', bottom=times[1], hatch='xxx')
            # 待做: error bar
        if mode == 'cluster':
            ax.set_ylim(0, 120)
        ax.set_title(titles[k], fontsize=base_size + 2)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, fontsize=base_size + 2)

        legend_colors = [Patch(facecolor=c, edgecolor='black') for c in colors]
        legend_hatchs = [Patch(facecolor='white', edgecolor='black', hatch='xxxx'), Patch(facecolor='white',edgecolor='black', hatch='....'), Patch(facecolor='white', edgecolor='black', hatch='////')]
        ax.legend(legend_hatchs + legend_colors, ['训练', '数据传输', '采样'] + ['优化前', '优化后'], loc='upper left', ncol=2, fontsize='x-small')
        ax.set_ylabel('每轮训练时间 (毫秒)', fontsize=base_size+2)
        ax.set_xlabel('算法', fontsize=base_size+2)
        ax.set_ylim(0, y_lims[f'{mode}_{file}'])
        fig.savefig(root_path + f'/exp_figs_final/exp_batch_models_{file}_{mode}.png')
        plt.close()

