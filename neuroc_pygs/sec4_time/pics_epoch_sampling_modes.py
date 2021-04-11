import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from neuroc_pygs.sec4_time.utils import datasets_maps, algorithms
from matplotlib.font_manager import _rebuild
_rebuild() 

base_size=12
plt.style.use("grayscale")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams["font.size"] = base_size

sampling_maps = {
    'cluster': '聚类采样',
    'graphsage': '邻居采样',
    'gcn_amazon-computers': 'GCN amazon-computers',
    'gat_flickr': 'GAT flickr'
}

headers = ['Name', 'Baseline', 'Batch Opt', 'Epoch Opt', 'Opt', 'Batch Ratio%', 'Epoch Raio%', 'Opt%']
root_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time'
real_path = root_path + f'/exp_res/sampling_epoch_modes.txt'
df = []
with open(real_path) as f:
    for line in f.readlines():
        df.append(line.strip()[1:-1].split(','))
df = pd.DataFrame(df, columns=headers)
df.index = [x[1:-1] for x in df['Name']]
del df['Name']
print(df)

# models
xs = ['graphsage', 'cluster']
for data in ['gcn_amazon-computers', 'gat_flickr']:
    tab_data = defaultdict(list)
    for mode in xs:
        index = f'{data}_{mode}_None'
        tmp_data = df.loc[index]
        tab_data['Baseline'].append(float(tmp_data['Baseline']))
        tab_data['Batch Opt'].append(float(tmp_data['Batch Opt']))
        tab_data['Epoch Opt'].append(float(tmp_data['Epoch Opt']))
        tab_data['Opt'].append(float(tmp_data['Opt']))

    x = np.arange(len(xs))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
    ax.set_title(sampling_maps[data], fontsize=base_size+2)
    ax.set_ylabel('30轮训练时间 (秒)', fontsize=base_size+2)
    ax.set_xlabel('采样方法', fontsize=base_size+2)
    ax.set_xticks(x)
    ax.set_xticklabels([sampling_maps[i] for i in xs], fontsize=base_size)

    ax.bar(x - 1.5 * width, tab_data['Baseline'], width, label='未优化')
    ax.bar(x - 0.5 * width, tab_data['Epoch Opt'], width, label='优化1')
    ax.bar(x + 0.5 * width, tab_data['Batch Opt'], width, label='优化2')
    ax.bar(x + 1.5 * width, tab_data['Opt'], width, label='优化1+优化2')
    if data == 'gcn_amazon-computers':
        ax.legend(fontsize='small', loc='lower center', ncol=2)
    else:
        ax.legend(fontsize='small')
    fig.savefig(root_path + f'/exp_figs_total/exp_epoch_sampling_modes_{data}.png')



