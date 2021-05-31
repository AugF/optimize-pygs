import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.font_manager import _rebuild
_rebuild()
config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)

base_size = 12
plt.style.use("grayscale")

plt.rcParams["font.size"] = base_size


file = 'gat_flickr'
titles = {
    'cluster': '聚类采样',
    'graphsage': '邻居采样'
}
linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (5, 5)), (0, (3, 10, 1, 10))]

df = {}
for mode in ['cluster', 'graphsage']:
    df_data = pd.read_csv('out_csv/' + mode + '_' +
                          file + '_batch_size.csv', index_col=0)
    df[mode] = {'Baseline': [], 'Optimize': [], 'x': [],
                'real_ratio': [], 'exp_ratio': [], 'r1': []}
    for var in df_data.index:
        df[mode]['Baseline'].append(float(df_data['baseline'][var]))
        df[mode]['Optimize'].append(float(df_data['opt'][var]))
        df[mode]['x'].append(100-float(100 * df_data['x'][var]))
        df[mode]['real_ratio'].append(1/float(df_data['real_ratio'][var]))
        df[mode]['exp_ratio'].append(1/float(df_data['exp_ratio'][var]))
        df[mode]['r1'].append(1/float(df_data['r1'][var]))
        
base_size = 14
i = 0
for item in df.keys():
    tab_data = df[item]

    xs = [0.01, 0.03, 0.06, 0.1, 0.25, 0.5]

    x = np.arange(len(xs))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
    ax.set_title(titles[item], fontsize=base_size+2)
    ax.set_ylabel('比值', fontsize=base_size+2)
    ax.set_xlabel('相对批规模', fontsize=base_size+2)
    line3, = ax.plot(x, tab_data['r1'], 'r^', label='优化效果', linestyle=linestyles[0])

    ax2 = ax.twinx()
    ax2.set_ylabel("30轮训练时间 (s)", fontsize=base_size + 2)
    
    line4, = ax2.plot(x, tab_data['Baseline'], 'ob', linestyle=linestyles[1],
                        label='优化前耗时')
    line5, = ax2.plot(x, tab_data['Optimize'], 'Dg', linestyle=linestyles[2],
                        label='优化后耗时')
    plt.legend(handles=[line3, line4, line5])
    plt.xticks(ticks=np.arange(len(xs)), labels=[
               f'{int(100*x)}%' for x in xs], fontsize=base_size+2)
    plt.yticks(fontsize=base_size+2)
    fig.tight_layout()  # 防止重叠

    fig.savefig(
        f'exp4_thesis_ppt_figs/epoch_sampling_figs/exp_epoch_sampling_batch_size_{file}_{item}_else.png', dpi=400)
    i += 2
