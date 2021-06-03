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

mode = 'graphsage'
df_data = pd.read_csv(f'out_epoch_csv/sampling_{mode}_models.csv', index_col=0)

data = 'amazon-computers'
# model
df = {}
df[data] = {'Baseline': [], 'Optimize': [], 'x': [],
            'real_ratio': [], 'exp_ratio': [], 'r1': []}
for alg in ['gcn', 'ggnn', 'gat', 'gaan']:
    df[data]['Baseline'].append(float(df_data['baseline'][alg + '_' + data]))
    df[data]['Optimize'].append(float(df_data['opt'][alg + '_' + data]))
    df[data]['x'].append(100-float(100 * df_data['x'][alg + '_' + data]))
    df[data]['real_ratio'].append(
        1/float(df_data['real_ratio'][alg + '_' + data]))
    df[data]['exp_ratio'].append(
        1/float(df_data['exp_ratio'][alg + '_' + data]))
    df[data]['r1'].append(1/float(df_data['r1'][alg + '_' + data]))

linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (5, 5)), (0, (3, 10, 1, 10))]


base_size = 14
i = 0
for item in df.keys():
    tab_data = df[item]

    xs = ['GCN', 'GGNN', 'GAT', 'GaAN']

    x = np.arange(len(xs))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
    ax.set_title(item, fontsize=base_size+2)
    ax.set_ylabel('比值', fontsize=base_size+2)
    ax.set_xlabel('算法', fontsize=base_size+2)
    line3, = ax.plot(x, tab_data['r1'], 'r^', label='优化效果', linestyle=linestyles[0])

    ax2 = ax.twinx()
    ax2.set_ylabel("30轮训练时间 (s)", fontsize=base_size + 2)
    
    line4, = ax2.plot(x, tab_data['Baseline'], 'ob', linestyle=linestyles[1],
                        label='优化前耗时')
    line5, = ax2.plot(x, tab_data['Optimize'], 'Dg', linestyle=linestyles[2],
                        label='优化后耗时')
    plt.legend(handles=[line3, line4, line5])
    plt.xticks(ticks=x, labels=xs, fontsize=base_size+2)
    plt.yticks(fontsize=base_size+2)
    fig.tight_layout()  # 防止重叠

    fig.savefig(
        f'exp4_thesis_ppt_figs/epoch_sampling_figs/exp_epoch_sampling_models_{mode}_{item}_else.png', dpi=400)
    i += 2
