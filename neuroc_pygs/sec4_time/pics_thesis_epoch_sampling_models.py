import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.font_manager import _rebuild
# print(_rebuild())
_rebuild()

base_size = 12
plt.style.use("grayscale")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams["font.size"] = base_size

mode = 'graphsage'
df_data = pd.read_csv(f'out_csv/sampling_{mode}_models.csv', index_col=0)

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

print(df)
colors = plt.get_cmap('Greys')(np.linspace(0.15, 0.85, 2))
colors = [colors[-1], colors[0]]
i = 0
for item in df.keys():
    tab_data = df[item]

    xs = ['GCN', 'GGNN', 'GAT', 'GaAN']

    x = np.arange(len(xs))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
    ax.set_title(item, fontsize=base_size+2)
    ax.set_ylabel('30轮训练时间 (s)', fontsize=base_size+2)
    ax.set_xlabel('算法', fontsize=base_size+2)
    ax.set_xticks(x)
    ax.set_xticklabels(xs, fontsize=base_size+2)
    ax.bar(x - width/2, tab_data['Baseline'], width,
           color=colors[0], edgecolor='black', label='优化前')
    ax.bar(x + width/2, tab_data['Optimize'], width,
           color=colors[1], edgecolor='black', label='优化后')
    # if i == 0:
    #     ax.legend(loc='upper left')
    # else:
    #     ax.legend(loc='upper right')
    ax.legend(loc='lower center', ncol=2)
    fig.savefig(
        f'exp4_thesis_figs/epoch_sampling_figs/exp_epoch_sampling_models_{mode}_{item}.png', dpi=400)
    i += 2


base_size = 10
i = 0
for item in df.keys():
    tab_data = df[item]

    xs = ['GCN', 'GGNN', 'GAT', 'GaAN']

    x = np.arange(len(xs))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
    ax.set_title(item, fontsize=base_size+2)
    ax.set_ylabel('比值', fontsize=base_size+2)
    ax.set_xlabel('算法', fontsize=base_size+2)
    line1, = ax.plot(x, tab_data['exp_ratio'],
                     'ob', label='理论加速比', linestyle='-')
    line2, = ax.plot(x, tab_data['real_ratio'],
                     'Dg', label='实际加速比', linestyle='-')
    line3, = ax.plot(x, tab_data['r1'], 'r^', label='优化效果', linestyle='-')

    ax2 = ax.twinx()
    ax2.set_ylabel('耗时比例 (%)', fontsize=base_size + 2)
    line4, = ax2.plot(x, tab_data['x'], 's--',
                      color='black', label="评估耗时占比" + r"$X$")
    plt.legend(handles=[line1, line2, line3, line4], fontsize='x-small')
    plt.xticks(ticks=x, labels=xs, fontsize=base_size)
    plt.yticks(fontsize=base_size)
    fig.tight_layout()  # 防止重叠

    fig.savefig(
        f'exp4_thesis_figs/epoch_sampling_figs/exp_epoch_sampling_models_{mode}_{item}_else.png', dpi=400)
    i += 2
