import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.font_manager import _rebuild
# print(_rebuild())
_rebuild() 

base_size = 12
plt.style.use("grayscale")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams["font.size"] = base_size

df = {}
for file in ['pubmed', 'amazon-computers', 'flickr', 'com-amazon']:
    df_data = pd.read_csv('out_csv/' + file + '.csv', index_col=0)
    df[file] = {'Baseline': [], 'Optimize': [], 'x': [], 'real_ratio': [], 'exp_ratio': []}
    for data in ['gcn', 'ggnn', 'gat', 'gaan']:
        df[file]['Baseline'].append(float(df_data['baseline'][data]))
        df[file]['Optimize'].append(float(df_data['opt'][data]))
        df[file]['x'].append(float(100 * df_data['x'][data]))
        df[file]['real_ratio'].append(float(df_data['real_ratio'][data]))
        df[file]['exp_ratio'].append(float(df_data['exp_ratio'][data]))

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
    ax.set_ylabel('50轮训练时间 (秒)', fontsize=base_size+2)
    ax.set_xlabel('算法', fontsize=base_size+2)
    ax.set_xticks(x)
    ax.set_xticklabels(xs, fontsize=base_size+2)
    ax.bar(x - width/2, tab_data['Baseline'], width, color=colors[0], edgecolor='black', label='优化前')
    ax.bar(x + width/2, tab_data['Optimize'], width, color=colors[1], edgecolor='black', label='优化后')
    # if i == 0:
    #     ax.legend(loc='upper left')
    # else:
    #     ax.legend(loc='upper right')
    ax.legend()
    fig.savefig(f'out_figs/exp_epoch_full_models_{item}.png')
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
    ax.set_ylabel('加速比', fontsize=base_size+2)
    ax.set_xlabel('算法', fontsize=base_size+2)
    line1, = ax.plot(x, tab_data['exp_ratio'], 'ob', label='预期加速比', linestyle='-')
    line2, = ax.plot(x, tab_data['real_ratio'], 'Dg', label='实际加速比', linestyle='-')
    
    ax2 = ax.twinx()
    ax2.set_ylabel("评估耗时占比" + r"$X$" + " (%)", fontsize=base_size + 2)
    line3, = ax2.plot(x, tab_data['x'], 'rs--', label='评估耗时占比' + r"$X$" )
    plt.legend(handles=[line1, line2, line3], fontsize='x-small')
    plt.xticks(ticks=x, labels=xs, fontsize=base_size)
    plt.yticks(fontsize=base_size)
    fig.tight_layout() # 防止重叠

    fig.savefig(f'out_figs/exp_epoch_full_models_{item}_else.png')
    i += 2

