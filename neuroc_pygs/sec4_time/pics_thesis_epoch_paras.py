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

index = {
    'gcn': [64, 128, 256, 512, 1024, 2048, 2304, 2560, 2816, 2944, 3072, 3200,
            3328],
    'gaan': [8, 16, 32, 64, 128, 256, 512, 768, 896, 1024, 1088]
}
titles = {'gcn': 'GCN', 'ggnn': 'GGNN', 'gat': 'GAT', 'gaan': 'GaAN'}

df = {}
for file in ['gcn', 'gaan']:
    df_data = pd.read_csv('out_csv/amazon-computers_' + file + '_batch_size.csv', index_col=0)
    df[file] = {'Baseline': [], 'Optimize': [], 'x': [], 'real_ratio': [], 'exp_ratio': []}
    print(df_data.index)
    for data in df_data.index:
        df[file]['Baseline'].append(float(df_data['baseline'][data]))
        df[file]['Optimize'].append(float(df_data['opt'][data]))
        df[file]['x'].append(float(100 * df_data['x'][data]))
        df[file]['real_ratio'].append(float(df_data['real_ratio'][data]))
        df[file]['exp_ratio'].append(float(df_data['exp_ratio'][data]))


base_size = 14
i = 0
for item in df.keys():
    tab_data = df[item]

    x = index[item]
    fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
    ax.set_title(titles[item] + ' amazon-computers', fontsize=base_size+2)
    ax.set_ylabel('加速比', fontsize=base_size+2)
    ax.set_xlabel('隐藏向量维度', fontsize=base_size+2)
    line1, = ax.plot(x, tab_data['exp_ratio'], 'ob', label='预期加速比', linestyle='-')
    line2, = ax.plot(x, tab_data['real_ratio'], 'Dg', label='实际加速比', linestyle='-')
    
    ax2 = ax.twinx()
    ax2.set_ylabel("评估耗时占比" + r"$X$" + " (%)", fontsize=base_size + 2)
    line3, = ax2.plot(x, tab_data['x'], 'rs--', label='评估耗时占比' + r"$X$" )
    plt.legend(handles=[line1, line2, line3], fontsize='small', loc='center right')
    # plt.xticks(ticks=xs[item], labels=[f'{j}k' for j in xs[item]], fontsize=base_size)
    plt.xticks(fontsize=base_size)
    plt.yticks(fontsize=base_size)
    fig.tight_layout() # 防止重叠

    fig.savefig(f'out_figs/exp_epoch_full_batch_size_{item}_else.png')
    i += 2

