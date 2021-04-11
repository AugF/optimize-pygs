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


titles = {'gcn': 'GCN', 'ggnn': 'GGNN', 'gat': 'GAT', 'gaan': 'GaAN'}

df = {}
df_data = pd.read_csv('out_csv/model_hidden_size.csv', index_col=0)
for file in ['gcn', 'gaan']:
    df[file] = {'Baseline': [], 'Optimize': [], 'x': [], 'real_ratio': [], 'exp_ratio': [], 'r1': []}
    for var in [64, 128, 256, 512, 1024, 2048]:
        df[file]['Baseline'].append(float(df_data['baseline'][f'{file}_{var}']))
        df[file]['Optimize'].append(float(df_data['opt'][f'{file}_{var}']))
        df[file]['x'].append(100-float(100 * df_data['x'][f'{file}_{var}']))
        df[file]['real_ratio'].append(float(df_data['real_ratio'][f'{file}_{var}']))
        df[file]['exp_ratio'].append(float(df_data['exp_ratio'][f'{file}_{var}']))
        df[file]['r1'].append(float(df_data['r1'][f'{file}_{var}']))

print(df)
base_size = 14
i = 0
for item in df.keys():
    tab_data = df[item]

    x = [64, 128, 256, 512, 1024, 2048]
    fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
    ax.set_title(titles[item] + ' amazon-computers', fontsize=base_size+2)
    ax.set_ylabel('加速比', fontsize=base_size+2)
    ax.set_xlabel('隐藏向量维度', fontsize=base_size+2)
    line1, = ax.plot(x, tab_data['exp_ratio'], 'ob', label='预期加速比', linestyle='-')
    line2, = ax.plot(x, tab_data['real_ratio'], 'Dg', label='实际加速比', linestyle='-')
    line3, = ax.plot(x, tab_data['r1'], 'r^', label='优化效果', linestyle='-')
    
    ax2 = ax.twinx()
    ax2.set_ylabel("评估耗时占比" + r"$X$" + " (%)", fontsize=base_size + 2)
    line4, = ax2.plot(x, tab_data['x'], 's--', color='black', label='耗时比例')
    plt.legend(handles=[line1, line2, line3, line4], fontsize='small', loc='center right')
    # plt.xticks(ticks=xs[item], labels=[f'{j}k' for j in xs[item]], fontsize=base_size)
    plt.xticks(fontsize=base_size)
    plt.yticks(fontsize=base_size)
    fig.tight_layout() # 防止重叠

    fig.savefig(f'out_figs/exp_epoch_full_sampling_batch_size_{item}_else.png')
    i += 2

