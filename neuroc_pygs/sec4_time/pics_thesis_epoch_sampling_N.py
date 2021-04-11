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
    'gcn_pubmed': [10, 20, 50, 80, 100, 200],
    'gaan_amazon-computers': [10, 20, 50, 80, 100, 200]
}
titles = {
    'gcn_pubmed': 'GCN pubmed',
    'gaan_amazon-computers': 'GaAN amazon-computers'
}

df = {}
for file in ['gcn_pubmed', 'gaan_amazon-computers']:
    df_data = pd.read_csv('out_csv/sampling_' + file + '_N.csv', index_col=0)
    df[file] = {'Baseline': [], 'Optimize': [], 'x': [], 'real_ratio': [], 'exp_ratio': [], 'r1': []}
    print(df_data.index)
    for data in df_data.index:
        df[file]['Baseline'].append(float(df_data['baseline'][data]))
        df[file]['Optimize'].append(float(df_data['opt'][data]))
        df[file]['x'].append(100 - float(100 * df_data['x'][data]))
        df[file]['real_ratio'].append(float(df_data['real_ratio'][data]))
        df[file]['exp_ratio'].append(float(df_data['exp_ratio'][data]))
        df[file]['r1'].append(float(df_data['r1'][data]))

print(df)
base_size = 14
i = 0
for item in df.keys():
    tab_data = df[item]

    x = index[item]
    fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
    ax.set_title(titles[item], fontsize=base_size+2)
    ax.set_ylabel('比值', fontsize=base_size+2)
    ax.set_xlabel('训练轮数', fontsize=base_size+2)
    line1, = ax.plot(x, tab_data['exp_ratio'], 'ob', label='预期加速比', linestyle='-')
    line2, = ax.plot(x, tab_data['real_ratio'], 'Dg', label='实际加速比', linestyle='-')
    line3, = ax.plot(x, tab_data['r1'], 'r^', label='优化效果', linestyle='-')
    
    ax2 = ax.twinx()
    ax2.set_ylabel("评估耗时占比" + r"$X$" + " (%)", fontsize=base_size + 2)
    line4, = ax2.plot(x, tab_data['x'], 's--', color='black', label='耗时比例')
    plt.legend(handles=[line1, line2, line3, line4], fontsize=base_size-2, loc='center right')
    plt.xticks(fontsize=base_size)
    plt.yticks(fontsize=base_size)
    fig.tight_layout() # 防止重叠

    fig.savefig(f'out_figs/exp_epoch_full_sampling_N_{item}_else.png')
    i += 2

