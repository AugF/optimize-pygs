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

index = {
    'gcn_pubmed': [10, 20, 50, 80, 100, 200],
    'gaan_amazon-computers': [10, 20, 50, 80, 100, 200]
}
titles = {
    'gcn_pubmed': 'GCN pubmed',
    'gaan_amazon-computers': 'GaAN amazon-computers'
}
linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (5, 5)), (0, (3, 10, 1, 10))]


df = {}
for file in ['gcn_pubmed', 'gaan_amazon-computers']:
    df_data = pd.read_csv('out_csv/sampling_' + file + '_N.csv', index_col=0)
    df[file] = {'Baseline': [], 'Optimize': [], 'x': [],
                'real_ratio': [], 'exp_ratio': [], 'r1': []}
    print(df_data.index)
    for data in df_data.index:
        df[file]['Baseline'].append(float(df_data['baseline'][data]))
        df[file]['Optimize'].append(float(df_data['opt'][data]))
        df[file]['x'].append(100 - float(100 * df_data['x'][data]))
        df[file]['real_ratio'].append(1/float(df_data['real_ratio'][data]))
        df[file]['exp_ratio'].append(1/float(df_data['exp_ratio'][data]))
        df[file]['r1'].append(1/float(df_data['r1'][data]))

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
    line3, = ax.plot(x, tab_data['r1'], 'r^', label='优化效果', linestyle=linestyles[0])

    ax2 = ax.twinx()
    ax2.set_ylabel("训练时间 (s)", fontsize=base_size + 2)
    
    line4, = ax2.plot(x, tab_data['Baseline'], 'ob', linestyle=linestyles[1],
                        label='优化前耗时')
    line5, = ax2.plot(x, tab_data['Optimize'], 'Dg', linestyle=linestyles[2],
                        label='优化后耗时')
    plt.legend(handles=[line3, line4, line5])
    plt.xticks(fontsize=base_size+2)
    plt.yticks(fontsize=base_size+2)
    fig.tight_layout()  # 防止重叠

    fig.savefig(
        f'exp4_thesis_ppt_figs/epoch_sampling_figs/exp_epoch_sampling_N_{item}_else.png', dpi=400)
    i += 2
