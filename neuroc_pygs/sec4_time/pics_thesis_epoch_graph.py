import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

titles = {'gcn': 'GCN', 'ggnn': 'GGNN', 'gat': 'GAT', 'gaan': 'GaAN'}
df = {}
index = {
    'gaan': [5, 10, 20, 50, 100, 150, 200],
    'gcn': [5, 10, 20, 40, 80, 85, 90, 95, 100, 200, 300, 400, 450, 500, 525,
            550, 575, 600]
}
names = ['gaan', 'gcn']
for i, file in enumerate(['amazon-computers_graph_gaan_100k_1024', 'amazon-computers_graph_gcn_2048_100k']):  # gcn, gaan
    df_data = pd.read_csv('out_csv/' + file + '.csv', index_col=0)
    df[names[i]] = {'Baseline': [], 'Optimize': [], 'x': [],
                    'real_ratio': [], 'exp_ratio': [], 'r1': []}
    for var in df_data.index:
        df[names[i]]['Baseline'].append(float(df_data['baseline'][var]))
        df[names[i]]['Optimize'].append(float(df_data['opt'][var]))
        df[names[i]]['x'].append(float(100 * df_data['x'][var]))
        df[names[i]]['real_ratio'].append(1/float(df_data['real_ratio'][var]))
        df[names[i]]['exp_ratio'].append(1/float(df_data['exp_ratio'][var]))
        df[names[i]]['r1'].append(1/float(df_data['r1'][var]))

print(df)

xs = {
    'gaan': [0, 50, 100, 150, 200],
    'gcn': [0, 200, 400, 600]
}
base_size = 14
i = 0
for item in df.keys():
    tab_data = df[item]

    x = index[item]
    fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
    ax.set_title('GCN', fontsize=base_size+2)
    ax.set_ylabel('比值', fontsize=base_size+2)
    ax.set_xlabel('边数', fontsize=base_size+2)
    line1, = ax.plot(x, tab_data['exp_ratio'],
                     'ob', label='理论加速比', linestyle='-')
    line2, = ax.plot(x, tab_data['real_ratio'],
                     'Dg', label='实际加速比', linestyle='-')
    line3, = ax.plot(x, tab_data['r1'], 'r^', label='优化效果', linestyle='-')

    ax2 = ax.twinx()
    ax2.set_ylabel('耗时比例 (%)', fontsize=base_size + 2)
    line4, = ax2.plot(x, tab_data['x'], 's--',
                      color='black', label="评估耗时占比" + r"$X$")
    plt.legend(handles=[line1, line2, line3, line4], fontsize='small')
    plt.xticks(ticks=xs[item], labels=[
               f'{j}k' for j in xs[item]], fontsize=base_size)
    plt.yticks(fontsize=base_size)
    fig.tight_layout()  # 防止重叠

    fig.savefig(
        f'exp4_thesis_figs/epoch_full_figs/exp_epoch_full_graph_{item}_else.png', dpi=400)
    i += 2
