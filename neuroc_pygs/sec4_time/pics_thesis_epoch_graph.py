import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
index = {
    'gaan': [5, 10, 20, 50, 100, 150, 200],
    'gcn': [5, 10, 20, 40, 80, 85, 90, 95, 100, 200, 300, 400, 450, 500, 525,
            550, 575, 600]
}
names = ['gaan', 'gcn']
for i, file in enumerate(['amazon-computers_graph_gaan_100k_1024', 'amazon-computers_graph_gcn_2048_100k']): # gcn, gaan
    df_data = pd.read_csv('out_csv/' + file + '.csv', index_col=0)
    df[names[i]] = {'Baseline': [], 'Optimize': [], 'x': [], 'real_ratio': [], 'exp_ratio': []}
    for var in df_data.index:
        df[names[i]]['Baseline'].append(float(df_data['baseline'][var]))
        df[names[i]]['Optimize'].append(float(df_data['opt'][var]))
        df[names[i]]['x'].append(float(100 * df_data['x'][var]))
        df[names[i]]['real_ratio'].append(float(df_data['real_ratio'][var]))
        df[names[i]]['exp_ratio'].append(float(df_data['exp_ratio'][var]))

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
    ax.set_title(titles[item] + ' amazon-computers', fontsize=base_size+2)
    ax.set_ylabel('加速比', fontsize=base_size+2)
    ax.set_xlabel('边数', fontsize=base_size+2)
    line1, = ax.plot(x, tab_data['exp_ratio'], 'ob', label='预期加速比', linestyle='-')
    line2, = ax.plot(x, tab_data['real_ratio'], 'Dg', label='实际加速比', linestyle='-')
    
    ax2 = ax.twinx()
    ax2.set_ylabel("评估耗时占比" + r"$X$" + " (%)", fontsize=base_size + 2)
    line3, = ax2.plot(x, tab_data['x'], 'rs--', label='评估耗时占比' + r"$X$" )
    plt.legend(handles=[line1, line2, line3], fontsize='small')
    plt.xticks(ticks=xs[item], labels=[f'{j}k' for j in xs[item]], fontsize=base_size)
    plt.yticks(fontsize=base_size)
    fig.tight_layout() # 防止重叠

    fig.savefig(f'out_figs/exp_epoch_full_graph_{item}_else.png')
    i += 2