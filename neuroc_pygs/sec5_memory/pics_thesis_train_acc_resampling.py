import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from matplotlib.font_manager import _rebuild
from neuroc_pygs.configs import PROJECT_PATH

_rebuild()
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)
plt.rcParams["font.size"] = 18

dir_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'out_train_data')

labels = ['GCN Pubmed', 'GCN Coauthor-Physics', 'GAT Pubmed', 'GAT Coauthor-Physics']
markers = 'oD^sdp'
linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (5, 5)), (0, (3, 10, 1, 10))]

fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
df_data = defaultdict(list)
for model in ['gcn', 'gat']:
    df = pd.read_csv(dir_path + f'/prove_train_acc_{model}.csv', index_col=0)
    df.index = [f"{df['Model'][i]}_{df['Data'][i]}_{df['Per'][i]}" for i in range(len(df.index))]
    for data in ['pubmed', 'coauthor-physics']:
        cur_name = model + '_' + data
        for rs in [0.0, 0.01, 0.03, 0.06, 0.1, 0.2, 0.5]:
            index = cur_name + '_' + str(rs)
            # df_data[cur_name].append(np.mean([df['Acc1'][index], df['Acc2'][index], df['Acc3'][index]]))
            df_data[cur_name].append(np.mean([df['Acc'][index]]))

df_data = pd.DataFrame(df_data)
ax.set_ylabel('精度', fontsize=20)
ax.set_xlabel('相对重采样比例', fontsize=20)
for j, c in enumerate(df_data.columns):
    ax.plot(df_data.index, df_data[c], label=labels[j], marker=markers[j], markersize=10, linestyle=linestyles[j])
ax.set_xticks(np.arange(7))
ax.set_xticklabels(['Zero', '1%', '3%', '6%', '10%', '20%', '50%'])
ax.legend()
fig.savefig(f'exp5_thesis_figs/exp_memory_training_resampling_acc.png', dpi=400)

