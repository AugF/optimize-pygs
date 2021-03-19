import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from matplotlib.font_manager import _rebuild
from neuroc_pygs.configs import PROJECT_PATH

_rebuild()

# plt.style.use("grayscale")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams["font.size"] = 14

dir_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_train_res')
dir_out = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_train_figs')


for model in ['gcn', 'gat']:
    for data in ['pubmed', 'coauthor-physics']:
        for run in range(3):
            fig, axes = plt.subplots(2, 3, figsize=(7*3, 5*2), tight_layout=True)
            df = defaultdict(dict)
            for discard_per in [0, 0.01, 0.03, 0.06, 0.1, 0.2, 0.5]:
                dis_per = str(int(100*discard_per))
                real_path = dir_path + f'/{model}_{data}_{dis_per}_{run}.csv'
                pd_data = pd.read_csv(real_path, index_col=0)
                df['train'][dis_per] = pd_data['train_acc']
                df['val'][dis_per] = pd_data['val_acc']
                df['test'][dis_per] = pd_data['test_acc']
                df['best val'][dis_per] = pd_data['best_val_acc']
                df['final test'][dis_per] = pd_data['final_test_acc']
                
            for i, x in enumerate(['train', 'val', 'test', 'best val', 'final test']):
                ax = axes[i//3][i%3]
                ax.set_title(model.upper() + ' ' + data.capitalize())
                ax.set_ylabel(x.capitalize() + ' Acc')
                df_data = pd.DataFrame(df[x])
                print(df_data)
                for j, c in enumerate(df_data.columns):
                    ax.plot(df_data.index, df_data[c], label=c)
                ax.legend()
            fig.savefig(dir_out + f'/exp_{model}_{data}_{run}.png')
            
                

