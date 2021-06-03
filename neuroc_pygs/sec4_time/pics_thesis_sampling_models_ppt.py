import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from neuroc_pygs.sec4_time.utils import datasets_maps, algorithms, sampling_modes
from matplotlib.font_manager import _rebuild
_rebuild()

base_size = 16
plt.style.use("grayscale")
plt.rcParams["font.size"] = base_size


headers = ['Name', 'Baseline', 'Batch Opt', 'Epoch Opt',
           'Opt', 'Batch Ratio%', 'Epoch Raio%', 'Opt%']
mode = 'cluster'

df_data = []
for exp_data in ['amazon-computers', 'flickr', 'pubmed']:
    for model in ['gcn', 'gaan', 'gat', 'ggnn']:
        real_path = 'out_total_csv/' + model + '_' + exp_data + '_' + mode + '_None.csv'
        res = open(real_path).read().split(',')
        df_data.append(res)
df = pd.DataFrame(df_data, columns=headers)
df.index = [x for x in df['Name']]
del df['Name']
print(df)

# models
xs = ['gcn', 'ggnn', 'gat', 'gaan']
for data in ['amazon-computers', 'flickr', 'pubmed']:
    tab_data = defaultdict(list)
    for model in xs:
        index = f'{model}_{data}_{mode}_None'
        tmp_data = df.loc[index]
        tab_data['Baseline'].append(float(tmp_data['Baseline']))
        tab_data['Batch Opt'].append(float(tmp_data['Batch Opt']))
        tab_data['Epoch Opt'].append(float(tmp_data['Epoch Opt']))
        tab_data['Opt'].append(float(tmp_data['Opt']))

    x = np.arange(len(xs))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
    ax.set_title(data, fontsize=base_size+2)
    ax.set_ylabel('30轮训练时间 (s)', fontsize=base_size+2)
    ax.set_xlabel('算法', fontsize=base_size+2)
    ax.set_xticks(x)
    ax.set_xticklabels([algorithms[i] for i in xs], fontsize=18)

    ax.bar(x - 1.5 * width, tab_data['Baseline'], width, label='未优化')
    ax.bar(x - 0.5 * width, tab_data['Batch Opt'], width, label='优化1')
    ax.bar(x + 0.5 * width, tab_data['Epoch Opt'], width, label='优化2')
    ax.bar(x + 1.5 * width, tab_data['Opt'], width, label='优化1+优化2')
    if data == 'flickr':
        ax.legend(fontsize='x-small', ncol=1)
    else:
        ax.legend(fontsize='x-small', loc='lower center', ncol=2)
    fig.savefig(
        f'exp4_thesis_figs/total_figs/exp_epoch_sampling_models_{mode}_{data}.png', dpi=400)
