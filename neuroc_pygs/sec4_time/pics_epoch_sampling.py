import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from neuroc_pygs.sec4_time.utils import datasets_maps, algorithms, sampling_modes

plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

root_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time'
real_path = root_path + '/exp_res/epoch/sampling_epochs.log'
headers = None
mydata = []
with open(real_path) as f:
    for line in f.readlines():
        if headers == None:
            headers = [l.strip() for l in line.split('|')][1:]
        else:
            mydata.append([l.strip() for l in line.split('|')][1:])

df = pd.DataFrame(mydata, columns=headers)
df.index = df['Name']
del df['Name']
print(df)

# model
tab_data = defaultdict(list)
xs = ['gcn', 'ggnn', 'gat', 'gaan']
for model in xs:
    index = f'{model}_amazon-computers'
    tmp_data = df.loc[index]
    tab_data['Baseline'].append(float(tmp_data['Baseline']))
    tab_data['Batch Opt'].append(float(tmp_data['Batch Opt']))
    tab_data['Epoch Opt'].append(float(tmp_data['Epoch Opt']))
    tab_data['Opt'].append(float(tmp_data['Opt']))

x = np.arange(len(xs))
width = 0.2
fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
ax.set_ylabel('Training Time (s)')
ax.set_xlabel('Algorithm')
ax.set_xticks(x)
ax.set_xticklabels([algorithms[i] for i in xs])

ax.bar(x - 1.5 * width, tab_data['Baseline'], width, label='Baseline')
ax.bar(x - 0.5 * width, tab_data['Batch Opt'], width, label='Batch Optimize')
ax.bar(x + 0.5 * width, tab_data['Epoch Opt'], width, label='Epoch Optimize')
ax.bar(x + 1.5 * width, tab_data['Opt'], width, label='Total Optimize')
ax.legend(fontsize='large')
fig.savefig(root_path + f'/exp_figs/exp_epoch_sampling_amazon-computers.png')

# datasets
tab_data = defaultdict(list)
xs = ['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'flickr']
for data in xs:
    index = f'gcn_{data}'
    tmp_data = df.loc[index]
    tab_data['Baseline'].append(float(tmp_data['Baseline']))
    tab_data['Batch Opt'].append(float(tmp_data['Batch Opt']))
    tab_data['Epoch Opt'].append(float(tmp_data['Epoch Opt']))
    tab_data['Opt'].append(float(tmp_data['Opt']))


locations = [-1.5, -0.5, 0.5, 1.5]
x = np.arange(len(xs))
width = 0.2
fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
ax.set_ylabel('Training Time (s)')
ax.set_xlabel('Dataset')
ax.set_xticks(x)
ax.set_xticklabels([datasets_maps[i] for i in xs])
ax.bar(x - 1.5 * width, tab_data['Baseline'], width, label='Baseline')
ax.bar(x - 0.5 * width, tab_data['Batch Opt'], width, label='Batch Optimize')
ax.bar(x + 0.5 * width, tab_data['Epoch Opt'], width, label='Epoch Optimize')
ax.bar(x + 1.5 * width, tab_data['Opt'], width, label='Total Optimize')
ax.legend(fontsize='large')
fig.savefig(root_path + f'/exp_figs/exp_epoch_sampling_gcn.png')

