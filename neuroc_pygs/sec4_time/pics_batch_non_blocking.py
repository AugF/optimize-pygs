import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

root_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time'
real_path = root_path + '/exp_res/batch/non_blocking_constrast.csv'
df = pd.read_csv(real_path, index_col=0)
df.index = df['Name']
del df['Name']
print(df)

xs = ['True', 'False']

fig, axes = plt.subplots(1, 2, figsize=(7*2, 5), tight_layout=True)
for i, file in enumerate(['flickr_gat', 'amazon-computers_gcn']):
    tab_data = []
    for v in xs:
        file_name = f'{file}_None_cluster_pin_memory_False_num_workers_0_non_blocking_{v}'
        print(file_name)
        tmp = []
        for name in ['Base', 'Opt']:
            times = 0
            for st in ['Transfer', 'Sampling', 'Training']:
                times += df.loc[file_name][name + ' ' + st] * 1000
            tmp.append(times)
        tab_data.append(tmp)

    tab_data = np.array(tab_data).T

    x = np.arange(len(xs))  # the label locations
    width = 0.35  # the width of the bars

    ax = axes[i]
    ax.set_ylabel('Training Time Per Batch (ms)')
    ax.set_xlabel('Non Blocking', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(xs, fontsize=16)
    ax.bar(x - width/2, tab_data[0], width, label='Baseline')
    ax.bar(x + width/2, tab_data[1], width, label='Optimize')
    ax.legend(fontsize='large')
fig.savefig(root_path + f'/exp_figs/exp_batch_non_blocking.png')


