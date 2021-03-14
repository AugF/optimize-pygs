import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

root_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time'
real_path = root_path + '/exp_res/batch/threads_constrast.csv'
df = pd.read_csv(real_path, index_col=0)
df.index = df['Name']
del df['Name']
print(df)

for file in ['flickr_gat', 'amazon-computers_gcn']:
    tab_data = []
    for nw in [0, 10, 20, 30, 40]:
        file_name = f'{file}_None_cluster_pin_memory_False_num_workers_{nw}_non_blocking_False'
        print(file_name)
        tmp = []
        for name in ['Base', 'Opt']:
            times = 0
            for st in ['Transfer', 'Sampling', 'Training']:
                times += df.loc[file_name][name + ' ' + st] * 1000
            tmp.append(times)
        tab_data.append(tmp)

    tab_data = np.array(tab_data).T
    xs = [0, 10, 20, 30, 40]

    x = np.arange(len(xs))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    ax.set_ylabel('Training Time Per Batch (ms)')
    ax.set_xlabel('Num Workers')
    ax.set_xticklabels([''] + xs)
    ax.bar(x - width/2, tab_data[0], width, label='Baseline')
    ax.bar(x + width/2, tab_data[1], width, label='Optimize')
    ax.legend(fontsize='large')
    fig.savefig(root_path + f'/exp_figs/exp_batch_threads_{file}.png')


