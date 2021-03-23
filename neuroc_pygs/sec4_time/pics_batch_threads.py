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

root_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time'
real_path = root_path + '/exp_res/sampling_training_contrast.txt'
df = pd.read_csv(real_path, index_col=0)
print(df)

titles = ['Flickr GAT', 'Amazon-computers GCN']

for k, file in enumerate(['flickr_gat', 'amazon-computers_gcn']):
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

    colors = plt.get_cmap('Greys')(np.linspace(0.15, 0.85, 2))
    colors = [colors[-1], colors[0]]
    
    fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
    ax.set_title(titles[k], fontsize=base_size + 2)
    ax.set_ylabel('每轮训练时间 (毫秒)', fontsize=base_size + 2)
    ax.set_xlabel('线程数', fontsize=base_size + 2)
    ax.set_xticks(x)
    ax.set_xticklabels(xs, fontsize=base_size + 2)
    ax.bar(x - width/2, tab_data[0], width, color=colors[0], edgecolor='black', label='优化前')
    ax.bar(x + width/2, tab_data[1], width, color=colors[1], edgecolor='black', label='优化后')
    ax.legend()
    fig.savefig(root_path + f'/exp_figs_final/exp_batch_threads_{file}.png')


