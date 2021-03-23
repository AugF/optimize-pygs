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

df = {
    'amazon-computers': {
        'Baseline': [25.7639, 73.323, 33.5073, 90.9117],
        'Optimize': [21.043, 60.3804, 28.4334, 66.6091],
    },
    'flickr': {
        'Baseline': [52.0265, 353.426, 63.8889, 157.202],
        'Optimize': [38.761, 264.562, 50.5601, 115.472]
    }
}

root_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time'
colors = plt.get_cmap('Greys')(np.linspace(0.15, 0.85, 2))
colors = [colors[-1], colors[0]]

i = 0
for item in df.keys():
    tab_data = df[item]

    xs = ['GCN', 'GGNN', 'GAT', 'GaAN']

    x = np.arange(len(xs))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
    ax.set_title(item.capitalize(), fontsize=base_size+2)
    ax.set_ylabel('100轮训练时间 (秒)', fontsize=base_size+2)
    ax.set_xlabel('算法', fontsize=base_size+2)
    ax.set_xticks(x)
    ax.set_xticklabels(xs, fontsize=base_size+2)
    ax.bar(x - width/2, tab_data['Baseline'], width, color=colors[0], edgecolor='black', label='优化前')
    ax.bar(x + width/2, tab_data['Optimize'], width, color=colors[1], edgecolor='black', label='优化后')
    if i == 0:
        ax.legend(loc='upper left')
    else:
        ax.legend(loc='upper right')
    fig.savefig(root_path + f'/exp_figs_final/exp_epoch_full_models_{item}.png')
    i += 2


