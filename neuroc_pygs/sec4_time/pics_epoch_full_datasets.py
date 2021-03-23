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
    'gcn': {
        'Baseline': [25.7639, 16.1384, 13.1664, 76.0500, 52.0265],
        'Optimize': [21.043, 15.8575, 14.67763, 61.0040, 38.761],
    },
    'gat': {
        'Baseline': [33.5073, 19.3508, 14.2551, 59.0031, 63.8889],
        'Optimize': [28.4334, 17.6148, 15.23369, 48.8233, 50.5601]
    }
}

root_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time'
colors = plt.get_cmap('Greys')(np.linspace(0.15, 0.85, 2))
colors = [colors[-1], colors[0]]
    

for i, item in enumerate(df.keys()):
    tab_data = df[item]

    xs = ['amp', 'pub', 'amc', 'cph', 'fli']
    
    x = np.arange(len(xs))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
    ax.set_title(item.upper(), fontsize=base_size+2)
    ax.set_ylabel('100轮训练时间 (秒)', fontsize=base_size+2)
    ax.set_xlabel('数据集', fontsize=base_size+2)
    ax.set_xticks(x)
    ax.set_xticklabels(xs, fontsize=base_size+2)
    ax.bar(x - width/2, tab_data['Baseline'], width, color=colors[0], edgecolor='black', label='优化前')
    ax.bar(x + width/2, tab_data['Optimize'], width, color=colors[1], edgecolor='black', label='优化后')
    ax.legend(loc='upper left')
    fig.savefig(root_path + f'/exp_figs_final/exp_epoch_full_datasets_{item}.png')