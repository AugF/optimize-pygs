import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

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

for item in df.keys():
    tab_data = df[item]

    xs = ['GCN', 'GGNN', 'GAT', 'GaAN']

    x = np.arange(len(xs))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    ax.set_ylabel('Training Time (s)')
    ax.set_xlabel('Algorithm')
    ax.set_xticks(x)
    ax.set_xticklabels(xs)
    ax.bar(x - width/2, tab_data['Baseline'], width, label='Baseline')
    ax.bar(x + width/2, tab_data['Optimize'], width, label='Optimize')
    ax.legend(fontsize='large')
    fig.savefig(root_path + f'/exp_figs/exp_epoch_full_batchs_{item}.png')


