# loss, acc
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.font_manager import _rebuild
from neuroc_pygs.configs import PROJECT_PATH

_rebuild()
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)
plt.rcParams["font.size"] = 14


linestyles = ['solid', 'dashed', 'dashdot', (0, (5, 5)), (0, (3, 10, 1, 10))]
y_labels = ['训练损失', '测试集精度']
labels = ['内存超限处理流程', '常规处理流程']
names = ['resampling.csv', 'origin.csv']

for j, var in enumerate(['loss', 'acc']):
    fig, ax = plt.subplots(figsize=(7/2, 6/2), tight_layout=True)
    ax.set_xlabel('批训练步')
    ax.set_ylabel(y_labels[j])
    for k in range(2):
        df = pd.read_csv('out_train_csv/gat_yelp_180_' + names[k])
        x_smooth = np.linspace(df.index.min(), df.index.max(), 300)
        y_smooth = make_interp_spline(df.index, df[var].values)(x_smooth)
        ax.plot(x_smooth, y_smooth, label=labels[k], linestyle=linestyles[k])
        ax.legend(fontsize='x-small')
    fig.savefig(f'exp5_thesis_figs/resampling/exp_memory_training_gat_yelp_180.png', dpi=400)

