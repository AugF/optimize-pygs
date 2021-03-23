# loss, acc
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.font_manager import _rebuild
from neuroc_pygs.configs import PROJECT_PATH

_rebuild()
# plt.style.use("grayscale")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams["font.size"] = 14

dir_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_motivation_diff')
dir_out = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_figs')
ratio_dict = pd.read_csv(os.path.join(PROJECT_PATH, 'sec5_memory/exp_automl_datasets_diff', 'regression_mape_res.csv'), index_col=0)


linestyles = ['solid', 'dashed', 'dashdot', (0, (5, 5)), (0, (3, 10, 1, 10))]
predict_model = 'automl'
y_labels = ['训练损失', '训练精度']
labels = ['优化前', '优化后']
for exp_model in ['gcn', 'gat']:
    memory_ratio = ratio_dict[exp_model][predict_model] + 0.001
    fig, axes = plt.subplots(1, 2, figsize=(7, 7/2), tight_layout=True)
    
    file_name = f'/reddit_{exp_model}_175_'
    names = ['cluster_v2.csv', f'automl_{str(int(100*memory_ratio))}_mape_diff_v3.csv']

    for j, var in enumerate(['loss', 'acc']):
        ax = axes[j]
        ax.set_xlabel('批训练步')
        ax.set_ylabel(y_labels[j])
        for k in range(2):
            df = pd.read_csv(dir_path + file_name + names[k], index_col=0)
            x_smooth = np.linspace(df.index.min(), df.index.max(), 300)
            y_smooth = make_interp_spline(df.index, df[var].values)(x_smooth)
            ax.plot(x_smooth, y_smooth, label=labels[k], linestyle=linestyles[k])
            ax.legend()

    fig.savefig(dir_out + f'/exp_memory_training_{exp_model}_{predict_model}_reddit_175_train_info.png')

