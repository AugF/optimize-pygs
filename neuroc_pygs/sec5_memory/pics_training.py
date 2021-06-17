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

dir_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_motivation_diff')
dir_out = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp5_thesis_figs')
ratio_dict = pd.read_csv(os.path.join(PROJECT_PATH, 'sec5_memory/exp_automl_datasets_diff', 'regression_mape_res.csv'), index_col=0)
ratio_dict = pd.read_csv(os.path.join(PROJECT_PATH, 'sec5_memory/exp_automl_datasets_diff', 'regression_linear_mape_res.csv'), index_col=0)

# 实际的展示前后的图
def fun(model, data, batch_size, predict_model='linear_model'):
    linestyles = ['solid', 'dashed', 'dashdot', (0, (5, 5)), (0, (3, 10, 1, 10))]
    y_labels = ['训练损失', '测试集精度']
    labels = ['内存超限处理流程', '常规处理流程']
    for exp_model in [model]:
        for exp_data in [data]:
            for bs in [batch_size]:
                if predict_model == 'automl':
                    memory_ratio = ratio_dict[exp_model][predict_model] + 0.001
                else:
                    memory_ratio = ratio_dict[exp_model][exp_data] + 0.001
                
                file_name = f'/{exp_data}_{exp_model}_{bs}_'
                # names = ['cluster_v2.csv', f'automl_{str(int(100*memory_ratio))}_mape_diff_v2.csv']
                names = ['cluster_v2.csv', f'{predict_model}_{str(int(100*memory_ratio))}_copy.csv']

                for j, var in enumerate(['loss', 'acc']):
                    fig, ax = plt.subplots(figsize=(7/2, 6/2), tight_layout=True)
                    ax.set_xlabel('批训练步')
                    ax.set_ylabel(y_labels[j])
                    for k in range(2):
                        df = pd.read_csv(dir_path + file_name + names[k], index_col=0)
                        x_smooth = np.linspace(df.index.min(), df.index.max(), 300)
                        y_smooth = make_interp_spline(df.index, df[var].values)(x_smooth)
                        ax.plot(x_smooth, y_smooth, label=labels[k], linestyle=linestyles[k])
                        ax.legend(fontsize='x-small')
                    fig.savefig(f'exp5_thesis_figs/resampling/exp_memory_training_{exp_model}_{predict_model}_{exp_data}_{bs}_{var}.png', dpi=400)
        
fun('gat', 'yelp', '180', 'linear_model')
# fun('gat', 'reddit', '180', 'linear_model')

