# loss, acc
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.font_manager import _rebuild
from neuroc_pygs.configs import PROJECT_PATH

_rebuild()

# plt.style.use("grayscale")
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.rcParams["font.size"] = 14

dir_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_motivation_final')
dir_out = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_train_figs')
ratio_dict = pd.read_csv(os.path.join(PROJECT_PATH, 'sec5_memory/exp_automl_datasets_final', 'regression_res.csv'), index_col=0)

# liner_model
def run(predict_model):
    for exp_model in ['gcn', 'gat']:
        memory_ratio = ratio_dict[exp_model][predict_model]
        for exp_data in ['reddit', 'yelp']:
            if exp_data == 'reddit' and exp_model == 'gat':
                re_bs = [170, 175, 180]
            else:
                re_bs = [175, 180, 185]
            
            fig, axes = plt.subplots(1, 2, figsize=(7, 5), tight_layout=True)
            ax_loss, ax_acc = axes[0], axes[1]
            ax_loss.set_title(f'{exp_model}_{exp_data}')
            ax_acc.set_title(f'{exp_model}_{exp_data}')
            ax_loss.set_xlabel('批训练步')
            ax_acc.set_xlabel('批训练步')
            ax_loss.set_ylabel('训练损失')
            ax_acc.set_ylabel('训练精度')
            
            for bs in [175]:
                file_name = f'{exp_data}_{exp_model}_{bs}_'
                # pics figs
                methods = ['baseline', predict_model]
                for j, var in enumerate(['cluster_v2', f'{predict_model}_{str(int(100*memory_ratio))}_v2_1']):
                    real_path = dir_path + '/' + file_name + var + '.csv'
                    print(real_path)
                    df = pd.read_csv(real_path, index_col=0)
                    losses, accs = df['loss'].values, df['acc'].values
                    ax_loss.plot(df.index, losses, label=f'{methods[j]}_{bs}')
                    ax_acc.plot(df.index, accs, label=f'{methods[j]}_{bs}')                
            ax_loss.legend()
            ax_acc.legend()
            fig.savefig(dir_out + f'/exp_train_{predict_model}_{exp_model}_{exp_data}.png')


if __name__ == '__main__':
    for predict_model in ['linear_model', 'automl']:
        run(predict_model)