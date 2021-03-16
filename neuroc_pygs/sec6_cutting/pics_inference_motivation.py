# BatchSize下的内存使用的箱线图和时间,精度的图
# https://matplotlib.org/2.0.2/examples/pylab_examples/boxplot_demo2.html
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.options import get_args
from matplotlib.font_manager import _rebuild
_rebuild() 

plt.style.use("grayscale")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams["font.size"] = 14

colors = plt.get_cmap('Greys')(np.linspace(0.15, 0.85, 2))
mode = 'cluster'
for model in ['reddit_sage', 'cluster_gcn']:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 7/2), tight_layout=True)
    for i, data in enumerate(['reddit', 'yelp']):
        ax = axes[i]
    
        ax.set_title(data.capitalize())
        ax.set_ylabel('峰值内存 (GB)', fontsize=14)
        ax.set_xlabel('批规模', fontsize=14)

        box_data = []

        batch_sizes = [1024, 2048, 4096, 8192, 16384]
        
        for bs in batch_sizes:
            # read file
            file_name = '_'.join([data, model, str(bs), mode, 'linear_model_v1'])
            real_path = os.path.join(PROJECT_PATH, 'sec5_memory/exp_motivation', file_name) + '.csv'
            print(real_path)
            if os.path.exists(real_path):
                res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')['memory']
                print(file_name, res)
            else:
                res = []
            box_data.append(list(map(lambda x: x/(1024*1024*1024), res)))
        bp = ax.boxplot(box_data, labels=batch_sizes)
        
    fig.savefig(os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_figs', f'{model}_{mode}_motivation_linear_model.png'))


