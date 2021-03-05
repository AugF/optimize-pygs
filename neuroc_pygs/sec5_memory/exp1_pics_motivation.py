import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.options import get_args


plt.style.use("ggplot")
plt.rcParams["font.size"] = 12
datasets = ['yelp', 'amazon']

mode = 'cluster'
for model in ['gat']:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 7/2), tight_layout=True)
    for i, data in enumerate(datasets):
        ax = axes[i]
    
        ax.set_title(data.capitalize())
        ax.set_ylabel('Peak Memory (GB)', fontsize=14)
        ax.set_xlabel('Batch Size', fontsize=14)

        box_data = []

        if data == 'amazon':
            re_batch_sizes = [0.01, 0.02, 0.04]
        else:
            re_batch_sizes = [0.1, 0.2, 0.4]
        
        for re_bs in re_batch_sizes:
            # read file
            file_name = '_'.join([data, model, str(re_bs), mode, 'final'])
            real_path = os.path.join(PROJECT_PATH, 'sec5_memory/motivation', file_name) + '.csv'
            if os.path.exists(real_path):
                res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')['memory']
                print(file_name, res)
            else:
                res = []
            box_data.append(list(map(lambda x: x/(1024*1024*1024), res)))
        ax.boxplot(box_data, labels=[int(rs * 1500) for rs in re_batch_sizes])
        
    fig.savefig(os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_figs', f'{model}_{mode}_motivation.png'))


mode = 'graphsage'
for model in ['gat']:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 7/2), tight_layout=True)
    for i, data in enumerate(datasets):
        ax = axes[i]
        
        ax.set_title(data.capitalize())
        ax.set_ylabel('Peak Memory (GB)', fontsize=14)
        ax.set_xlabel('Batch Size', fontsize=14)

        box_data = []
        batch_sizes = [51200, 102400, 204800]
        for bs in batch_sizes:
            # read file
            file_name = '_'.join([data, model, str(bs), mode, 'final'])
            real_path = os.path.join(PROJECT_PATH, 'sec5_memory/motivation', file_name) + '.csv'
            print(real_path)
            if os.path.exists(real_path):
                res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')['memory']
                print(file_name)
            else:
                res = []
            box_data.append(list(map(lambda x: x/(1024*1024*1024), res)))
        ax.boxplot(box_data, labels=[str(bs) for bs in batch_sizes])
        
    fig.savefig(os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_figs', f'{model}_{mode}_motivation.png'))


