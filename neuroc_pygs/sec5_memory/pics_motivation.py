import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.options import get_args


datasets = ['reddit', 'yelp', 'amazon']
# cluster
re_batch_sizes = [0.002, 0.004, 0.006, 0.008, 0.01]
cluster_labels = [f'{rs * 1500}%' for rs in re_batch_sizes]
# graphsage
sizes = [[25, 10], [50, 25], [5, 10]]
batch_sizes = [2048, 4096, 10240]
graphsage_labels = [str(bs) for bs in batch_sizes]
modes = ['cluster', 'graphsage']


plt.style.use("ggplot")

mode = 'cluster'
for model in ['gcn', 'gat']:
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7 * 3, 5), tight_layout=True)
    for i, data in enumerate(datasets):
        ax = axes[i]
        
        ax.set_ylabel('Peak Memory (Byte)')
        ax.set_xlabel('Batch Size')

        box_data = []

        for re_bs in re_batch_sizes:
            # read file
            file_name = '_'.join([data, model, str(re_bs), mode])
            real_path = os.path.join(PROJECT_PATH, 'sec5_memory/motivation', file_name) + '.csv'
            if os.path.exists(real_path):
                res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')['memory']
                print(file_name, res)
            else:
                res = []
            box_data.append(res)
        ax.boxplot(box_data, labels=cluster_labels)
        
    fig.savefig(os.path.join(PROJECT_PATH, 'sec5_memory', 'tmp', f'{model}_{mode}_motivation.png'))


mode = 'graphsage'
for size in sizes:
    for model in ['gcn', 'gat']:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7 * 3, 5), tight_layout=True)
        for i, data in enumerate(datasets):
            ax = axes[i]
            
            ax.set_ylabel('Peak Memory (Byte)')
            ax.set_xlabel('Batch Size')

            box_data = []

            for bs in batch_sizes:
                # read file
                file_name = '_'.join([data, model, str(size), str(bs), mode])
                real_path = os.path.join(PROJECT_PATH, 'sec5_memory/motivation', file_name) + '.csv'
                print(real_path)
                if os.path.exists(real_path):
                    res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')['memory']
                    print(file_name)
                else:
                    res = []
                box_data.append(res)
            ax.boxplot(box_data, labels=graphsage_labels)
            
        fig.savefig(os.path.join(PROJECT_PATH, 'sec5_memory', 'tmp', f'{model}_{mode}_{str(size)}_motivation.png'))


