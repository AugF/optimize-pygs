import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.options import get_args


plt.style.use("ggplot")
plt.rcParams["font.size"] = 12
datasets = ['yelp', 'amazon']


for model in ['gat']:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 7/2), tight_layout=True)
    for i, data in enumerate(datasets):
        ax = axes[i]
    
        ax.set_title(data.capitalize())
        ax.set_ylabel('Peak Memory (GB)', fontsize=14)
        ax.set_xlabel('Batch Size', fontsize=14)

        box_data = []

        for bs in []:
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

