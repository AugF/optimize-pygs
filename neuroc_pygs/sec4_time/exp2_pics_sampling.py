import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.sec4_time.utils import datasets_maps, algorithms, sampling_modes

small_datasets = ['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
models = ['gcn', 'ggnn', 'gat', 'gaan']


dir_path = os.path.join(PROJECT_PATH, 'sec4_time', 'exp_res')
df = pd.read_csv(os.path.join(dir_path, 'sampling_inference_small_datasets_0_cpu_final.csv'), index_col=0)
df.index = df['Name']
del df['Name']
print(df)

for model in models:
    base_times, opt_times, opt_error = [], [], []
    for data in small_datasets:
        index = f'{data}_{model}'
        tmp_data = df.loc[index]
        base_times.append(tmp_data['Base Sampling'], tmp_data['Base Transfer'], tmp_data['Base Training'])
        opt_times.append(tmp_data['Opt Sampling'], tmp_data['Opt Transfer'], tmp_data['Opt Training'])
        opt_error.append(tmp_data['Opt max'], tmp_data['Opt min'])
    base_times, opt_times, opt_error = np.array(base_times.T).cumsum(axis=0), np.array(opt_times.T).cumsum(axis=0), np.array(opt_error.T)

    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    xticklabels = [datasets_maps[_] for _ in small_datasets]
    x = np.arange(len(xticklabels))

    locations = [-1, 1]
    colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))
    width = 0.4
    for i, times in enumerate([base_times, opt_times]):
        ax.bar(x + locations[i] * width, times[0], width, color=colors[i], edgecolor='black', hatch="////")
        ax.bar(x + locations[i] * width, times[1], width, color=colors[i], edgecolor='black', bottom=times[0], hatch='....')
        ax.bar(x + locations[i] * width, times[2], width, color=colors[i], edgecolor='black', bottom=times[1], hatch='xxxx')
        # 待做: error bar
    ax.set_title(algorithms[model])
    ax.set_xticklabels(xticklabels)

    legend_colors = [Line2D([0], [0], color=c, lw=4) for c in colors]
    legend_hatchs = [Patch(facecolor='white', edgecolor='r', hatch='////'), Patch(facecolor='white',edgecolor='r', hatch='....'), Patch(facecolor='white', edgecolor='r', hatch='xxxx')]
    ax.legend(legend_colors + legend_hatchs, ['Baseline', 'Optimize'] + ['Inference on GPU', 'Data Transferring', 'Sampling'], ncol=2, loc="upper left", fontsize=16)
    
    fig.savefig(dir_path + f'/exp_sampling_inference_{model}.png')
    plt.close()


