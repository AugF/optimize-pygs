import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from neuroc_pygs.sec4_time.utils import datasets_maps, algorithms, sampling_modes
from matplotlib.font_manager import _rebuild
_rebuild()
config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)


def float_x(x):
    return [float(i) for i in x]


base_size = 12
plt.style.use("grayscale")

plt.rcParams["font.size"] = base_size


xs = [0.01, 0.03, 0.06, 0.1, 0.25, 0.5]

datasets_maps = {
    'pub': 'pubmed', 'amc': 'amazon-computers', 'fli': 'flickr', 'red': 'reddit'
}
algs_maps = {
    'gcn': 'GCN', 'ggnn': 'GGNN', 'gat': 'GAT', 'gaan': 'GaAN'
}
file_names = ['GCN pubmed', 'GCN amazon-computers', 'GaAN amazon-computers']
files = ['gcn_pub', 'gcn_amc', 'gaan_amc']
linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (5, 5)), (0, (3, 10, 1, 10))]
mode_maps = {
    'cluster': '聚类采样',
    'graphsage': '邻居采样'
}

for mode in ['cluster', 'graphsage']:
    for k, file in enumerate(files):
        df = {'base_times': [], 'opt_times': [], 'baseline': [], 'opt': [],
              'Baseline': [], 'Optimize': [], 'real_ratio': [], 'exp_ratio': [], 'r1': [], 'y': [], 'z': []}
        df_data = pd.read_csv(
            f'out_batch_csv/batch_size_{mode}_{file}.csv', index_col=0)
        for ds in xs:
            tmp_data = df_data.loc[ds]
            df['base_times'].append(
                float_x([tmp_data['base_sample'], tmp_data['base_move'], tmp_data['base_cal']]))
            df['opt_times'].append(
                float_x([tmp_data['opt_sample'], tmp_data['opt_move'], tmp_data['opt_cal']]))
            df['Baseline'].append(float(tmp_data['baseline']))
            df['Optimize'].append(float(tmp_data['opt']))
            df['real_ratio'].append(1/float(tmp_data['real_ratio']))
            df['exp_ratio'].append(1/float(tmp_data['exp_ratio']))
            df['r1'].append(1/float(tmp_data['r1']))
            df['y'].append(100 * float(tmp_data['y']))
            df['z'].append(100 * float(tmp_data['z']))

        base_size = 14
        # fig1
        fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
        x = np.arange(len(xs))
        locations = [-1, 1]
        colors = plt.get_cmap('Greys')(np.linspace(0.15, 0.75, 2))
        colors = [colors[-1], colors[0]]
        width = 0.35
        base_times, opt_times = np.cumsum(np.array(
            df['base_times']).T, axis=0) * 1000, np.cumsum(np.array(df['opt_times']).T, axis=0)*1000
        for i, times in enumerate([base_times, opt_times]):
            ax.bar(x + locations[i] * width / 2, times[0], width,
                   color=colors[i], edgecolor='black', hatch="///")
            ax.bar(x + locations[i] * width / 2, times[1] - times[0], width,
                   color=colors[i], edgecolor='black', bottom=times[0], hatch='...')
            ax.bar(x + locations[i] * width / 2, times[2] - times[1], width,
                   color=colors[i], edgecolor='black', bottom=times[1], hatch='xxx')

        # fig3
        fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
        ax.set_title(mode_maps[mode], fontsize=base_size+2)
        ax.set_ylabel('比值', fontsize=base_size+2)
        ax.set_xlabel('相对批规模', fontsize=base_size+2)
        line3, = ax.plot(x, df['r1'], '^r', label='优化效果', linestyle=linestyles[1])

        ax2 = ax.twinx()
        ax2.set_ylabel("50批次训练时间 (s)", fontsize=base_size + 2)
        
        line4, = ax2.plot(x, [1000 * t for t in df['Baseline']], 'ob', linestyle=linestyles[1],
                          label='优化前耗时')
        line5, = ax2.plot(x, [1000 * t for t in df['Optimize']], 'Dg', linestyle=linestyles[2],
                           label='优化后耗时')
        plt.legend(handles=[line3,
                            line4, line5])
        plt.xticks(ticks=x, labels=[
                   f'{int(100*s)}%' for s in xs], fontsize=base_size + 2)
        plt.yticks(fontsize=base_size + 2)
        fig.tight_layout()  # 防止重叠

        fig.savefig(
            f'exp4_thesis_ppt_figs/batch_figs/exp_batch_batch_size_{mode}_else_{file}.png', dpi=400)
