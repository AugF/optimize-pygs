import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from neuroc_pygs.sec4_time.utils import algorithms, sampling_modes
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


file_names = ['gcn', 'gat']
# vs = ['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
datasets = ['pubmed', 'amazon-computers', 'flickr', 'reddit']
modes = ['graphsage', 'cluster']
titles = {
    'gcn': 'GCN', 'ggnn': 'GGNN', 'gat': 'GAT', 'gaan': 'GaAN'
}
algs = ['gcn', 'gaan']

datasets_maps = {
    'pubmed': 'pub', 'amazon-computers': 'amc', 'flickr': 'fli', 'reddit': 'red'
}

df_data = pd.read_csv(f'out_batch_csv/batches_infer.csv', index_col=0)
if True:
    df = {}
    for alg in algs:
        df[alg] = {'base_times': [], 'opt_times': [],
                   'baseline': [], 'opt': []}

    for data in datasets:
        for alg in algs:
            tmp_data = df_data.loc[f'{alg}_{data}_1024']
            df[alg]['base_times'].append(
                float_x([tmp_data['base_sample'], tmp_data['base_move'], tmp_data['base_cal']]))
            df[alg]['opt_times'].append(
                float_x([tmp_data['opt_sample'], tmp_data['opt_move'], tmp_data['opt_cal']]))
    base_size = 12
    for alg in algs:
        fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
        xticklabels = [datasets_maps[d] for d in datasets]
        x = np.arange(len(xticklabels))

        locations = [-1, 1]
        colors = plt.get_cmap('Greys')(np.linspace(0.15, 0.75, 2))
        colors = [colors[-1], colors[0]]
        # colors = ['blue', 'cyan']
        width = 0.35
        base_times, opt_times = np.cumsum(np.array(
            df[alg]['base_times']).T, axis=0) * 1000, np.cumsum(np.array(df[alg]['opt_times']).T, axis=0)*1000
        for i, times in enumerate([base_times, opt_times]):
            ax.bar(x + locations[i] * width / 2, times[0], width,
                   color=colors[i], edgecolor='black', hatch="///")
            ax.bar(x + locations[i] * width / 2, times[1] - times[0], width,
                   color=colors[i], edgecolor='black', bottom=times[0], hatch='...')
            ax.bar(x + locations[i] * width / 2, times[2] - times[1], width,
                   color=colors[i], edgecolor='black', bottom=times[1], hatch='xxx')
            # ??????: error bar

        ax.set_title(titles[alg], fontsize=base_size+2)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, fontsize=base_size+2)

        legend_colors = [Patch(facecolor=c, edgecolor='black') for c in colors]
        legend_hatchs = [Patch(facecolor='white', edgecolor='black', hatch='xxxx'), Patch(
            facecolor='white', edgecolor='black', hatch='....'), Patch(facecolor='white', edgecolor='black', hatch='////')]
        ax.legend(legend_hatchs + legend_colors, ['GPU??????', '????????????', '??????'] + [
                  '?????????', '?????????'], ncol=2, loc='upper left', fontsize='x-small')
        # if mode == 'cluster':
        #     ax.set_ylim(0, 100)
        # else:
        #     ax.set_ylim(0, 2000)
        ax.set_ylabel('????????????????????? (ms)', fontsize=base_size+2)
        ax.set_xlabel('?????????', fontsize=base_size+2)
        fig.savefig(
            f'exp4_thesis_figs/batch_infer_figs/exp_batch_infer_datasets_batch_{alg}.png', dpi=400)
        plt.close()


if True:
    df = {}
    for alg in algs:
        df[alg] = {'Baseline': [], 'Optimize': [], 'real_ratio': [],
                   'exp_ratio': [], 'r1': [], 'y': [], 'z': []}

    for data in datasets:
        for alg in algs:
            tmp_data = df_data.loc[f'{alg}_{data}_1024']
            df[alg]['Baseline'].append(float(tmp_data['baseline']))
            df[alg]['Optimize'].append(float(tmp_data['opt']))
            df[alg]['real_ratio'].append(1/float(tmp_data['real_ratio']))
            df[alg]['exp_ratio'].append(1/float(tmp_data['exp_ratio']))
            df[alg]['r1'].append(1/float(tmp_data['r1']))
            df[alg]['y'].append(100 * float(tmp_data['y']))
            df[alg]['z'].append(100 * float(tmp_data['z']))

    base_size = 12
    for alg in algs:
        tab_data = df[alg]

        xs = [datasets_maps[d] for d in datasets]

        x = np.arange(len(xs))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
        ax.set_title(titles[alg], fontsize=base_size+2)
        ax.set_ylabel('50?????????????????? (s)', fontsize=base_size+2)
        ax.set_xlabel('?????????', fontsize=base_size+2)
        ax.set_xticks(x)
        ax.set_xticklabels(xs, fontsize=base_size+2)
        ax.bar(x - width/2, tab_data['Baseline'], width,
               color=colors[0], edgecolor='black', label='?????????')
        ax.bar(x + width/2, tab_data['Optimize'], width,
               color=colors[1], edgecolor='black', label='?????????')
        ax.legend()
        fig.savefig(
            f'exp4_thesis_figs/batch_infer_figs/exp_batch_infer_datasets_total_{alg}.png', dpi=400)

    base_size = 14
    for alg in algs:
        tab_data = df[alg]

        xs = [datasets_maps[d] for d in datasets]
        x = np.arange(len(xs))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
        ax.set_title(titles[alg], fontsize=base_size+2)
        ax.set_ylabel('??????', fontsize=base_size+2)
        ax.set_xlabel('?????????', fontsize=base_size+2)
        line1, = ax.plot(x, tab_data['exp_ratio'],
                         'ob', label='???????????????', linestyle='-')
        line2, = ax.plot(x, tab_data['real_ratio'],
                         'Dg', label='???????????????', linestyle='-')
        line3, = ax.plot(x, tab_data['r1'], '^r', label='????????????', linestyle='-')

        ax2 = ax.twinx()
        ax2.set_ylabel("???????????? (%)", fontsize=base_size + 2)
        line4, = ax2.plot(x, tab_data['y'], 's--',
                          color='black', label='??????????????????' + r"$Y$")
        line5, = ax2.plot(x, tab_data['z'], 'd--',
                          color='black', label='????????????????????????' + r"$Z$")
        plt.legend(handles=[line1, line2, line3,
                            line4, line5], fontsize='xx-small')
        plt.xticks(ticks=x, labels=xs, fontsize=base_size)
        plt.yticks(fontsize=base_size)
        fig.tight_layout()  # ????????????

        fig.savefig(
            f'exp4_thesis_figs/batch_infer_figs/exp_batch_infer_datasets_else_{alg}.png', dpi=400)
