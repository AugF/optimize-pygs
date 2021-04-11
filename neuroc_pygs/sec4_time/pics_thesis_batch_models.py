import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from neuroc_pygs.sec4_time.utils import datasets_maps, algorithms, sampling_modes
from matplotlib.font_manager import _rebuild
# print(_rebuild())
_rebuild() 

def float_x(x):
    return [float(i) for i in x]

base_size = 12
plt.style.use("grayscale")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams["font.size"] = base_size


file_names = ['gcn', 'gat']
datasets = ['pub', 'fli']
datasets_maps = {
    'pub': 'pubmed', 'amc': 'amazon-computers', 'fli': 'flickr', 'red': 'reddit'
}

algs_maps = {
    'gcn': 'GCN', 'ggnn': 'GGNN', 'gat': 'GAT', 'gaan': 'GaAN'
}
MODES = ['邻居采样', '聚类采样']
algs = ['gcn', 'ggnn', 'gat', 'gaan']

for i, mode in enumerate(['cluster']):
    df = {}
    for data in datasets:
        df[data] = {'base_times': [], 'opt_times': [], 'baseline': [], 'opt': []}

    for data in datasets:
        df_data = pd.read_csv(f'out_batch_csv/models_{mode}_{data}.csv', index_col=0)
        for alg in algs:
            tmp_data = df_data.loc[alg]
            df[data]['base_times'].append(float_x([tmp_data['base_sample'], tmp_data['base_move'], tmp_data['base_cal']]))
            df[data]['opt_times'].append(float_x([tmp_data['opt_sample'], tmp_data['opt_move'], tmp_data['opt_cal']]))
    
    base_size = 12
    for data in datasets:
        fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
        xticklabels = [algs_maps[alg] for alg in algs]
        x = np.arange(len(xticklabels))

        locations = [-1, 1]
        colors = plt.get_cmap('Greys')(np.linspace(0.15, 0.75, 2))
        colors = [colors[-1], colors[0]]
        # colors = ['blue', 'cyan']
        width = 0.35
        base_times, opt_times = np.cumsum(np.array(df[data]['base_times']).T, axis=0) * 1000, np.cumsum(np.array(df[data]['opt_times']).T, axis=0)*1000
        for i, times in enumerate([base_times, opt_times]):
            ax.bar(x + locations[i] * width / 2, times[0], width, color=colors[i], edgecolor='black', hatch="///")
            ax.bar(x + locations[i] * width / 2, times[1] - times[0], width, color=colors[i], edgecolor='black', bottom=times[0], hatch='...')
            ax.bar(x + locations[i] * width / 2, times[2] - times[1], width, color=colors[i], edgecolor='black', bottom=times[1], hatch='xxx')
            # 待做: error bar

        ax.set_title(datasets_maps[data], fontsize=base_size+2)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, fontsize=base_size+2)

        legend_colors = [Patch(facecolor=c, edgecolor='black') for c in colors]
        legend_hatchs = [Patch(facecolor='white', edgecolor='black', hatch='xxxx'), Patch(facecolor='white',edgecolor='black', hatch='....'), Patch(facecolor='white', edgecolor='black', hatch='////')]
        ax.legend(legend_hatchs + legend_colors, ['GPU计算', '数据传输', '采样'] + ['优化前', '优化后'], ncol=2, fontsize='xx-small')
        # if mode == 'cluster':
        #     ax.set_ylim(0, 100)
        # else:
        #     ax.set_ylim(0, 2000)
        ax.set_ylabel('每批次训练时间 (毫秒)', fontsize=base_size+2)
        ax.set_xlabel('数据集', fontsize=base_size+2)
        fig.savefig(f'out_thesis_figs/exp_batch_models_{mode}_batch_{data}.png')
        plt.close()



for i, mode in enumerate(['cluster']):
    df = {}
    for data in datasets:
        df[data] = {'Baseline': [], 'Optimize': [], 'real_ratio': [], 'exp_ratio': [], 'r1': [], 'y': [], 'z': []}

    for data in datasets:
        df_data = pd.read_csv(f'out_batch_csv/models_{mode}_{data}.csv', index_col=0)
        for alg in algs:
            df[data]['Baseline'].append(float(df_data['baseline'][alg]))
            df[data]['Optimize'].append(float(df_data['opt'][alg]))
            df[data]['real_ratio'].append(float(df_data['real_ratio'][alg]))
            df[data]['exp_ratio'].append(float(df_data['exp_ratio'][alg]))
            df[data]['r1'].append(float(df_data['r1'][alg]))
            df[data]['y'].append(100 * float(df_data['y'][alg]))
            df[data]['z'].append(100 * float(df_data['z'][alg]))

    # base_size = 12
    # for data in datasets:
    #     tab_data = df[data]

    #     xs = [algs_maps[alg] for alg in algs]
        
    #     x = np.arange(len(xs))  # the label locations
    #     width = 0.35  # the width of the bars

    #     fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
    #     ax.set_title(datasets_maps[data], fontsize=base_size+2)
    #     ax.set_ylabel('50批次训练时间 (秒)', fontsize=base_size+2)
    #     ax.set_xlabel('算法', fontsize=base_size+2)
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(xs, fontsize=base_size+2)
    #     ax.bar(x - width/2, tab_data['Baseline'], width, color=colors[0], edgecolor='black', label='优化前')
    #     ax.bar(x + width/2, tab_data['Optimize'], width, color=colors[1], edgecolor='black', label='优化后')
    #     ax.legend(ncol=2, fontsize='x-small')
    #     fig.savefig(f'out_thesis_figs/exp_batch_models_{mode}_total_{data}.png')

    base_size = 14
    for data in datasets:
        tab_data = df[data]

        xs = [algs_maps[alg] for alg in algs]
        x = np.arange(len(xs))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
        ax.set_title(datasets_maps[data], fontsize=base_size+2)
        ax.set_ylabel('比值', fontsize=base_size+2)
        ax.set_xlabel('算法', fontsize=base_size+2)
        line1, = ax.plot(x, tab_data['exp_ratio'], 'ob', label='预期加速比', linestyle='-')
        line2, = ax.plot(x, tab_data['real_ratio'], 'Dg', label='实际加速比', linestyle='-')
        line3, = ax.plot(x, tab_data['r1'], '^r', label='优化效果', linestyle='-')
        
        ax2 = ax.twinx()
        ax2.set_ylabel("耗时比例 （百分比)", fontsize=base_size + 2)
        line4, = ax2.plot(x, tab_data['y'], 's--', color='black', label='采样耗时比例' + r"$Y$" )
        line5, = ax2.plot(x, tab_data['z'], 'd--', color='black', label='数据传输耗时比例' + r"$Z$" )
        plt.legend(handles=[line1, line2, line3, line4, line5], ncol=2, fontsize='x-small')
        plt.xticks(ticks=x, labels=xs, fontsize=base_size)
        plt.yticks(fontsize=base_size)
        fig.tight_layout() # 防止重叠

        fig.savefig(f'out_thesis_figs/exp_batch_models_{mode}_else_{data}.png')
