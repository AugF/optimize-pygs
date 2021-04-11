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


xs = [10, 20, 50, 80, 100, 140]

datasets_maps = {
    'pub': 'pubmed', 'amc': 'amazon-computers', 'fli': 'flickr', 'red': 'reddit'
}
algs_maps = {
    'gcn': 'GCN', 'ggnn': 'GGNN', 'gat': 'GAT', 'gaan': 'GaAN'
}
file_names = ['GCN amazon-computers', 'GaAN amazon-computers']
files = ['gcn', 'gaan']

for mode in ['cluster', 'graphsage']:
    df_data = pd.read_csv(f'out_batch_csv/batch_N.csv', index_col=0)
    for k, file in enumerate(files):
        df = {'base_times': [], 'opt_times': [], 'baseline': [], 'opt': [], 
            'Baseline': [], 'Optimize': [], 'real_ratio': [], 'exp_ratio': [], 'r1': [], 'y': [], 'z': []}
        for ds in xs:
            tmp_data = df_data.loc[f'{file}_{mode}_{ds}']
            df['base_times'].append(float_x([tmp_data['base_sample'], tmp_data['base_move'], tmp_data['base_cal']]))
            df['opt_times'].append(float_x([tmp_data['opt_sample'], tmp_data['opt_move'], tmp_data['opt_cal']]))
            df['Baseline'].append(float(tmp_data['baseline']))
            df['Optimize'].append(float(tmp_data['opt']))
            df['real_ratio'].append(float(tmp_data['real_ratio']))
            df['exp_ratio'].append(float(tmp_data['exp_ratio']))
            df['r1'].append(float(tmp_data['r1']))
            df['y'].append(100 * float(tmp_data['y']))
            df['z'].append(100 * float(tmp_data['z']))

        # fig1
        fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
        x = np.arange(len(xs))
        locations = [-1, 1]
        colors = plt.get_cmap('Greys')(np.linspace(0.15, 0.75, 2))
        colors = [colors[-1], colors[0]]
        width = 0.35
        base_times, opt_times = np.cumsum(np.array(df['base_times']).T, axis=0) * 1000, np.cumsum(np.array(df['opt_times']).T, axis=0)*1000
        for i, times in enumerate([base_times, opt_times]):
            ax.bar(x + locations[i] * width / 2, times[0], width, color=colors[i], edgecolor='black', hatch="///")
            ax.bar(x + locations[i] * width / 2, times[1] - times[0], width, color=colors[i], edgecolor='black', bottom=times[0], hatch='...')
            ax.bar(x + locations[i] * width / 2, times[2] - times[1], width, color=colors[i], edgecolor='black', bottom=times[1], hatch='xxx')
            # 待做: error bar

        ax.set_title(file_names[k], fontsize=base_size+2)
        legend_colors = [Patch(facecolor=c, edgecolor='black') for c in colors]
        legend_hatchs = [Patch(facecolor='white', edgecolor='black', hatch='xxxx'), Patch(facecolor='white',edgecolor='black', hatch='....'), Patch(facecolor='white', edgecolor='black', hatch='////')]
        ax.legend(legend_hatchs + legend_colors, ['GPU计算', '数据传输', '采样'] + ['优化前', '优化后'], ncol=2,  fontsize='xx-small')

        ax.set_ylabel('每批次训练时间 (毫秒)', fontsize=base_size+2)
        ax.set_xlabel('边数', fontsize=base_size-2)
        ax.set_xticks(x)
        ax.set_xticklabels(xs, fontsize=base_size)
        fig.savefig(f'out_batch_figs/exp_batch_N_{mode}_batch_{file}.png')

        base_size = 10
        # fig2
        fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
        ax.set_title(file_names[k], fontsize=base_size+2)
        ax.set_ylabel('50批训练时间 (秒)', fontsize=base_size+2)
        ax.set_xlabel('边数', fontsize=base_size+2)
        ax.set_xticks(x)
        ax.set_xticklabels(xs, fontsize=base_size)
        ax.bar(x - width/2, df['Baseline'], width, color=colors[0], edgecolor='black', label='优化前')
        ax.bar(x + width/2, df['Optimize'], width, color=colors[1], edgecolor='black', label='优化后')
        ax.legend(loc='lower left', ncol=2, fontsize='small')
        fig.savefig(f'out_batch_figs/exp_batch_N_{mode}_total_{file}.png')

        # fig3
        fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
        ax.set_title(file_names[k], fontsize=base_size+2)
        ax.set_ylabel('比值', fontsize=base_size+2)
        ax.set_xlabel('边数', fontsize=base_size+2)
        line1, = ax.plot(x, df['exp_ratio'], 'ob', label='预期加速比', linestyle='-')
        line2, = ax.plot(x, df['real_ratio'], 'Dg', label='实际加速比', linestyle='-')
        line3, = ax.plot(x, df['r1'], '^', label='优化效果', linestyle='-')
        
        ax2 = ax.twinx()
        ax2.set_ylabel("评估耗时占比" + r"$X$" + " (%)", fontsize=base_size + 2)
        line4, = ax2.plot(x, df['y'], 'rs--', label='采样耗时占比' + r"$Y$" )
        line5, = ax2.plot(x, df['z'], 'rd--', label='数据传输耗时占比' + r"$Z$" )
        plt.legend(handles=[line1, line2, line3, line4, line5], fontsize='xx-small')
        plt.xticks(ticks=x, labels=xs, fontsize=base_size)
        plt.yticks(fontsize=base_size)
        fig.tight_layout() # 防止重叠

        fig.savefig(f'out_batch_figs/exp_batch_N_{mode}_else_{file}.png')

