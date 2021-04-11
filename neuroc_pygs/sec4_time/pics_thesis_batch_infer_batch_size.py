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


xs = [1024, 2048, 4096, 8192]

datasets_maps = {
    'pub': 'pubmed', 'amc': 'amazon-computers', 'fli': 'flickr', 'red': 'reddit'
}
algs_maps = {
    'gcn': 'GCN', 'ggnn': 'GGNN', 'gat': 'GAT', 'gaan': 'GaAN'
}
file_names = ['GaAN amazon-computers']
files = ['gaan_amazon-computers']


ts = [8, 4, 2, 1]
df_data = pd.read_csv(f'out_batch_csv/batches_infer.csv', index_col=0)
if True:
    for k, file in enumerate(files):
        df = {'base_times': [], 'opt_times': [], 'baseline': [], 'opt': [], 
            'Baseline': [], 'Optimize': [], 'real_ratio': [], 'exp_ratio': [], 'r1': [], 'y': [], 'z': []}
        for t, ds in enumerate(xs):
            tmp_data = df_data.loc[f'{file}_{ds}']
            df['base_times'].append(float_x([tmp_data['base_sample']/ts[t], tmp_data['base_move']/ts[t], tmp_data['base_cal']/ts[t]]))
            df['opt_times'].append(float_x([tmp_data['opt_sample']/ts[t], tmp_data['opt_move']/ts[t], tmp_data['opt_cal']/ts[t]]))
            df['Baseline'].append(float(tmp_data['baseline']))
            df['Optimize'].append(float(tmp_data['opt']))
            df['real_ratio'].append(float(tmp_data['real_ratio']))
            df['exp_ratio'].append(float(tmp_data['exp_ratio']))
            df['r1'].append(float(tmp_data['r1']))
            df['y'].append(100 * float(tmp_data['y']))
            df['z'].append(100 * float(tmp_data['z']))

        base_size = 12
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
        ax.legend(legend_hatchs + legend_colors, ['GPU计算', '数据传输', '采样'] + ['优化前', '优化后'], ncol=2)

        ax.set_ylabel('每批次推理时间 (毫秒)', fontsize=base_size+2)
        ax.set_xlabel('相对批规模', fontsize=base_size+2)
        ax.set_xticks(x)
        ax.set_xticklabels(xs, fontsize=base_size)
        fig.savefig(f'out_thesis_figs/exp_batch_infer_batch_size_batch_{file}.png')

        base_size = 14
        # fig2
        fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
        ax.set_title(file_names[k], fontsize=base_size+2)
        ax.set_ylabel('50批推理时间 (秒)', fontsize=base_size+2)
        ax.set_xlabel('相对批规模', fontsize=base_size+2)
        ax.set_xticks(x)
        ax.set_xticklabels(xs, fontsize=base_size)
        ax.bar(x - width/2, df['Baseline'], width, color=colors[0], edgecolor='black', label='优化前')
        ax.bar(x + width/2, df['Optimize'], width, color=colors[1], edgecolor='black', label='优化后')
        ax.legend()
        fig.savefig(f'out_thesis_figs/exp_batch_batch_size_total_{file}.png')

        # fig3
        fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
        ax.set_title(file_names[k], fontsize=base_size+2)
        ax.set_ylabel('比值', fontsize=base_size+2)
        ax.set_xlabel('相对批规模大小', fontsize=base_size+2)
        line1, = ax.plot(x, df['exp_ratio'], 'ob', label='预期加速比', linestyle='-')
        line2, = ax.plot(x, df['real_ratio'], 'Dg', label='实际加速比', linestyle='-')
        line3, = ax.plot(x, df['r1'], '^r', label='优化效果', linestyle='-')
        
        ax2 = ax.twinx()
        ax2.set_ylabel("耗时比例 （百分比)", fontsize=base_size + 2)
        line4, = ax2.plot(x, df['y'], 's--', color='black', label='采样耗时比例' + r"$Y$" )
        line5, = ax2.plot(x, df['z'], 'd--', color='black', label='数据传输耗时比例' + r"$Z$" )
        plt.legend(handles=[line1, line2, line3, line4, line5], fontsize='x-small')
        plt.xticks(ticks=x, labels=xs, fontsize=base_size)
        plt.yticks(fontsize=base_size)
        fig.tight_layout() # 防止重叠

        fig.savefig(f'out_thesis_figs/exp_batch_infer_batch_size_else_{file}.png')

