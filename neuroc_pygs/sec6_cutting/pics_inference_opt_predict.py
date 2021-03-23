import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Patch
from matplotlib.font_manager import _rebuild
_rebuild() 

base_size = 14
# plt.style.use("grayscale")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams["font.size"] = base_size

dir_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec6_cutting'

df = pd.read_csv(dir_path + f'/exp_opt_res/opt_res.txt')
df.index = [f"{df['Name'][i]}_{df['Method'][i]}_{df['BatchSize'][i]}" for i in range(len(df.index))]
del df['Name'], df['Method'], df['BatchSize']

filenames = ['SAGE Reddit', 'ClusterGCN Ogbn-Products']
methods = ['baseline', 'random', 'degree3', 'pr4']
batch_sizes = {
    'SAGE Reddit': [8700, 8800, 8900],
    'ClusterGCN Ogbn-Products': [9000, 9100, 9200]
}
labels = ['随机剪枝', '度数剪枝', 'PageRank剪枝']
markers = 'oD^sdp'
linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (5, 5))]

for file in filenames:
    dd = defaultdict(list)
    xs = batch_sizes[file]
    baselines = []
    for method in methods:
        for bs in xs:
            acc = float(df['Acc'][file + '_' + method + '_' + str(bs)])
            if method == 'baseline':
                baselines.append(acc)
            else:
                dd[method].append(acc)
    dd = pd.DataFrame(dd)
    
    x = np.arange(len(xs))
    fig, ax = plt.subplots(figsize=(7/1.5 ,5/1.5), tight_layout=True)
    for j, c in enumerate(dd.columns):
        ax.plot(dd.index, dd[c], label=labels[j], marker=markers[j], markersize=8, linestyle=linestyles[j])
    ax.plot(dd.index, baselines, label='基准线', linestyle=(0, (5, 1)), linewidth=2, color='blue')
    ax.set_title(file, fontsize=base_size + 2)
    ax.set_xticks(x)
    ax.set_xticklabels(xs, fontsize=base_size + 2)
    ax.set_ylabel('精度', fontsize=base_size+2)
    ax.set_xlabel('批规模', fontsize=base_size+2)
    ax.legend(fontsize='small', ncol=2)
    fig.savefig(dir_path + f'/exp_figs/exp_memory_inference_motivation_{file.lower()}_acc.png')

