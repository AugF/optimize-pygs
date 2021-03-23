import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Patch
from neuroc_pygs.sec4_time.utils import datasets_maps, algorithms, sampling_modes
from matplotlib.font_manager import _rebuild
_rebuild() 

def float_x(x):
    return [float(i) for i in x]

base_size = 12
# plt.style.use("grayscale")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams["font.size"] = base_size

root_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec6_cutting'
real_path = root_path + '/exp_res/cutting_method_acc.txt'
headers = None
mydata = []
with open(real_path) as f:
    for line in f.readlines():
        if headers == None:
            headers = [l.strip() for l in line.split('|')][1:]
        else:
            mydata.append([l.strip() for l in line.split('|')][1:])

df = pd.DataFrame(mydata, columns=headers)
df.index = [df['Model'][i] + '_' + df['Data'][i] + '_' + df['Method'][i] + '_' + df['Per'][i] for i in range(len(df.index))]
del df['Model'], df['Data'], df['Per'], df['Method']

# 对所有gcn, gat都画一遍
markers = 'oD^sdp'
linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (5, 5))]
xs = [0.01, 0.03, 0.06, 0.1, 0.2, 0.5]
labels = ['随机剪枝', '度数剪枝', 'PageRank剪枝']
for model in ['gcn', 'gat']:
    for data in ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics']:
        cur_name = model + '_' + data
        df_data = defaultdict(list)
        for method in ['none_full', 'random', 'degree3', 'pr4']:
            if method == 'none_full':
                index = cur_name + '_' + method
                if index not in df['Acc']:
                    continue
                full_acc = float(df['Acc'][index])
                print(f"{cur_name}, full acc: {full_acc}")
            else:
                for per in xs:
                    index = cur_name + '_' + method + '_' + str(per)
                    if index not in df['Acc']:
                        continue
                    df_data[method].append(df['Acc'][index])
        if len(df_data) == 0:
            continue
        df_data = pd.DataFrame(df_data, dtype=np.float32)
        min_val, max_val = np.min(df_data.values), np.max(df_data.values)
        step = (max_val - min_val) / 3

        df_data.to_csv(root_path + f'/exp_cutting_res/{model}_{data}.csv')
        fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
        ax.set_title(model.upper() + ' ' + data.capitalize(), fontsize=base_size+2)
        ax.set_xlabel('剪枝相对比例 (百分比)', fontsize=base_size+2)
        ax.set_ylabel('测试集精度', fontsize=base_size+2)
        ax.set_xticks(range(len(df_data.index)))
        ax.set_xticklabels([str(int(100 * x)) + '%' for x in xs], fontsize=base_size+2)
        for j, c in enumerate(df_data.columns):
            ax.plot(df_data.index, df_data[c], label=labels[j], marker=markers[j], linestyle=linestyles[j], markersize=8)
        ax.plot(df_data.index, len(df_data.index) * [full_acc], label='基准线', linestyle=(0, (5, 1)), linewidth=2, color='blue')
        ax.legend(fontsize='small')
        fig.savefig(root_path + f'/exp_cutting_figs/exp_memory_inference_cutting_methods_{cur_name}.png')