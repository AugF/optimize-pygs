import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

def float_x(x):
    return [float(i) for i in x]

plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

root_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time'
real_path = root_path + '/exp_res/batch/batch_size_constrast.log'
headers = None
mydata = []
with open(real_path) as f:
    for line in f.readlines():
        if headers == None:
            headers = [l.strip() for l in line.split('|')][1:]
        else:
            mydata.append([l.strip() for l in line.split('|')][1:])

df = pd.DataFrame(mydata, columns=headers)
df.index = df['Name']
del df['Name']
print(df)


file_names = ['amazon-computers_gcn', 'flickr_gat']
titles = ['GCN Amazon-computers', 'GAT Flickr']
xs = [0.01, 0.03, 0.06, 0.1, 0.25, 0.5]
modes = ['graphsage', 'cluster']
MODES = ['GraphSAGE Sampler', 'Cluster Sampler']

for k, mode in enumerate(modes):
    for file in file_names:
        base_times, opt_times, base_error, opt_error = [], [], [], []
        for v in xs:
            index = f'{file}_{v}_{mode}_pin_memory_False_num_workers_0_non_blocking_False'
            tmp_data = df.loc[index]
            base_times.append(float_x([tmp_data['Base Sampling'], tmp_data['Base Transfer'], tmp_data['Base Training']]))
            opt_times.append(float_x([tmp_data['Opt Sampling'], tmp_data['Opt Transfer'], tmp_data['Opt Training']]))
            base_error.append(float_x([tmp_data['Base max'], tmp_data['Base min']]))
            opt_error.append(float_x([tmp_data['Opt max'], tmp_data['Opt min']]))
        
        base_times, opt_times, base_error, opt_error = np.cumsum(np.array(base_times).T, axis=0) * 1000, np.cumsum(np.array(opt_times).T, axis=0)*1000, np.array(base_error).T * 1000, np.array(opt_error).T * 1000 # 单位ms

        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        xticklabels = [f'{int(100 * i)}%' for i in xs]
        x = np.arange(len(xticklabels))

        locations = [-1, 1]
        colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, 4))
        # colors = ['blue', 'cyan']
        width = 0.35
        errors_bar = [base_error, opt_error]
        for i, times in enumerate([base_times, opt_times]):
            ax.bar(x + locations[i] * width / 2, times[0], width, color=colors[i], edgecolor='black', hatch="///")
            ax.bar(x + locations[i] * width / 2, times[1], width, color=colors[i], edgecolor='black', bottom=times[0], hatch='...')
            ax.bar(x + locations[i] * width / 2, times[2], width, yerr=[errors_bar[i][1], errors_bar[i][0]], color=colors[i], edgecolor='black', bottom=times[1], hatch='xxx')
            # 待做: error bar
        ax.set_title(MODES[k])
        ax.set_xticklabels([''] + xticklabels)

        legend_colors = [Line2D([0], [0], color=c, lw=4) for c in colors]
        legend_hatchs = [Patch(facecolor='white', edgecolor='r', hatch='xxxx'), Patch(facecolor='white',edgecolor='r', hatch='....'), Patch(facecolor='white', edgecolor='r', hatch='////')]
        ax.legend(legend_hatchs + legend_colors, ['Training', 'Data Transferring', 'Sampling'] + ['Baseline', 'Optimize'], ncol=1, loc='upper left')
        ax.set_ylabel('Training Time Per Batch (ms)')
        ax.set_xlabel('Relative Batch Size')
        fig.savefig(root_path + f'/exp_figs/exp_batch_batch_size_{file}_{mode}.png')
        plt.close()

