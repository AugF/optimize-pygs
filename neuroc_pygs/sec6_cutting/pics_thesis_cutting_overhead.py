import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.font_manager import _rebuild
_rebuild()
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimHei'],
}

base_size = 14
# plt.style.use("grayscale")

plt.rcParams["font.size"] = base_size

names = ['随机剪枝', '度数剪枝', 'PageRank剪枝']

fix_edges = {
    0.01: [0.022017478942871094, 1.7769372463226318, 13.826231956481934],
    0.03: [0.014621496200561523, 1.796976089477539, 13.806233882904053],
    0.06: [0.015668392181396484, 1.7926790714263916, 13.838400840759277],
    0.1: [0.01675581932067871, 1.7738869190216064, 13.793803215026855],
    0.2: [0.02049398422241211, 1.7696924209594727, 13.876492023468018],
    0.5: [0.028832674026489258, 1.7719838619232178, 13.816819906234741]
}

fix_cutting_nums = {
    10: [0.004849672317504883, 0.17540454864501953, 0.2069997787475586],
    15: [0.004189491271972656, 0.25641298294067383, 0.43677711486816406],
    30: [0.006650447845458984, 0.5153770446777344, 1.976743459701538],
    50: [0.009080886840820312, 0.8558712005615234, 4.821713209152222],
    70: [0.01281118392944336, 1.1991987228393555, 8.677909135818481],
    90: [0.015047788619995117, 1.5526838302612305, 11.347145557403564]
}

markers = 'oD^sdp'
linestyles = ['solid', 'dotted', 'dashed',
              'dashdot', (0, (5, 5)), (0, (3, 1, 1, 1))]

# pics fix edges
fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
x = list(fix_edges.keys())
arr1 = np.array(list(fix_edges.values())).T
labels = [0.01, 0.1, 0.2, 0.5]

for i in range(3):
    ax.plot(x, arr1[i], marker=markers[i],
            linestyle=linestyles[i], label=names[i])
ax.set_xlabel('相对剪枝比例', fontsize=base_size + 2)
ax.set_ylabel('耗时 (s)', fontsize=base_size + 2)
ax.legend(fontsize='small', loc='center')
ax.set_xticks(labels)
ax.set_xticklabels([f'{int(100*i)}%' for i in labels])
fig.savefig(
    f'exp6_thesis_figs/exp_memory_inference_cutting_overhead_fix_edges.png', dpi=400)

# pics fix cutting nums
fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
x = list(fix_cutting_nums.keys())
arr1 = np.array(list(fix_cutting_nums.values())).T
labels = [10, 30, 50, 70, 90]

for i in range(3):
    ax.plot(x, arr1[i], marker=markers[i],
            linestyle=linestyles[i], label=names[i])
ax.set_xlabel('边数', fontsize=base_size + 2)
ax.set_ylabel('耗时 (s)', fontsize=base_size + 2)
ax.legend(fontsize='small')
ax.set_xticks(labels)
ax.set_xticklabels([f'{i}k' for i in labels])
fig.savefig(f'exp6_thesis_figs/exp_memory_inference_cutting_overhead_fix_cutting_nums.png', dpi=400)
