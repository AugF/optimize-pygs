import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import _rebuild
_rebuild() 

# plt.style.use("grayscale")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams["font.size"] = 24

root_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time'
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = '训练', '评估'
full_batch_xs = [0.12717956, 0.05978919]
sampling_xs = [3.53070061, 6.20208459]

files = ['sampling', 'full']
titles = ['批训练', '全数据训练']
colors = plt.get_cmap('Greys')(np.linspace(0.15, 0.75, 2))
colors = [colors[-1], colors[0]]

for i, xs in enumerate([sampling_xs, full_batch_xs]):
	sizes = [100 * i / sum(xs) for i in xs]
	explode = (0, 0) 
	fig1, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
	ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
			shadow=True, startangle=90)
	ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
	ax.set_title(titles[i])
	fig1.savefig(root_path + f'/exp_figs_final/exp_epoch_motivation_{files[i]}.png')