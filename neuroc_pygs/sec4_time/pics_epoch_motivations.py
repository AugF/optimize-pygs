import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

root_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/exp_figs'
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Training', 'Evaluation'
xs = [3.03089526, 5.36482068]
sizes = [100 * i / sum(xs) for i in xs]
# sizes = [15, 30, 45, 10]
explode = (0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

fig1.savefig(root_path + '/exp_epoch_motivation_sampling.png')