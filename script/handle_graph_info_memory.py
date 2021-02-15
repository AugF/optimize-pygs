import os
import re
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
modes = ['cluster', 'graphsage']
datasets = ['graph_10k_50', 'amazon-computers', 'flickr', 'amazon-photo', 'coauthor-physics', 'com-amazon', 'pubmed']
algorithms = ['gcn', 'ggnn', 'gat', 'gaan']

percents = [1, 3, 6, 10, 25, 50]
cluster_batchs = [15, 45, 90, 150, 375, 750]
graphsage_batchs = {
    'graph_10k_50': [100, 300, 600, 1000, 2500, 5000], 
    'amazon-photo': [77, 230, 459, 765, 1913, 3825],
    'pubmed': [198, 592, 1184, 1972, 4930, 9859],
    'amazon-computers': [138, 413, 826, 1376, 3438, 6876],
    'coauthor-physics': [345, 1035, 2070, 3450, 8624, 17247],
    'flickr': [893, 2678, 5355, 8925, 22313, 44625],
    'com-amazon': [3349, 10046, 20092, 33487, 83716, 167432]
}


os.chdir("/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs")
dir_path = "log"
dir_out = "res"

# 与数据集，算法和Batch Size应该都是无关的
for mode in modes:
    for data in datasets:
        fig, axes = plt.subplots(4, 6, figsize=(28, 21), tight_layout=True)
        k = 0
        for alg in algorithms:
            df = {}
            if mode == 'cluster':
                for i, cs in enumerate(cluster_batchs):
                    file_path = os.path.join(dir_path, '_'.join([mode, alg, data, str(cs)]) + '.log')
                    df[percents[i]] = {}
                    df[percents[i]]['nodes'], df[percents[i]]['edges'], df[percents[i]]['memory'] = [], [], []
                    with open(file_path) as f:
                        for line in f:
                            graph_match_line = re.match("nodes: (.*), edges: (.*)", line) 
                            max_memory_match_line = re.match("max memory\(bytes\):  (.*)", line)
                            if graph_match_line:
                                nodes, edges = int(graph_match_line.group(1)), int(graph_match_line.group(2))
                                df[percents[i]]['nodes'].append(nodes)
                                df[percents[i]]['edges'].append(edges)
                            if max_memory_match_line:
                                max_memory = int(max_memory_match_line.group(1)) / (1024 * 1024) # MB
                                df[percents[i]]['memory'].append(max_memory)
            else:
                for i, gs in enumerate(graphsage_batchs[data]):
                    file_path = os.path.join(dir_path, '_'.join([mode, alg, data, str(gs)]) + '.log')
                    df[percents[i]] = {}
                    df[percents[i]]['nodes'], df[percents[i]]['edges'], df[percents[i]]['memory'] = [], [], []
                    with open(file_path) as f:
                        for line in f:
                            graph_match_line = re.match("nodes: (.*), edges: (.*)", line) 
                            max_memory_match_line = re.match("max memory\(bytes\):  (.*)", line)
                            if graph_match_line:
                                nodes, edges = int(graph_match_line.group(1)), int(graph_match_line.group(2))
                                df[percents[i]]['nodes'].append(nodes)
                                df[percents[i]]['edges'].append(edges)
                            if max_memory_match_line:
                                max_memory = int(max_memory_match_line.group(1)) / 1024  # KB
                                df[percents[i]]['memory'].append(max_memory)
            colors = "rgb"
            for i, per in enumerate(percents):
                print(mode, data, alg, per)
                print(df[per])
                try:
                    df_per = pd.DataFrame(df[per]) # 如果不能运行则跳过
                except ValueError:
                    continue
                ax = axes[k][i]
                ax.set_title(f"{mode}_{data}_{alg}_{per}")
                ax.set_xlabel("Batch")
                ax.set_ylabel("Number")
                for j, c in enumerate(df_per.columns[:-1]):
                    ax.plot(df_per.index, df_per[c], label=c, color=colors[j])
                ax_twins = ax.twinx()
                c = df_per.columns[-1]
                ax_twins.plot(df_per.index, df_per[c], label=c, color=colors[-1])
                ax_twins.set_ylabel("KB")
                ax_twins.legend()
                ax.legend()
            k += 1
        fig.savefig(dir_out + "/" + mode + "_" + data + ".png")
        plt.close()