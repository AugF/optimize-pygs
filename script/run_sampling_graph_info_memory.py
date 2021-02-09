import os
import time

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
for mode in modes:
    sh_commands = []
    for data in datasets:
        for alg in algorithms:
            if mode == 'cluster':
                cmd = "python -m code.main_sampling_graph_info --device cuda:1 --mode {} --model {} --dataset {} --batch_partitions {} >>{} 2>&1"
                for cs in cluster_batchs:
                    file_path = os.path.join(dir_path, '_'.join([mode, alg, data, str(cs)]) + '.log')
                    if os.path.exists(file_path):
                        continue
                    sh_commands.append(cmd.format(mode, alg, data, str(cs), file_path))
            else:
                cmd = "python -m code.main_sampling_graph_info --device cuda:2 --mode {} --model {} --data {} --epochs 50 --batch_size {} >>{} 2>&1"
                for gs in graphsage_batchs[data]:
                    file_path = os.path.join(dir_path, '_'.join([mode, alg, data, str(gs)]) + '.log')
                    if os.path.exists(file_path):
                        continue
                    sh_commands.append(cmd.format(mode, alg, data, str(gs), file_path))
                    
    with open("sh_" + mode + "_graph_info_memory.sh", "w") as f:
        for sh in sh_commands:
            f.write(sh + '\n')