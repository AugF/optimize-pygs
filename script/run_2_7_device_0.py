import os 

datasets = ['graph_10k_50', 'amazon-computers', 'flickr', 'amazon-photo', 'coauthor-physics', 'com-amazon', 'pubmed']
algs = ['gcn', 'ggnn', 'gat', 'gaan']
modes = ['cluster', 'graphsage']

os.chdir("/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs")
print(os.getcwd())

# fig1
sh_str = "python -m code.main_sampling_graph_info --model gcn --device cuda:1 --dataset"
for mode in modes:
    for data in datasets:
        os.system(sh_str + " " + data + " --mode " + mode + f" 1>log/graph_info_{mode}_gcn_{data}.log 2>&1")

