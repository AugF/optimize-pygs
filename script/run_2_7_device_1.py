import os 

datasets = ['graph_10k_50', 'amazon-computers', 'flickr', 'amazon-photo', 'coauthor-physics', 'com-amazon', 'pubmed']
algs = ['gcn', 'ggnn', 'gat', 'gaan']

os.chdir("/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs")
print(os.getcwd())
# get memory graph info
sh_str = "python -m code.optimize_sample.optimize_cluster --mode cluster --device cuda:2 --dataset"
for alg in algs:
    for data in datasets:
        os.system(sh_str + " " + data + " --model " + alg + f" 1>log/optimize_cluster_{alg}_{data}.log 2>&1")
