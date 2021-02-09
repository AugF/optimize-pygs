import os
import re

datasets = ['graph_10k_50', 'amazon-computers', 'flickr', 'amazon-photo', 'coauthor-physics', 'com-amazon', 'pubmed']
algs = ['gcn', 'ggnn', 'gat', 'gaan']
modes = ['cluster', 'graphsage']

dir_path = "/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/log"

for mode in modes:
    df = {}
    for data in datasets:
        file_path = dir_path + f"/graph_info_{mode}_gcn_{data}.log"
        df[data] = {}
        df[data]['graph'], df[data]['memory'] = [], []
        with open(file_path) as f:
            for line in f:
                graph_match_line = re.match("nodes: (.*), edges: (.*)", line) 
                max_memory_match_line = re.match("max memory\(bytes\):  (.*)", line)
                if graph_match_line:
                    nodes, edges = int(graph_match_line.group(1)), int(graph_match_line.group(2))
                    df[data]['graph'].append((nodes, edges))
                if max_memory_match_line:
                    # MB
                    max_memory = int(max_memory_match_line.group(1)) / (1024 * 1024)
                    df[data]['memory'].append(max_memory)
    
    # fig1: 对于不同数据集所得到的结果
    # fig2: 对于不同Batch Size的不同采样方法所得到的不同结果