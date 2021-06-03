import re
import pandas as pd
from collections import defaultdict

dir_path = 'out_epoch_log'
dir_out = 'out_epoch_csv'
def read_file(file_name):
    dd_data = defaultdict(list)
    exp_ratio_re = r'Average.*, x: (.*), exp_ratio: (.*)'
    opt_re = r'model: (.*), dataset: (.*), baseline: (.*), opt:(.*), ratio: (.*)'
    ratio_cnt, opt_cnt = 0, 0
    with open(dir_path + '/' + file_name) as f:
        for line in f:
            ratio_line = re.match(exp_ratio_re, line)
            opt_line = re.match(opt_re, line)
            if ratio_line:
                dd_data[ratio_cnt].extend([float(ratio_line.group(1)), float(ratio_line.group(2))])
                ratio_cnt += 1
            if opt_line:
                res = [float(opt_line.group(3)), float(opt_line.group(4)), float(opt_line.group(5))]
                dd_data[opt_cnt].extend(res)
                opt_cnt += 1
    return dd_data


index = {
    'amazon-computers_graph_gcn_2048_100k.log': {
        'model': 'gcn_2048',
        'data': 'random_100k_',
        'vars': [5, 10, 20, 40, 80, 85, 90, 95, 100]
    },
    'amazon-computers_graph_gcn_2048_100k_2.log': { # 
        'model': 'gcn_2048',
        'data': 'random_100k_',
        'vars': [200, 300, 400, 450, 500, 525, 550, 575, 600]
    },
    'amazon-computers_graph_gaan_100k_1024.log': { # 
        'model': 'gaan_1024',
        'vars': [5, 10, 20, 50, 100, 150, 200]
    },
    'com-amazon.log': {
        'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    },
    'pubmed.log': {
        'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    },
    'amazon-computers.log': {
        'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    },
    'flickr.log': {
        'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    },
    'amazon-computers_batch_size.log': { # 超参数
        'model': 'gcn',
        'vars': [64, 128, 256, 512, 1024, 2048]
    },
    'amazon-computers_batch_size_2.log': { # 超参数
        'model': 'gcn',
        'vars': [2304, 2560, 2816, 2944, 3072, 3200, 3328]
    },
    'amazon-computers_gaan_batch_size.log': { # per
        'model': 'gaan',
        'vars': [8, 16, 32, 64, 128, 256, 512, 768, 896, 1024, 1088]
    },
    'gcn_pubmed_N.log':{
        'model': 'gcn_amazon-computers',
        'vars': [10, 20, 40, 100, 250, 500, 1000, 2000, 4000, 10000]
    },
    'gaan_amazon_computers_N.log':{
        'model': 'gaan_64_amazon-computers',
        'vars': [10, 20, 50, 80, 100, 200]
    },
    'gaan_amazon-computers_eval_per.log':{
        'model': 'gaan_64_amazon-computers',
        'vars': [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    }
}


for key in index.keys():
# for key in ['pubmed.log', 'amazon-computers.log', 'flickr.log', 'com-amazon.log']:
    if 'N' in key:
    # if True:
        print(key)
        dd_data = read_file(key)
        dd_data = pd.DataFrame(dd_data.values(), index=index[key]['vars'], columns=['x', 'exp_ratio', 'baseline', 'opt', 'real_ratio'])
        dd_data['r1'] = dd_data['real_ratio'] / dd_data['exp_ratio']
        dd_data['max(x,1-x)'] = [max(dd_data['x'][i], 1- dd_data['x'][i]) for i in dd_data.index]
        res = dd_data.reindex(columns=['baseline', 'opt', 'x', 'max(x,1-x)', 'exp_ratio', 'real_ratio', 'r1'])
        res.to_csv(dir_out + f'/{key[:-4]}.csv')
    