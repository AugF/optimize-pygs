import re
import pandas as pd
from collections import defaultdict

dir_path = 'out_epoch_sampling_log'
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
    'cluster_batch_size.log': {
        'vars': [0.01, 0.03, 0.06, 0.1, 0.25, 0.5]
    },
    'graphsage_batch_size.log': {
        'vars': [0.01, 0.03, 0.06, 0.1, 0.25, 0.5]
    },
    'cluster_gat_flickr_batch_size.log': {
        'vars': [0.01, 0.03, 0.06, 0.1, 0.25, 0.5]
    },
    'graphsage_gat_flickr_batch_size.log': {
        'vars': [0.01, 0.03, 0.06, 0.1, 0.25, 0.5]
    },
    'model_datasets.log': {
        'vars': ['gcn_pubmed', 'gcn_amazon-computers', 'gcn_flickr', 'ggnn_pubmed', 'gat_pubmed', 'gaan_pubmed']

    },
    'model_datasets_v2.log': {
        'vars': ['gcn_coauthor-physics', 'gat_amazon-computers', 'gat_coauthor-physics', 'gat_flickr', 'gaan_flickr', 'ggnn_flickr']
    },
    'model_hidden_size.log': {
        'props': 'gcn_gaan_amazon-computers',
        'vars': [f'gcn_{i}' for i in [64, 128, 256, 512, 1024, 2048]] + [f'gaan_{i}' for i in [64, 128, 256, 512, 1024, 2048]] 
    },
    'model_hidden_size.log': {
        'props': 'gcn_gaan_amazon-computers',
        'vars': [f'gcn_{i}' for i in [64, 128, 256, 512, 1024, 2048]] + [f'gaan_{i}' for i in [64, 128, 256, 512, 1024, 2048]] 
    },
    'gcn_amazon-computers_N.log': {
        'vars': [10, 20, 50, 80, 100, 200]
    },
    'gaan_amazon-computers_N.log': {
        'vars': [10, 20, 50, 80, 100, 200]
    },
    'graphsage_models.log': {
        'vars': ['gcn_pubmed', 'ggnn_pubmed', 'gat_pubmed', 'gaan_pubmed'] +
         ['gcn_amazon-computers', 'ggnn_amazon-computers', 'gat_amazon-computers', 'gaan_amazon-computers']
    },
    'gcn_datasets.log': {
        'vars': ['pubmed', 'amazon-computers', 'flickr', 'com-amazon']
    },
    'gaan_datasets.log': {
        'vars': ['pubmed', 'amazon-computers', 'flickr', 'com-amazon']
    }
}

for key in index.keys():
    # for key in ['pubmed.log', 'amazon-computers.log', 'flickr.log', 'com-amazon.log']:
    if 'graphsage_models' in key:
    # if True:
        print(key)
        dd_data = read_file(key)
        print(dd_data)
        dd_data = pd.DataFrame(dd_data.values(), index=index[key]['vars'], columns=['x', 'exp_ratio', 'baseline', 'opt', 'real_ratio'])
        dd_data['r1'] = dd_data['real_ratio'] / dd_data['exp_ratio']
        dd_data['max(x,1-x)'] = [max(dd_data['x'][i], 1- dd_data['x'][i]) for i in dd_data.index]
        res = dd_data.reindex(columns=['baseline', 'opt', 'x', 'max(x,1-x)', 'exp_ratio', 'real_ratio', 'r1'])
        res.to_csv(dir_out + f'/sampling_{key[:-4]}.csv')