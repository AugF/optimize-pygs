import re
import numpy as np
import pandas as pd
from collections import defaultdict

dir_path = 'exp_batch_data'
dir_out = 'out_batch_csv'
columns = ['baseline', 'opt', 'real_ratio', 'base_sample', 'base_move', 'base_cal', 'opt_sample', 'opt_move', 'opt_cal', 'y', 'z', 'exp_ratio']

def read_file(file_name, dir_path='opt_batches'):
    dd_data = defaultdict(list)
    opt_re = r'model: (.*), data: (.*), baseline: (.*), opt:(.*), ratio: (.*)'
    base_time_re = r'Baseline: sample: (.*), move: (.*), cal: (.*)'
    opt_time_re = r'Opt: sample: (.*), move: (.*), cal: (.*)'
    exp_ratio_re = r'y: (.*), z: (.*), exp_ratio: (.*)'
    ratio_cnt, opt_cnt, base_time_cnt, opt_time_cnt = 0, 0, 0, 0
    with open(dir_path + '/' + file_name) as f:
        for line in f:
            ratio_line = re.match(exp_ratio_re, line)
            opt_line = re.match(opt_re, line)
            base_time_line = re.match(base_time_re, line)
            opt_time_line = re.match(opt_time_re, line)
            if ratio_line:
                dd_data[ratio_cnt].extend([float(ratio_line.group(1)), float(ratio_line.group(2)), float(ratio_line.group(3))])
                ratio_cnt += 1
            if opt_line:
                res = [float(opt_line.group(3)), float(opt_line.group(4)), float(opt_line.group(5))]
                dd_data[opt_cnt].extend(res)
                opt_cnt += 1
            if base_time_line:
                dd_data[base_time_cnt].extend([float(base_time_line.group(1)), float(base_time_line.group(2)), float(base_time_line.group(3))])
                base_time_cnt += 1
            if opt_time_line:
                res = [float(opt_time_line.group(1)), float(opt_time_line.group(2)), float(opt_time_line.group(3))]
                dd_data[opt_time_cnt].extend(res)
                opt_time_cnt += 1
    return dd_data


index = {
    'pubmed_cluster.log': {
        'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    },
    'amc_cluster.log': { #
        'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    },
    'fli_cluster.log': { 
        'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    },
    'reddit_cluster.log': { 
        'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    },
    # 'pubmed_cluster_v2.log': {
    #     'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    # },
    # 'amc_cluster_v2.log': { #
    #     'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    # },
    # 'fli_cluster_v2.log': { 
    #     'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    # },
    # 'reddit_cluster_v2.log': { 
    #     'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    # },
    'pubmed_graphsage_full.log': {
        'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    },
    # 'pubmed_graphsage_v2.log': {
    #     'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    # },
    # 'pubmed_graphsage_v3.log': {
    #     'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    # },
    'amc_sage_full.log': { # v2, not full to cuda
        'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    },
    # 'amc_sage_v2.log': { # v2
    #     'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    # },
    # 'amc_sage_v3.log': { # v2
    #     'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    # },
    'fli_sage_full.log': { # v2
        'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    },
    # 'fli_sage_v2.log': { # v2
    #     'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    # },
    # 'fli_sage_v3.log': { # v2
    #     'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    # },
    'reddit_sage_full.log': { 
        'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    },
    # 'reddit_sage_v2.log': { 
    #     'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    # },
    # 'reddit_sage_v3.log': { 
    #     'vars': ['gcn', 'ggnn', 'gat', 'gaan']
    # },
    'gcn_amc_cluster_ds.log': {
        'vars': [16, 64, 256, 1024, 10240, 102400, 256000, 512000]
    },
    'gaan_amc_cluster_ds.log': {
        'vars': [16, 64, 256, 1024, 10240, 25600, 51200, 102400, 204800]
    },
    'gcn_amc_sage_ds.log': {
        'vars': [16, 64, 256, 1024, 10240]
    },
    'gaan_amc_sage_ds.log': {
        'vars': [16, 64, 256, 1024, 10240, 25600]
    },
    'gcn_amc_sage_ds_full.log': {
        'vars': [16, 64, 256, 1024, 2560, 5120, 7680, 10240, 10880, 11520]
    },
    'gaan_amc_sage_ds_full.log': {
        'vars': [16, 64, 256, 1024, 10240, 25600]
    },
    'batches_graph.log': {
        'props': 'random100k_-k',
        'vars': [f'gcn_{i}' for i in [5, 10, 20, 50, 100, 150, 200]] + [f'gaan_{i}' for i in [5, 10, 20, 50, 100, 150, 200]] 
    },
    'batches_N.log': {
        'props': 'batch_size',
        'vars': [f'gcn_cluster_{i}' for i in [10, 20, 50, 80, 100, 140]] + [f'gcn_graphsage_{i}' for i in [10, 20, 50, 80, 100, 140]] 
        + [f'gaan_cluster_{i}' for i in [10, 20, 50, 80, 100, 140]] + [f'gaan_graphsage_{i}' for i in [10, 20, 50, 80, 100, 140]] 
    },
    'batches_rs_gcn_pubmed.log': {
        'vars': [f'gcn_pub_cluster_{i}' for i in [0.01, 0.03, 0.06, 0.1, 0.25, 0.5]] + [f'gcn_pub_graphsage_{i}' for i in [0.01, 0.03, 0.06, 0.1, 0.25, 0.5]]
    },
    'batches_ds_gcn_pubmed.log': {
        'vars': [f'gcn_pub_cluster_{i}' for i in [16, 64, 256, 1024, 10240, 25600]] + [f'gcn_pub_graphsage_{i}' for i in [16, 64, 256, 1024, 10240, 25600]]
    }
}

# sage.log: 使用全部增速
# sage_v2.log: 使用一半的增速
def run():
    for key in index.keys():
        if 'batches_ds_gcn_pubmed.log' in key:
            print(key)
            dd_data = read_file(key)
            dd_data = pd.DataFrame(dd_data.values(), index=index[key]['vars'], columns=columns)
            dd_data['1-y-z'] = [1 - dd_data['y'][i] - dd_data['z'][i] for i in dd_data.index]
            dd_data['r1'] = dd_data['real_ratio'] / dd_data['exp_ratio'] # 真实的比上期待的，越接近1表示越好
            dd_data['y1'] = [dd_data['opt_sample'][i] / (dd_data['opt_sample'][i] + dd_data['opt_move'][i] + dd_data['opt_cal'][i]) for i in dd_data.index]
            dd_data['z1'] = [dd_data['opt_move'][i] / (dd_data['opt_sample'][i] + dd_data['opt_move'][i] + dd_data['opt_cal'][i]) for i in dd_data.index]
            dd_data['1-y1-z1'] = [dd_data['opt_cal'][i] / (dd_data['opt_sample'][i] + dd_data['opt_move'][i] + dd_data['opt_cal'][i]) for i in dd_data.index]
            # dd_data['exp_real_ratio'] = [1/50 + 49 * max(y, 1-y)/50 for y in dd_data['y']]
            # dd_data['r2'] = dd_data['real_ratio'] / dd_data['exp_real_ratio']
            dd_data['max(y,z,1-y-z)'] = [max(1 - dd_data['y'][i] - dd_data['z'][i], max(dd_data['y'][i], dd_data['z'][i])) for i in dd_data.index]
            res = dd_data.reindex(columns=['baseline', 'opt', 'y', 'z', 'max(y,z,1-y-z)', 'exp_ratio', 'real_ratio', 'r1', 'base_sample', 'base_move', 'base_cal', 'opt_sample', 'opt_move', 'opt_cal', 'y', 'z', '1-y-z', 'y1', 'z1', '1-y1-z1'])
            res.to_csv(dir_out + f'/{key[:-4]}.csv')


def run_batch_size():
    index = [0.01, 0.03, 0.06, 0.1, 0.25, 0.5]

    for mode in ['cluster', 'sage']:
        for model in ['gcn', 'gaan']:
            key = model + '_amc_' + mode + '_batch_size.log'
            print(key)
            dd_data = read_file(key)
            dd_data = pd.DataFrame(dd_data.values(), index=index, columns=columns)
            dd_data['1-y-z'] = [1 - dd_data['y'][i] - dd_data['z'][i] for i in dd_data.index]
            dd_data['r1'] = dd_data['real_ratio'] / dd_data['exp_ratio'] # 真实的比上期待的，越接近1表示越好
            dd_data['y1'] = [dd_data['opt_sample'][i] / (dd_data['opt_sample'][i] + dd_data['opt_move'][i] + dd_data['opt_cal'][i]) for i in dd_data.index]
            dd_data['z1'] = [dd_data['opt_move'][i] / (dd_data['opt_sample'][i] + dd_data['opt_move'][i] + dd_data['opt_cal'][i]) for i in dd_data.index]
            dd_data['1-y1-z1'] = [dd_data['opt_cal'][i] / (dd_data['opt_sample'][i] + dd_data['opt_move'][i] + dd_data['opt_cal'][i]) for i in dd_data.index]
            # dd_data['exp_real_ratio'] = [1/50 + 49 * max(y, 1-y)/50 for y in dd_data['y']]
            # dd_data['r2'] = dd_data['real_ratio'] / dd_data['exp_real_ratio']
            dd_data['max(y,z,1-y-z)'] = [max(1 - dd_data['y'][i] - dd_data['z'][i], max(dd_data['y'][i], dd_data['z'][i])) for i in dd_data.index]
            res = dd_data.reindex(columns=['baseline', 'opt', 'y', 'z', 'max(y,z,1-y-z)', 'exp_ratio', 'real_ratio', 'r1', 'base_sample', 'base_move', 'base_cal', 'opt_sample', 'opt_move', 'opt_cal', 'y', 'z', '1-y-z', 'y1', 'z1', '1-y1-z1'])
            res.to_csv(dir_out + f'/sampling_{key[:-4]}.csv')


index = []
for alg in ['gcn', 'ggnn', 'gat', 'gaan']:
    for mode in ['cluster', 'graphsage']:
        for data in ['amazon-photo', 'coauthor-physics']:
            index.append(f'{alg}_{mode}_{data}')

key = 'batches_amc.log'

dd_data = read_file(key)
dd_data = pd.DataFrame(dd_data.values(), index=index[:-1], columns=columns)
dd_data['1-y-z'] = [1 - dd_data['y'][i] - dd_data['z'][i] for i in dd_data.index]
dd_data['r1'] = dd_data['real_ratio'] / dd_data['exp_ratio'] # 真实的比上期待的，越接近1表示越好
dd_data['y1'] = [dd_data['opt_sample'][i] / (dd_data['opt_sample'][i] + dd_data['opt_move'][i] + dd_data['opt_cal'][i]) for i in dd_data.index]
dd_data['z1'] = [dd_data['opt_move'][i] / (dd_data['opt_sample'][i] + dd_data['opt_move'][i] + dd_data['opt_cal'][i]) for i in dd_data.index]
dd_data['1-y1-z1'] = [dd_data['opt_cal'][i] / (dd_data['opt_sample'][i] + dd_data['opt_move'][i] + dd_data['opt_cal'][i]) for i in dd_data.index]
# dd_data['exp_real_ratio'] = [1/50 + 49 * max(y, 1-y)/50 for y in dd_data['y']]
# dd_data['r2'] = dd_data['real_ratio'] / dd_data['exp_real_ratio']
dd_data['max(y,z,1-y-z)'] = [max(1 - dd_data['y'][i] - dd_data['z'][i], max(dd_data['y'][i], dd_data['z'][i])) for i in dd_data.index]
res = dd_data.reindex(columns=['baseline', 'opt', 'y', 'z', 'max(y,z,1-y-z)', 'exp_ratio', 'real_ratio', 'r1', 'base_sample', 'base_move', 'base_cal', 'opt_sample', 'opt_move', 'opt_cal', 'y', 'z', '1-y-z', 'y1', 'z1', '1-y1-z1'])
res.to_csv(dir_out + f'/{key[:-4]}.csv')