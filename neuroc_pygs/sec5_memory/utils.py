import os, sys
import copy
import torch
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, NamedTuple, Union, Callable
from torch import Tensor
from torch_sparse import SparseTensor
from sklearn.metrics import mean_squared_error
from neuroc_pygs.configs import PROJECT_PATH, EXP_DATASET
from neuroc_pygs.options import get_args, build_dataset, build_model
    

def convert_datasets():
    import pandas as pd
    dir_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec5_memory/exp_motivation_diff'
    dir_out = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec5_memory/exp_automl_datasets_diff'
    default_args = '--hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32'
    # gcn: layers, n_features, n_classes, hidden_dims
    # gat: layers, n_features, n_classes, head_dims, heads
    for exp_model in ['gcn', 'gat']:
        for exp_data in ['yelp', 'reddit']:
            Xs = []
            if exp_data == 'reddit' and exp_model == 'gat':
                re_bs = [170, 175, 180]
            else:
                re_bs = [175, 180, 185]
            for rs in re_bs:
                real_path = dir_path + f'/{exp_data}_{exp_model}_{rs}_cluster_v2.csv'
                sys.argv = [sys.argv[0], '--dataset', exp_data, '--batch_partitions', str(rs), '--model', exp_model] + default_args.split(' ')
                args = get_args()
                data = build_dataset(args)
                model = build_model(args, data)
                paras_dict = model.get_hyper_paras()
                paras = [v for v in paras_dict.values()]
                df = pd.read_csv(real_path, index_col=0).values
                X, y = df[:, :2], df[:, -2:]
                X_paras = np.array(paras * len(y)).reshape(-1, len(paras))
                X = np.concatenate([X, X_paras, y], axis=1)
                Xs.append(X)
            Xs = np.concatenate(Xs, axis=0)
            pd.DataFrame(Xs).to_csv(dir_out + f'/{exp_model}_{exp_data}_automl_model_diff_v2.csv')


import re
from collections import defaultdict
from tabulate import tabulate
dir_out = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_train_res')
log_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec5_memory/exp_log_diff/prove_train_acc_resampling.log'
tab_data = []
headers = ['Model', 'Data', 'Per', 'Acc1', 'Acc2', 'Acc3', 'Use time']
columns = ['train_acc', 'val_acc', 'test_acc', 'best_val_acc', 'final_test_acc']
small_datasets =  ['pubmed', 'coauthor-physics']

f = open(log_path)
for data in small_datasets:
    for model in ['gcn', 'gat']:
        for discard_per in [0, 0.01, 0.03, 0.06, 0.1, 0.2, 0.5]:
            test_accs = []
            for run in range(3):
                real_path = dir_out + f'/{model}_{data}_{str(int(100*discard_per))}_{run}.csv'
                if os.path.exists(real_path):
                    test_acc = pd.read_csv(real_path, index_col=0).values[-1, -1]
                    test_accs.append(test_acc)
                    continue
                df = defaultdict(list)
                # on  df:50
                cnt = 0
                while cnt < 50:
                    mystr = f.readline()
                    if not mystr.startswith('Epoch'):
                        continue
                    matchline = re.match(r'.*Train: (.*), Val: (.*), Test: (.*), Best Val: (.*), Test: (.*)', mystr)
                    print(mystr)
                    if matchline:
                        df['train_acc'].append(float(matchline.group(1)))
                        df['val_acc'].append(float(matchline.group(2)))
                        df['test_acc'].append(float(matchline.group(3)))
                        df['best_val_acc'].append(float(matchline.group(4)))
                        df['final_test_acc'].append(float(matchline.group(5)))   
                        cnt += 1
                pd.DataFrame(df).to_csv(real_path)
                # off
            print('begin matchline')
            while True:
                mystr = f.readline()
                if not mystr.startswith('['):
                    continue
                matchline = re.match(r".*, (.*), (.*)]", mystr) 
                if matchline:
                    test_acc, use_time = float(matchline.group(1)), float(matchline.group(2))
                    break
            res = [model, data, discard_per] + test_accs + [use_time]
            print(res)
            tab_data.append(res)
print(tabulate(tab_data, headers=headers, tablefmt='github'))
pd.DataFrame(tab_data, columns=headers).to_csv(dir_out + '/prove_train_acc_all.csv')
                                