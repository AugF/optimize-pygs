import os
import torch
import sys
import time 
import gc

import numpy as np 
import pandas as pd
import traceback

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from tabulate import tabulate
from collections import defaultdict
from neuroc_pygs.options import build_dataset, get_args, build_dataset, build_model_optimizer
from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.configs import MODEL_PARAS


def train(model, batch, optimizer): # 全数据训练
    model.train()
    optimizer.zero_grad()
    logits = model(batch.x, batch.edge_index)
    loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
    loss.backward()
    optimizer.step()


def run_random_forest(): 
    args = get_args()
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model = model.to(args.device)
    memory = torch.cuda.memory_stats(args.device)["allocated_bytes.all.current"]
    torch.cuda.reset_max_memory_allocated(args.device)
    torch.cuda.empty_cache()

    data = data.to(args.device)
    peak_memorys = []
    for epoch in range(5): 
        train(model, data, optimizer)
        peak_memory = torch.cuda.memory_stats(args.device)["allocated_bytes.all.peak"]
        torch.cuda.reset_max_memory_allocated(args.device)
        torch.cuda.empty_cache()
        print(f'Epoch: {epoch}, memory: {memory}, peak_memory: {peak_memory}, differ memory: {peak_memory - memory}')
        memory = torch.cuda.memory_stats(args.device)["allocated_bytes.all.current"]
        if epoch > 0:
            peak_memorys.append(peak_memory)
    paras_dict = model.get_hyper_paras() # 获取超参数
    res = [data.num_nodes, data.num_edges] + [v for v in paras_dict.values()] + [np.mean(peak_memory), np.mean(peak_memory) - memory]
    print(res)
    return res


def build_random_forest_dataset(model, file_suffix='v0'):
    # 边数和点数
    default_args = '--hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32'
    tab_data = []
    vars_set = range(5000, 100001, 5000) 
    for nodes in vars_set:
        for edges in vars_set:
            exp_data = f'random_{int(nodes/1000)}k_{int(edges/1000)}k'
            print(exp_data)
            if not os.path.exists('/mnt/data/wangzhaokang/wangyunpan/data/' + exp_data):
                continue
            sys.argv = [sys.argv[0], '--dataset', exp_data, '--model', model] + default_args.split(' ')
            try:
                tab_data.append(run_random_forest())
                gc.collect()
            except Exception as e:
                print(e.args)
                print(traceback.format_exc())
    pd.DataFrame(tab_data).to_csv(dir_path + f'/{model}_nodes_edges_random_forest_{file_suffix}.csv')

    # 特征数
    tab_data = []
    nodes, expect_edges = 35000, 45000
    base_name = f'random_{int(nodes/1000)}k_{int(expect_edges/1000)}k'
    for features in range(50, 5001, 500): # 10
        exp_data = base_name + f'_{features}_7'
        if not os.path.exists('/mnt/data/wangzhaokang/wangyunpan/data/' + exp_data):
            continue
        sys.argv = [sys.argv[0], '--dataset', exp_data, '--model', model]
        try:
            tab_data.append(run_random_forest())
            gc.collect()
        except Exception as e:
            print(e.args)
            print(traceback.format_exc())
    pd.DataFrame(tab_data).to_csv(dir_path + f'/{model}_features_random_forest_{file_suffix}.csv')  

    # 类别数
    tab_data = []
    for classes in range(3, 301, 30): # 10
        exp_data = base_name + f'_500_{classes}'
        if not os.path.exists('/mnt/data/wangzhaokang/wangyunpan/data/' + exp_data):
            continue
        if os.path.exists(dir_path + f'/{model}_{exp_data}_random_forest_{file_suffix}.csv'):
            continue
        sys.argv = [sys.argv[0], '--dataset', exp_data, '--model', model]
        for para, para_values in MODEL_PARAS[model].items():
            for p_v in para_values: #  paras: 13
                sys.argv += [f'--{para}', str(p_v)]
                try:
                    tab_data.append(run_random_forest())
                    gc.collect()
                except Exception as e:
                    print(e.args)
                    print(traceback.format_exc())
    pd.DataFrame(tab_data).to_csv(dir_path + f'/{model}_classes_random_forest_{file_suffix}.csv')
    
    # 算法自带的参数
    tab_data = []
    init_argv = [sys.argv[0], '--model', model]
    for dataset in ['random_15k_20k', 'random_25k_10k']:
        for para, para_values in MODEL_PARAS[model].items():
            for p_v in para_values: 
                sys.argv = init_argv + [f'--{para}', str(p_v), '--dataset', dataset]
                try:
                    tab_data.append(run_random_forest())
                    gc.collect()
                except Exception as e:
                    print(e.args)
                    print(traceback.format_exc())
    pd.DataFrame(tab_data).to_csv(dir_path + f'/{model}_paras_random_forest_{file_suffix}.csv')
    return


def build_random_forest_datasets():
    for model in ['gcn', 'gat']:
        build_random_forest_dataset(model)
        

if __name__ == '__main__':
    dir_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'out_random_forest_datasets')
    build_random_forest_datasets()