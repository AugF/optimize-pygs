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
from joblib import dump, load

from tabulate import tabulate
from collections import defaultdict
from neuroc_pygs.options import build_dataset, get_args, build_dataset, build_model_optimizer
from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.sec5_memory.configs import MODEL_PARAS


dir_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_automl_datasets')

def train(model, batch, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(batch.x, batch.edge_index)
    loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
    loss.backward()
    optimizer.step()


def run_automl():
    args = get_args()
    print(args)
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model = model.to(args.device)
    memory = torch.cuda.memory_stats(args.device)["allocated_bytes.all.peak"]
    torch.cuda.reset_max_memory_allocated(args.device)
    torch.cuda.empty_cache()
    print(f'device: {args.device}, model memory: {memory}, model: {args.model}')

    data = data.to(args.device)
    peak_memorys = []
    for epoch in range(5): # 实验测试都一样
        train(model, data, optimizer)
        peak_memory = torch.cuda.memory_stats(args.device)["allocated_bytes.all.peak"]
        torch.cuda.reset_max_memory_allocated(args.device)
        torch.cuda.empty_cache()
        print(f'Epoch: {epoch}, peak_memory: {peak_memory}')
        if epoch > 0:
            peak_memorys.append(peak_memory)
    paras_dict = model.get_hyper_paras()
    res = [data.num_nodes, data.num_edges] + [v for v in paras_dict.values()] + [np.mean(peak_memory)]
    print(res)
    return res


def run_linear():
    args = get_args()
    print(args)
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model = model.to(args.device)
    memory = torch.cuda.memory_stats(args.device)["allocated_bytes.all.peak"]
    torch.cuda.reset_max_memory_allocated(args.device)
    torch.cuda.empty_cache()
    print(f'device: {args.device}, model memory: {memory}, model: {args.model}')

    data = data.to(args.device)
    peak_memorys = []
    for epoch in range(5): # 实验测试都一样
        train(model, data, optimizer)
        peak_memory = torch.cuda.memory_stats(args.device)["allocated_bytes.all.peak"]
        torch.cuda.reset_max_memory_allocated(args.device)
        torch.cuda.empty_cache()
        print(f'Epoch: {epoch}, peak_memory: {peak_memory}')
        if epoch > 0:
            peak_memorys.append(peak_memory)
    res = [args.dataset, data.num_nodes, data.num_edges, np.mean(peak_memory)]
    print(res)
    return res


def run_automl_dataset(model):
    default_args = '--hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32'
    t1 = time.time()
    tab_data = []
    # vars_set = set(range(5000, 100001, 5000)).union(set(range(5000, 100001, 2000)))
    vars_set = range(5000, 100001, 5000) 
    for nodes in vars_set:
        for edges in vars_set:
            exp_data = f'random_{int(nodes/1000)}k_{int(edges/1000)}k'
            print(exp_data)
            if not os.path.exists('/mnt/data/wangzhaokang/wangyunpan/data/' + exp_data):
                continue
            sys.argv = [sys.argv[0], '--dataset', exp_data, '--device', 'cuda:2', '--model', model] + default_args.split(' ')
            try:
                tab_data.append(run_automl())
                gc.collect()
            except Exception as e:
                print(e.args)
                print(traceback.format_exc())
    pd.DataFrame(tab_data).to_csv(dir_path + f'/{model}_nodes_edges_automl_model_v2.csv')
    
    tab_data = []
    nodes, expect_edges = 35000, 45000
    base_name = f'random_{int(nodes/1000)}k_{int(expect_edges/1000)}k'
    for features in range(50, 5001, 500): # 10
        exp_data = base_name + f'_{features}_7'
        if not os.path.exists('/mnt/data/wangzhaokang/wangyunpan/data/' + exp_data):
            continue
        sys.argv = [sys.argv[0], '--dataset', exp_data, '--device', 'cuda:2', '--model', model]
        try:
            tab_data.append(run_automl())
            gc.collect()
        except Exception as e:
            print(e.args)
            print(traceback.format_exc())
    pd.DataFrame(tab_data).to_csv(dir_path + f'/{model}_features_automl_model_v2.csv')  

    tab_data = []
    for classes in range(3, 301, 30): # 10
        exp_data = base_name + f'_500_{classes}'
        print(exp_data)
        if not os.path.exists('/mnt/data/wangzhaokang/wangyunpan/data/' + exp_data):
            continue
        if os.path.exists(dir_path + f'/{model}_{exp_data}_automl_model_v2.csv'):
            continue
        sys.argv = [sys.argv[0], '--dataset', exp_data, '--device', 'cuda:2', '--model', model]
        for para, para_values in MODEL_PARAS[model].items():
            for p_v in para_values: #  paras: 13
                sys.argv += [f'--{para}', str(p_v)]
                try:
                    tab_data.append(run_automl())
                    gc.collect()
                except Exception as e:
                    print(e.args)
                    print(traceback.format_exc())
        pd.DataFrame(tab_data).to_csv(dir_path + f'/{model}_classes_automl_model_v2.csv')

    
    tab_data = []
    init_argv = [sys.argv[0], '--dataset', 'random_15k_20k', '--device', 'cuda:2', '--model', model]
    for para, para_values in MODEL_PARAS[model].items():
        for p_v in para_values: #  paras: 13
            sys.argv = init_argv + [f'--{para}', str(p_v)]
            try:
                tab_data.append(run_automl())
                gc.collect()
            except Exception as e:
                print(e.args)
                print(traceback.format_exc())
    pd.DataFrame(tab_data).to_csv(dir_path + f'/{model}_paras_automl_model_v4.csv')

    t2 = time.time()
    return t2 - t1


def run_dataset(model):
    headers = ['Name', 'Nodes', 'Edges', 'Peak Memory']
    default_args = '--hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32'
    tab_data = []
    t1 = time.time()
    vars_set = set(range(5000, 100001, 5000)).union(set(range(5000, 100001, 2000)))
    for nodes in vars_set:
        for edges in vars_set:
            exp_data = f'random_{int(nodes/1000)}k_{int(edges/1000)}k'
            print(exp_data)
            if not os.path.exists('/mnt/data/wangzhaokang/wangyunpan/data/' + exp_data):
                continue
            sys.argv = [sys.argv[0], '--dataset', exp_data, '--device', 'cuda:2', '--model', model] + default_args.split(' ')
            tab_data.append(run_linear())
            gc.collect()
    t2 = time.time()
    pd.DataFrame(tab_data, columns=headers).to_csv(os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_automl_datasets', f'{model}_linear_model_v2.csv'))
    return t2 - t1


def build_datasets():
    for model in ['gcn', 'gat']:
        use_time = run_dataset(model)
    print(f'{model} build datasets use time: {use_time}s')


def build_automl_datasets():
    for model in ['gcn', 'gat']:
        use_time = run_automl_dataset(model)
    print(f'{model} build datasets use time: {use_time}s')


def run_exp():
    for model in ['gcn', 'gat']:
        real_path = dir_path + f'/{model}_linear_model_v1.csv'
        df = pd.read_csv(real_path, index_col=0)
        X, y = df[['Nodes', 'Edges']].values, df['Peak Memory'].values / (1024 * 1024)
        np.random.seed(1)
        mask = np.arange(len(y))
        np.random.shuffle(mask)
        X, y = X[mask], y[mask]
        X_train, y_train, X_test, y_test = X[:-1000], y[:-1000], X[-1000:], y[-1000:]
        t1 = time.time()
        reg = LinearRegression().fit(X_train, y_train)
        t2 = time.time()
        dump(reg, dir_path + f'/{model}_linear_model_v1.pth')
        # reg = load(dir_path + f'/{model}_linear_model_v0.pth')
        y_pred = reg.predict(X_test)
        t3 = time.time()
        mse = mean_squared_error(y_pred, y_test)
        max_bias, max_bias_per = 0, 0
        for i in range(1000):
            max_bias = max(max_bias, abs(y_pred[i] - y_test[i]))
            max_bias_per = max(max_bias_per, abs(y_pred[i] - y_test[i]) / y_pred[i])
        print(model, t2 - t1, (t3 - t2) / 100)
        print(model, mse, max_bias, max_bias_per)


if __name__ == '__main__':
    build_automl_datasets()
    # build_datasets()
    # run_exp()
    # reg = load(dir_path + f'/gcn_linear_model_v1.pth')
    # res = reg.predict([[85000, 85000]])
    # print(res)
