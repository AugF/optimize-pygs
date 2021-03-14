import os
import torch
import sys
import time 

import numpy as np 
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump, load

from tabulate import tabulate
from collections import defaultdict
from neuroc_pygs.options import build_dataset, get_args, build_dataset, build_model_optimizer
from neuroc_pygs.configs import PROJECT_PATH


dir_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_motivation_datasets')

def train(model, batch, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(batch.x, batch.edge_index)
    loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(model, batch, split='val'):
    model.eval()
    logits = model(batch.x, batch.edge_index)
    mask = getattr(batch, split + '_mask')
    loss = model.loss_fn(logits[mask], batch.y[mask])
    acc = model.evaluator(logits[mask], batch.y[mask])
    return acc, loss.item()


def run():
    args = get_args()
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model = model.to(args.device)
    memory = torch.cuda.memory_stats(args.device)["allocated_bytes.all.peak"]
    torch.cuda.reset_max_memory_allocated(args.device)
    print(f'device: {args.device}, model memory: {memory}, model: {args.model}')

    data = data.to(args.device)
    for epoch in range(2): # 实验测试都一样
        train(model, data, optimizer)
        val_acc, val_loss = test(model, data, split='val')
        peak_memory = torch.cuda.memory_stats(args.device)["allocated_bytes.all.peak"]
        torch.cuda.reset_max_memory_allocated(args.device)
        print(f'Epoch: {epoch:03d}, peak_memory: {peak_memory}')

    res = [args.dataset, data.num_nodes, data.num_edges, peak_memory]
    print(res)
    return res


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
            tab_data.append(run())
    t2 = time.time()
    pd.DataFrame(tab_data, columns=headers).to_csv(os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_motivation_datasets', f'{model}_linear_model_v1.csv'))
    print(model, tabulate(tab_data, headers=headers, tablefmt='github'))
    return t2 - t1


def build_datasets():
    for model in ['gcn', 'gat']:
        use_time = run_dataset(model)
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
    run_exp()
    # reg = load(dir_path + f'/gcn_linear_model_v1.pth')
    # res = reg.predict([[85000, 85000]])
    # print(res)
