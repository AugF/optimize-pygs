import os
import torch
import sys

import numpy as np 
import matplotlib.pyplot as plt

from tabulate import tabulate
from collections import defaultdict
from neuroc_pygs.options import build_dataset, get_args, build_dataset, build_model_optimizer
from neuroc_pygs.configs import PROJECT_PATH


root = '/mnt/data/wangzhaokang/wangyunpan/data'
datasets = ['amc', 'fli']

def test_equal():
    args = get_args()
    # 0->59
    for data_name in datasets:
        equals_data = []
        for i in range(40):
            file_name = f'random_{data_name}{i}'
            file_path = os.path.join(root, file_name)
            print(file_path)
            args.dataset = file_name
            data = build_dataset(args)
            print(data.num_nodes, data.num_edges)
            equals_data.append(data.edge_index)
        tmp = equals_data[0]
        for i in range(1, len(equals_data)):
            print(tmp.equal(equals_data[i]))


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
    for epoch in range(1): # 实验测试都一样
        train(model, data, optimizer)
        val_acc, val_loss = test(model, data, split='val')
        print(f'Epoch: {epoch:03d}, val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}')

    peak_memory = torch.cuda.memory_stats(args.device)["allocated_bytes.all.peak"]
    torch.cuda.reset_max_memory_allocated(args.device)
    print(args.dataset, peak_memory, '\n')
    return [args.dataset, data.num_nodes, data.num_edges, peak_memory]


def prove_memory():
    amc_datasets = ['amazon-computers'] + [f'random_amc{i}' for i in range(40)]
    fli_datasets = ['flickr'] + [f'random_fli{i}' for i in range(40)]

    tab_data = []
    for exp_datasets in [amc_datasets, fli_datasets]:
        for exp_data in exp_datasets:
            sys.argv = [sys.argv[0], '--dataset', exp_data]
            tab_data.append(run())
    print(tabulate(tab_data, headers=['Name', 'Nodes', 'Edges', 'Peak Memory'], tablefmt='github'))


def get_memory_curve():
    nodes_datasets = [1] + np.arange(2.5, 50, 2.5).tolist()
    edges_datasets = np.arange(1, 10).tolist() + np.arange(10, 71, 5).tolist()

    tab_data = []
    for i, exp_datasets in enumerate([nodes_datasets, edges_datasets]):
        for var in exp_datasets:
            exp_data = f'random_{var}_500k' if not i else f'random_10k_{var}'
            sys.argv = [sys.argv[0], '--dataset', exp_data, '--device', 'cuda:2']
            tab_data.append(run())
    np.save(os.path.join(PROJECT_PATH, 'sec5_memory', 'log', 'gat_memory_curve_data.npy'), tab_data)
    print(tabulate(tab_data, headers=['Name', 'Nodes', 'Edges', 'Peak Memory'], tablefmt='github'))


def pics_memory_curve():
    tab_data = np.load(os.path.join(PROJECT_PATH, 'sec5_memory', 'log', 'gat_memory_curve_data.npy'))
    tab_data = np.array(tab_data)[:,3]
    tab_data = list(map(lambda x: int(x), tab_data))
    # 
    nodes_x = [1] + np.arange(2.5, 50, 2.5).tolist()
    edges_x = np.arange(1, 10).tolist() + np.arange(10, 71, 5).tolist()
    
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(7 * 2, 5), tight_layout=True)
    axes[0].plot(nodes_x, tab_data[:20], marker='o')
    axes[1].plot(edges_x, tab_data[20:], marker='D')
    fig.savefig(os.path.join(PROJECT_PATH, 'sec5_memory', 'log', 'gat_memory_curve.png'))

