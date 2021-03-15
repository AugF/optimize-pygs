import os
import torch
import sys

import numpy as np 
import matplotlib.pyplot as plt

from tabulate import tabulate
from collections import defaultdict
from neuroc_pygs.options import build_dataset, get_args, build_dataset, build_model_optimizer
from neuroc_pygs.configs import PROJECT_PATH


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


def get_linear_plane(model='gat'):
    default_args = '--hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32'
    tab_data = []
    for nodes in range(5000, 100001, 5000):
        for expect_edges in range(5000, 100001, 5000):
            exp_data = f'random_{int(nodes/1000)}k_{int(expect_edges/1000)}k'
            sys.argv = [sys.argv[0], '--dataset', exp_data, '--device', 'cuda:2', '--model', model] + default_args.split(' ')
            tab_data.append(run())
    np.save(os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_res', f'{model}_linear_plane_data.npy'), tab_data)
    print(tabulate(tab_data, headers=['Name', 'Nodes', 'Edges', 'Peak Memory'], tablefmt='github'))


def pics_linear_plane(model='gat'):
    tab_data = np.load(os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_res', f'{model}_memory_linear_plane_data.npy'))
    nodes = list(map(lambda x: int(x), tab_data[:, 1]))
    edges = list(map(lambda x: int(x), tab_data[:, 2]))
    memory = list(map(lambda x: int(x) / (1024*1024), tab_data[:, 3]))
    print(nodes, edges, memory)

    # from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('Number of Vertices', fontsize=12)
    ax.set_ylabel('Number of Edges', fontsize=12)
    ax.set_zlabel('Peak Memory (MB)', fontsize=12)
    ax.scatter3D(nodes, edges, memory, cmap='Blues')
    fig.savefig(os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_figs', f'{model}_memory_2dims_curve_data.png'))


if __name__ == '__main__':
    get_linear_plane()