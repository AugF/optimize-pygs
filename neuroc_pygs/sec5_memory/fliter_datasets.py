import os
import torch
import numpy as np 
from collections import defaultdict

from neuroc_pygs.options import build_dataset, get_args, build_dataset, build_model_optimizer
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
    print(model.loss_fn)
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

    data = data.to(args.device)
    for epoch in range(1): # 实验测试都一样
        train(model, data, optimizer)
        val_acc, val_loss = test(model, data, split='val')
        print(f'Epoch: {epoch:03d}, val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}')

    peak_memory = torch.cuda.memory_stats(args.device)["allocated_bytes.all.peak"]
    torch.cuda.reset_max_memory_allocated(args.device)
    print(args.dataset, peak_memory)


if __name__ == '__main__':
    import sys
    sys.argv = [sys.argv[0], '--dataset', 'random_fli0']
    run()