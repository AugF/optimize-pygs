import time
import os
import numpy as np

from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.samplers.prefetch_generator import BackgroundGenerator
from neuroc_pygs.samplers.data_prefetcher import DataPrefetcher


def train(model, optimizer, data, loader_iter, loader_num, device, mode):
    model.train()
    all_acc, all_loss = [], []
    for i in range(loader_num):
        if mode == 'cluster':
            optimizer.zero_grad()
            batch = next(loader_iter)
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            acc = model.evaluator(logits[batch.train_mask], batch.y[batch.train_mask])
            optimizer.step()
        elif mode == 'graphsage':
            batch_size, n_id, adjs = next(loader_iter)
            x, y = data.x[n_id], data.y[n_id[:batch_size]]
            x, y = x.to(device), y.to(device)
            adjs = [adj.to(device) for adj in adjs]
            # task3
            logits = model(x, adjs)
            loss = model.loss_fn(logits, y)
            loss.backward()
            acc = model.evaluator(logits, y) / batch_size
            optimizer.step()
        all_loss.append(loss.item())
        all_acc.append(acc)
    return np.mean(all_acc), np.mean(all_loss)


def train_cuda(model, optimizer, data, loader_iter, loader_num, device, mode):
    model.train()
    all_acc, all_loss = [], []
    for i in range(loader_num):
        if mode == 'cluster':
            optimizer.zero_grad()
            batch = next(loader_iter)
            logits = model(batch.x, batch.edge_index)
            loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            acc = model.evaluator(logits[batch.train_mask], batch.y[batch.train_mask])
            optimizer.step()
        elif mode == 'graphsage':
            batch_size, x, y, adjs  = next(loader_iter)
            logits = model(x, adjs)
            loss = model.loss_fn(logits, y)
            loss.backward()
            acc = model.evaluator(logits, y) / batch_size
            optimizer.step()
        all_loss.append(loss.item())
        all_acc.append(acc)
    return np.mean(all_acc), np.mean(all_loss)


if __name__ == '__main__':
    # gpu上的结果都一样, cpu因为随机种子不一样，所以结果不一样
    from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer, build_train_loader
    args = get_args()
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    train_loader = build_train_loader(args, data)
    loader_iter, loader_num, device, mode = iter(train_loader), len(train_loader), args.device, args.mode
    model = model.to(device)
    loader_iter = BackgroundGenerator(loader_iter)
    loader_iter = DataPrefetcher(loader_iter, mode, device, None if mode == 'cluster' else data)
    # res = train(model, optimizer, data, loader_iter, loader_num, device, mode)
    res = train_cuda(model, optimizer, data, loader_iter, loader_num, device, mode) 
    print(res)