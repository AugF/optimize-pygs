import time
import os
import numpy as np

from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.samplers.prefetch_generator import BackgroundGenerator
from neuroc_pygs.samplers.data_prefetch import DataPrefetcher
from neuroc_pygs.options import get_args, build_datasets, build_model_optimizer, build_train_loader


def train(model, data, loader_iter, loader_num, optimizer, device, mode):
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


def train_cuda(model, data, loader_iter, loader_num, optimizer, device, mode):
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



