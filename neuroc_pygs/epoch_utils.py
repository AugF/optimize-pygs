import os, sys
import time
import torch
import numpy as np
import torch.nn.functional as F

def train_full(model, data, optimizer):
    model.train()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(F.log_softmax(out, dim=1)[data.train_mask], data.y[data.train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_full(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    logits, accs = F.log_softmax(out, dim=1), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


@torch.no_grad()
def infer(model, data, subgraphloader, df_time=None):
    model.eval()
    model.reset_parameters()
    y_pred = model.inference(data.x, subgraphloader, df_time=df_time) # 这里使用inference_cuda作为测试
    y_true = data.y.cpu()

    accs, losses = [], []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        loss = model.loss_fn(y_pred[mask], y_true[mask])
        acc = model.evaluator(y_pred[mask], y_true[mask]) 
        losses.append(loss.item())
        accs.append(acc)
    return accs, losses


def train(model, optimizer, data, loader, device, mode, non_blocking=False, df=None, opt_flag=False):
    model.reset_parameters()
    model.train()
    all_loss = []
    loader_num, loader_iter = len(loader), iter(loader)
    for _ in range(loader_num):
        if mode == 'cluster':
            t1 = time.time()
            batch = next(loader_iter)
            t2 = time.time()
            batch = batch.to(device, non_blocking=non_blocking)
            t3 = time.time()
            batch_size = batch.train_mask.sum().item()
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            t4 = time.time()
        elif mode == 'graphsage':
            t1 = time.time()
            if opt_flag:
                batch_size, n_id, adjs, x, y = next(loader_iter)
            else:
                batch_size, n_id, adjs = next(loader_iter)
                x, y = data.x[n_id], data.y[n_id[:batch_size]]
            t2 = time.time()
            x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
            adjs = [adj.to(device, non_blocking=non_blocking) for adj in adjs]
            t3 = time.time()
            optimizer.zero_grad()
            logits = model(x, adjs)
            loss = model.loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            t4 = time.time()
        if df is not None:
            df['sample'].append(t2 - t1)
            df['move'].append(t3 - t2)
            df['cal'].append(t4 - t3)
            df['cnt'][0] += 1
            if df['cnt'][0] >= df['max_cnt'][0]:
                break
            print(f"Batch:{df['cnt'][0]}, sample: {t2-t1}, move: {t3-t2}, cal: {t4-t3}")
        all_loss.append(loss.item() * batch_size)
    return np.sum(all_loss) / int(data.train_mask.sum())

