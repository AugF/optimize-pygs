import torch
import time
import os.path as osp
import numpy as np

from neuroc_pygs.utils import BatchLogger
from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.samplers.prefetch_generator import BackgroundGenerator


def train(model, data, train_loader, optimizer, args, opt_loader=False):
    mode, device, log_batch, log_batch_dir = args.mode, args.device, args.log_batch, args.log_batch_dir
    model.train()
    logger = BatchLogger()
    all_acc, all_loss = [], []
    st0 = time.time()
    loader_iter, loader_num = iter(train_loader), len(train_loader)
    st1 = time.time()
    if log_batch:
        print("start", st1 - st0)
    if opt_loader:
        loader_iter = BackgroundGenerator(loader_iter)
    for i in range(loader_num):
        t1 = time.time()
        if mode == "cluster":
            # task1
            optimizer.zero_grad()
            batch = next(loader_iter)
            # task2
            t2 = time.time()
            batch = batch.to(device)
            # task3
            t3 = time.time()
            logits = model(batch.x, batch.edge_index)
            loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            acc = model.evaluator(logits[batch.train_mask], batch.y[batch.train_mask])
            optimizer.step()
        else:
            # task1
            batch_size, n_id, adjs = next(loader_iter)
            x, y = data.x[n_id], data.y[n_id[:batch_size]]
            t2 = time.time()
            x, y = x.to(device), y.to(device)
            adjs = [adj.to(device) for adj in adjs]
            t3 = time.time()
            # task3
            logits = model(x, adjs)
            loss = model.loss_fn(logits, y)
            loss.backward()
            acc = model.evaluator(logits, y) / batch_size
            optimizer.step()
        t4 = time.time()
        if log_batch:
            logger.add_batch(t2 - t1, t3 - t2, t4 - t3)
        all_loss.append(loss.item())
        all_acc.append(acc)
    if log_batch:
        logger.print_batch()
        if log_batch_dir:
            logger.save(osp.join(PROJECT_PATH, 'log', args.log_epoch_dir))
    return np.mean(all_acc), np.mean(all_loss)


@torch.no_grad()
def test(model, data, subgraph_loader, args, split="val", opt_loader=False):
    device, log_batch, log_batch_dir = args.device, args.log_batch, args.log_batch_dir
    model.eval()
    logger = BatchLogger()
    all_loss, all_acc = [], []
    loader_iter, loader_num = iter(subgraph_loader), len(subgraph_loader)
    if opt_loader:
        loader_iter = BackgroundGenerator(loader_iter)
    for i in range(loader_num):
        # start 
        t1 = time.time()
        batch_size, n_id, adjs = next(loader_iter)
        x, y = data.x[n_id], data.y[n_id[:batch_size]]
        # task2
        t2 = time.time()
        x, y = x.to(device), y.to(device)
        adjs = [adj.to(device) for adj in adjs]
        # task3
        t3 = time.time()
        logits = model(x, adjs)
        loss = model.loss_fn(logits, y)
        acc = model.evaluator(logits, y) / batch_size
        # end
        all_loss.append(loss.item())
        all_acc.append(acc)
        t4 = time.time()
        if log_batch:
            logger.add_batch(t2-t1, t3-t2, t4-t3)
    if log_batch:
        logger.print_batch()
        if log_batch_dir:
            logger.save(osp.join(PROJECT_PATH, 'log', args.log_epoch_dir))
    return np.mean(all_acc), np.mean(all_loss)


@torch.no_grad()
def infer(model, data, subgraph_loader, args, split="val", opt_loader=False):
    device, log_batch, log_batch_dir = args.device, args.log_batch, args.log_batch_dir
    model.eval()
    y_pred = model.inference(data.x, subgraph_loader, log_batch, opt_loader)
    y_true = data.y.cpu()

    mask = getattr(data, split + "_mask")
    loss = model.loss_fn(y_pred[mask], y_true[mask])
    acc = model.evaluator(y_pred[mask], y_true[mask]) 
    return acc, loss


if __name__ == '__main__':
    from neuroc_pygs.options import prepare_trainer
    data, train_loader, subgraph_loader, model, optimizer, args = prepare_trainer(log_batch=True, mode='graphsage') # 观察一下
    model = model.to(args.device)
    train(model, data, train_loader, optimizer, args)