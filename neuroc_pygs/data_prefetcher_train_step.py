import torch
import time
import os.path as osp
import numpy as np

from neuroc_pygs.utils import BatchLogger
from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.samplers import DataPrefetcher
from neuroc_pygs.train_step import train


def opt_train(model, data, train_loader, optimizer, args):
    mode, device, log_batch, log_batch_dir = args.mode, args.device, args.log_batch, args.log_batch_dir
    model.train()
    all_acc, all_loss = [], []
    st0 = time.time()
    loader_iter = DataPrefetcher(loader=train_loader, sampler=mode, device=args.device, data=None if args.model == 'cluster' else data)
    loader_num = len(train_loader)
    st1 = time.time()
    if log_batch:
        print("start time", st1 - st0)
    logger = BatchLogger()
    for i in range(loader_num):
        t1 = time.time()
        if mode == "cluster":
            optimizer.zero_grad()
            batch = loader_iter.next()
            t3 = time.time()
            logits = model(batch.x, batch.edge_index)
            loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            acc = model.evaluator(logits[batch.train_mask], batch.y[batch.train_mask])
            optimizer.step()
        else:
            # task1
            batch_size, x, y, adjs = next(loader_iter)
            t3 = time.time()
            # task3
            logits = model(x, adjs)
            loss = model.loss_fn(logits, y)
            loss.backward()
            acc = model.evaluator(logits, y) / batch_size
            optimizer.step()
        t4 = time.time()
        if log_batch:
            logger.add_batch(t3 - t1, 0, t4 - t3)
        all_loss.append(loss.item())
        all_acc.append(acc)
    if log_batch:
        logger.print_batch()
    return np.mean(all_acc), np.mean(all_loss)



def func(data, train_loader, subgraph_loader, model, optimizer, args):
    model, data = model.to(args.device), data.to(args.device)
    args.log_batch = True
    # begin test
    t1 = time.time()
    train_acc = train(model, data, train_loader, optimizer, args)
    t2 = time.time()
    train_acc = opt_train(model, data, train_loader, optimizer, args)
    t3 = time.time()
    base_time, opt_time = t2 - t1, t3 - t2
    ratio = 100 * (base_time - opt_time) / base_time
    return base_time, opt_time, ratio


if __name__ == '__main__':
    from neuroc_pygs.configs import ALL_MODELS, EXP_DATASET, MODES, EXP_RELATIVE_BATCH_SIZE
    from neuroc_pygs.options import run, run_all
    import sys
    # run_all(func, runs=3, path='data_prefetcher_all_3_2.out')
    run(func, runs=1, path='preloader_models_3_2.out', model='gcn')
    # run(func, runs=1, path='preloader_datasets_3_2.out', dataset=EXP_DATASET)
    # run(func, runs=1, path='preloader_modes_3_2.out', mode=MODES)
    # run(func, runs=1, path='preloader_relative_batch_sizes_3_2.out', relative_batch_size=EXP_RELATIVE_BATCH_SIZE)