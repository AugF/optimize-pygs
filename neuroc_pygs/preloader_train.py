import time
import numpy as np
import os.path as osp
from neuroc_pygs.options import run_all
from neuroc_pygs.utils import to, BatchLogger
from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.train_step import train


def opt_train(model, data, train_loader, optimizer, mode, args):
    loader_iter, loader_num = iter(train_loader), len(train_loader)
    if mode == "cluster":
        return cluster_train_pipeline(loader_num, loader_iter, optimizer, model, args)
    else:
        return graphsage_train_pipeline(loader_num, loader_iter, optimizer, model, data, args)

# 类
def cluster_train_pipeline(loader_num, loader_iter, optimizer, model, args):
    # https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39
    device = args.device
    model.train()
    all_loss, all_acc = [], []
    next_batch = next(loader_iter)
    next_batch = to(next_batch, device, non_blocking=True)
    for i in range(loader_num):
        batch = next_batch
        if i + 1 != loader_num:
            # task1
            next_batch = next(loader_iter)
            # task2
            next_batch = to(next_batch, device, non_blocking=True)
        # task3
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        acc = model.evaluator(logits[batch.train_mask], batch.y[batch.train_mask])

        all_loss.append(loss.item())
        all_acc.append(acc)
    return np.mean(all_acc), np.mean(all_loss)


def graphsage_train_pipeline(loader_num, loader_iter, device, optimizer, model, data, args):
    # https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39
    all_loss, all_acc = [], []
    
    next_batch_size, n_id, next_adjs = next(loader_iter)
    next_x, next_y = data.x[n_id], data.y[n_id[:next_batch_size]]
    next_adjs = [to(adj, device, non_blocking=True) for adj in next_adjs]
    x, y = to(x, device, non_blocking=True), to(y, device, non_blocking=True)

    for i in range(loader_num):
        # task1
        batch_size, adjs, x, y = next_batch_size, next_adjs, next_x, next_y
        if i + 1 != loader_num:
            # task1
            next_batch_size, n_id, next_adjs = next(loader_iter)
            next_x, next_y = data.x[n_id], data.y[n_id[:next_batch_size]]
            # task2
            next_adjs = [to(adj, device, non_blocking=True) for adj in next_adjs]
            x, y = to(x, device, non_blocking=True), to(y, device, non_blocking=True)
        # task3
        optimizer.zero_grad()
        logits = model(x, adjs)
        loss = model.loss_fn(logits, y)
        loss.backward()
        acc = model.evaluator(logits, y) / batch_size
        all_loss.append(loss.item())
        all_acc.apppend(acc)
    return np.mean(all_loss), np.mean(all_loss)


def func(data, train_loader, subgraph_loader, model, optimizer, args):
    model, data = model.to(args.device), data.to(args.device)
    # begin test
    t1 = time.time()
    train_acc = train(model, data, train_loader, optimizer, args.mode, args.device, log_batch=args.log_batch)
    t2 = time.time()
    train_acc = opt_train(model, data, train_loader, optimizer, args.mode, args.device, log_batch=args.log_batch)
    t3 = time.time()
    base_time, opt_time = t2 - t1, t3 - t2
    ratio = 100 * (base_time - opt_time) / base_time
    return base_time, opt_time, ratio


if __name__ == '__main__':
    from neuroc_pygs.configs import ALL_MODELS, EXP_DATASET, MODES, EXP_RELATIVE_BATCH_SIZE
    from neuroc_pygs.options import run
    run(func, runs=1, path='preloader_models.out', model=ALL_MODELS)
    run(func, runs=1, path='preloader_datasets.out', dataset=EXP_DATASET)
    run(func, runs=1, path='preloader_modes.out', mode=MODES)
    run(func, runs=1, path='preloader_relative_batch_sizes.out', mode=EXP_RELATIVE_BATCH_SIZE)