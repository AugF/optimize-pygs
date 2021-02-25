import copy
import time
import torch
import numpy as np
from neuroc_pygs.options import get_args, build_dataset, build_loader, build_model


def train(model, data, train_loader, optimizer, mode, device, print_flag=False):
    model.train()
    all_loss = []
    loader_iter, loader_num = iter(train_loader), len(train_loader)
    for i in range(loader_num):
        if mode == "cluster":
            # task1
            t1 = time.time()
            batch = next(loader_iter)
            t2 = time.time()
            # task2
            batch = batch.to(device)
            # task3
            t3 = time.time()
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            if print_flag:
                print(f"batch {i} use time task1: {(t2-t1):.6f}, task2: {(t3-t2):.6f}, task3: {(time.time()-t3):.6f}")
        else:
            # task1
            batch_size, n_id, adjs = next(loader_iter)
            x, y = data.x[n_id], data.y[n_id[:batch_size]]
            # task2
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            adjs = [adj.to(device) for adj in adjs]
            # task3
            logits = model(x, adjs)
            loss = model.loss_fn(logits, y)
            loss.backward()
        optimizer.step()
        all_loss.append(loss.item())
    return np.mean(all_loss)


def opt_train(model, data, train_loader, optimizer, mode, device, print_flag=False):
    loader_iter, loader_num = iter(train_loader), len(train_loader)
    if mode == "cluster":
        return cluster_train_pipeline(loader_num, loader_iter, device, optimizer, model)


# 类
def cluster_train_pipeline(loader_num, loader_iter, device, optimizer, model):
    # https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39
    next_batch = next(loader_iter)
    next_batch = next_batch.to(device)
    for i in range(loader_num):
        # task1
        batch = next_batch
        if i + 1 != loader_num:
            next_batch = next(loader_iter)
            # task2
            next_batch = next_batch.to(device)
        # task3
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()


def run(print_flag, **kwargs):
    args = get_args()
    for key, value in kwargs.items():
        if key == 'print_flag' or key in args.__dict__.keys():
            args.__setattr__(key, value)
    print(f'num_workers: {args.num_workers}, pin_memory: {args.pin_memory}, print_flag: {print_flag}')
    data = build_dataset(args)
    train_loader, subgraph_loader = build_loader(args, data)
    model, optimizer = build_model(args, data) 
    model, data = model.to(args.device), data.to(args.device)
    # begin test
    t1 = time.time()
    _ = train(model, data, train_loader, optimizer, args.mode, args.device, print_flag)
    t2 = time.time()
    _ = opt_train(model, data, train_loader, optimizer, args.mode, args.device, print_flag)
    t3 = time.time()
    base_time, opt_time = t2 - t1, t3 - t2
    ratio = 100 * (base_time - opt_time) / base_time
    return base_time, opt_time, ratio


def test_loader_paras():
    for num_workers in [0, 2, 5, 10, 20, 40]:
        for pin_memory in [True, False]:
            print(f"num_workers: {num_workers}, pin_memory: {pin_memory}, time: {run(num_workers, pin_memory)}")


def get_print():
    res = run(num_workers=10, pin_memory=True, print_flag=True)
    print('base_time: {}, opt_time: {}, ratio: {}'.format(*res))

if __name__ == "__main__":
    # test_loader_paras()
    get_print()
