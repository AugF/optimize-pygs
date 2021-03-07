import os
import time 
import traceback
import numpy as np

from tabulate import tabulate
from torch_geometric.data import Data
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer, build_train_loader
from neuroc_pygs.configs import ALL_MODELS, MODES, EXP_DATASET, PROJECT_PATH, EXP_RELATIVE_BATCH_SIZE
from neuroc_pygs.samplers.cuda_prefetcher import CudaDataLoader


def train(model, optimizer, data, loader, device, mode, non_blocking=False):
    model.reset_parameters()
    model.train()
    all_acc, all_loss = [], []
    for batch in loader:
        if mode == 'cluster':
            optimizer.zero_grad()
            batch = batch.to(device, non_blocking=non_blocking)
            logits = model(batch.x, batch.edge_index)
            loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            acc = model.evaluator(logits[batch.train_mask], batch.y[batch.train_mask])
            optimizer.step()
        elif mode == 'graphsage':
            batch_size, n_id, adjs = batch
            x, y = data.x[n_id], data.y[n_id[:batch_size]]
            x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
            adjs = [adj.to(device, non_blocking=non_blocking) for adj in adjs]
            # task3
            logits = model(x, adjs)
            loss = model.loss_fn(logits, y)
            loss.backward()
            acc = model.evaluator(logits, y) / batch_size
            optimizer.step()
        all_loss.append(loss.item())
        all_acc.append(acc)
    return np.mean(all_acc), np.mean(all_loss)


def train_cuda(model, optimizer, data, loader, device, mode, non_blocking=False):
    model.reset_parameters()
    model.train()
    all_acc, all_loss = [], []
    for batch in loader:
        if mode == 'cluster':
            optimizer.zero_grad()
            # batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            acc = model.evaluator(logits[batch.train_mask], batch.y[batch.train_mask])
            optimizer.step()
        elif mode == 'graphsage':
            batch_size, n_id, adjs = batch
            x, y = data.x[n_id], data.y[n_id[:batch_size]]
            x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
            logits = model(x, adjs)
            loss = model.loss_fn(logits, y)
            loss.backward()
            acc = model.evaluator(logits, y) / batch_size
            optimizer.step()
        all_loss.append(loss.item())
        all_acc.append(acc)
    return np.mean(all_acc), np.mean(all_loss)



args = get_args()
args.pin_memory = False
print(args)

tab_data = []
tab_data.append(['Name', 'Baseline', 'Opt', 'Ratio(%)'])
for exp_data in ['pubmed']:
    args.dataset = exp_data
    data = build_dataset(args)
    for exp_model in ALL_MODELS:
        args.model = exp_model
        model, optimizer = build_model_optimizer(args, data)
        for exp_rs in [None] + EXP_RELATIVE_BATCH_SIZE:
            args.relative_batch_size = exp_rs
            for exp_mode in ['graphsage']:
                args.mode = exp_mode

                model = model.to(args.device)
                cur_name = f'{args.dataset}_{args.model}_{args.mode}_{args.relative_batch_size}'
                print(cur_name)

                # opt
                args.num_workers = 0
                loader = build_train_loader(args, data)
                st1 = time.time()
                opt_loader = CudaDataLoader(loader, device=args.device, sampler=args.mode)
                st2 = time.time()
                opt_st = st2 - st1
                opt_times = []
                for non_blocking in [True, False]:
                    print(f'non_blocking={non_blocking}')
                    t1 = time.time()
                    train(model, optimizer, data, opt_loader, args.device, args.mode, non_blocking)
                    t2 = time.time()
                    opt_time = t2 - t1
                    print(f'opt_time: {opt_time}')
                    opt_times.append(opt_time)

                # baseline
                base_times = []
                for num_workers in range(0, 41, 10):
                    args.num_workers = num_workers
                    for non_blocking in [True, False]:
                        print(f'num_workers: {num_workers}, non_blocking={non_blocking}')
                        loader = build_train_loader(args, data)
                        t1 = time.time()
                        train(model, optimizer, data, loader, args.device, args.mode, non_blocking)
                        t2 = time.time()
                        base_time = t2 - t1
                        print(f'base_time: {base_time}')
                        base_times.append(base_time)
                
                best_opt_time, best_base_time = min(opt_times), min(base_times)
                res = [cur_name, best_base_time, best_opt_time, 100 * (best_base_time - best_opt_time) / best_base_time]
                print(res)
                tab_data.append(res)


np.save(os.path.join(PROJECT_PATH, 'sec4_time', 'exp_res', 'sampling_train_final.npy'), np.array(tab_data))
print(tabulate(tab_data[1:], headers=tab_data[0], tablefmt='github'))
