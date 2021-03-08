import os
import time 
import traceback
import numpy as np
import pandas as pd

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


def run_all(file_name='sampling_training_final', dir_name='sampling_train', datasets=EXP_DATASET, models=ALL_MODELS, rses=[None]+EXP_RELATIVE_BATCH_SIZE,
         modes=MODES, pin_memorys=[True, False], workers=list(range(0, 41, 10)), non_blockings=[True, False]):
    args = get_args()
    print(args)

    tab_data = []
    headers = ['Name', 'Baseline', 'Opt', 'Ratio Avg(%)', 'Avg Ratio(%)']

    for exp_data in datasets:
        args.dataset = exp_data
        data = build_dataset(args)
        print(f'begin {args.dataset} dataset ...')
        for exp_model in models: # 4
            args.model = exp_model
            model, optimizer = build_model_optimizer(args, data)
            for exp_rs in rses: # 6
                args.relative_batch_size = exp_rs
                for exp_mode in modes: # 2
                    args.mode = exp_mode

                    model = model.to(args.device) 
                    for pin_memory in pin_memorys: # 2
                        args.pin_memory = pin_memory
                        file_name = f'{args.dataset}_{args.model}_{args.relative_batch_size}_{args.mode}_pin_memory_{args.pin_memory}'
                        real_path = os.path.join(PROJECT_PATH, 'sec4_time', 'exp_res', dir_name, file_name + '.csv')
                        # if os.path.exists(real_path):
                        #     tmp_data = pd.read_csv(real_path, index_col=0).values
                        # else:
                        if True:
                            tmp_data = []
                            for num_workers in workers: # 4
                                args.num_workers = num_workers
                                for non_blocking in non_blockings: # 2
                                    cur_name = file_name + f'_num_workers_{args.num_workers}_non_blocking_{non_blocking}'
                                    print(cur_name)
                                    try:
                                        loader = build_train_loader(args, data)
                                        opt_loader = CudaDataLoader(loader, device=args.device)
                                        base_times, opt_times, ratios = [], [], []
                                        for _ in range(5):
                                            t0 = time.time()
                                            train(model, optimizer, data, loader, args.device, args.mode, non_blocking)
                                            t1 = time.time()
                                            train_cuda(model, optimizer, data, opt_loader, args.device, args.mode, non_blocking)
                                            t2 = time.time()
                                            base_time, opt_time = t1 - t0, t2 - t1
                                            ratio = 100 * (base_time - opt_time) / base_time
                                            base_times.append(base_time)
                                            opt_times.append(opt_time)
                                            ratios.append(ratio)
                                            print(f'base time: {base_time}, opt_time: {opt_time}, ratio: {ratio}')
                                        avg_ratio = 100 * (np.mean(base_times) - np.mean(opt_times)) / np.mean(base_times)
                                        res = [cur_name, np.mean(base_times), np.mean(opt_times), np.mean(ratios), avg_ratio, np.max(ratios)]
                                        print(res)
                                        tmp_data.append(res)
                                        
                                    except Exception as e:
                                        print(e.args)
                                        print("======")
                                        print(traceback.format_exc())
                            # pd.DataFrame(tmp_data, columns=headers).to_csv(real_path)
                        tab_data.extend(tmp_data)

    pd.DataFrame(tmp_data, columns=headers).to_csv(os.path.join(PROJECT_PATH, 'sec4_time', 'exp_res', file_name + '.csv'))
    # np.save(os.path.join(PROJECT_PATH, 'sec4_time', 'exp_res', file_name + '.npy'), np.array(tab_data))
    print(tabulate(tab_data, headers=headers, tablefmt='github'))


if __name__ == '__main__':
    import sys
    sys.argv = [sys.argv[0], '--device', 'cuda:2', '--num_workers', '0']
    small_datasets = ['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    run_all(file_name='sampling_training_small_datasets_0', dir_name='sampling_train_small_datasets_0', datasets=small_datasets, models=ALL_MODELS, rses=[None] + EXP_RELATIVE_BATCH_SIZE,
         modes=['cluster', 'graphsage'], pin_memorys=[False], workers=[0], non_blockings=[False])
