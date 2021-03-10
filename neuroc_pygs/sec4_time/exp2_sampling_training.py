import os
import copy
import time 
import traceback
import numpy as np
import pandas as pd

from tabulate import tabulate
from torch_geometric.data import Data
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer, build_train_loader
from neuroc_pygs.configs import ALL_MODELS, MODES, EXP_DATASET, PROJECT_PATH, EXP_RELATIVE_BATCH_SIZE
from neuroc_pygs.samplers.cuda_prefetcher import CudaDataLoader


def train(model, optimizer, data, loader, device, mode, df, non_blocking=False):
    model.reset_parameters()
    model.train()
    all_loss = []
    loader_num, loader_iter = len(loader), iter(loader)
    for _ in range(loader_num):
        t0 = time.time()
        if mode == 'cluster':
            batch = next(loader_iter)
            t1 = time.time()
            batch = batch.to(device, non_blocking=non_blocking)
            t2 = time.time()
            batch_size = batch.train_mask.sum().item()
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
        elif mode == 'graphsage':
            batch_size, n_id, adjs = next(loader_iter)
            x, y = data.x[n_id], data.y[n_id[:batch_size]]
            t1 = time.time()
            x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
            adjs = [adj.to(device, non_blocking=non_blocking) for adj in adjs]
            t2 = time.time()
            # task3
            optimizer.zero_grad()
            logits = model(x, adjs)
            loss = model.loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        all_loss.append(loss.item() * batch_size)
        t3 = time.time()
        df.append([t1 - t0, t2 - t1, t3 - t2])
        # print(f'Batch {_}: sampling time: {t1-t0}, to_time: {t2-t1}, training_time: {t3-t2}')
    return np.sum(all_loss) / int(data.train_mask.sum())


def run_all(file_name='sampling_training_final_v2', dir_name='sampling_train', datasets=EXP_DATASET, models=ALL_MODELS, rses=[None]+EXP_RELATIVE_BATCH_SIZE,
         modes=MODES, pin_memorys=[True, False], workers=list(range(0, 41, 10)), non_blockings=[True, False]):
    args = get_args()
    print(args)

    headers = ['Name', 'Base Sampling', 'Base Transfer', 'Base Training', 'Opt Sampling', 'Opt Transfer', 'Opt Training', 'Base max', 'Base min', 'Opt max', 'Opt min', 'Ratio(%)']
    for exp_data in datasets:
        args.dataset = exp_data
        data = build_dataset(args)
        print(f'begin {args.dataset} dataset ...')
        data_path = os.path.join(PROJECT_PATH, 'sec4_time', 'exp_res', file_name + f'_{args.dataset}.csv')
        print(data_path)
        if os.path.exists(data_path):
            continue
        tab_data = []
        for exp_model in models: # 4
            args.model = exp_model
            model, optimizer = build_model_optimizer(args, data)
            for exp_rs in rses: # 6
                if exp_rs != None:
                    exp_rs = float(exp_rs)
                args.relative_batch_size = exp_rs
                for exp_mode in modes: # 2
                    args.mode = exp_mode

                    model = model.to(args.device) 
                    for pin_memory in pin_memorys: # 2
                        args.pin_memory = pin_memory
                        file_name = f'{args.dataset}_{args.model}_{args.relative_batch_size}_{args.mode}_pin_memory_{args.pin_memory}'
                        real_path = os.path.join(PROJECT_PATH, 'sec4_time', 'exp_res', dir_name, file_name + '.csv')
                        if True:
                            tmp_data = []
                            for num_workers in workers: # 4
                                args.num_workers = num_workers
                                for non_blocking in non_blockings: # 2
                                    cur_name = file_name + f'_num_workers_{args.num_workers}_non_blocking_{non_blocking}'
                                    print(cur_name)
                                    try:
                                        loader = build_train_loader(args, data)
                                        opt_loader = CudaDataLoader(copy.deepcopy(loader), device=args.device)
                                        loader_num = len(loader)
                                        base_times, opt_times = [], []
                                        for _ in range(50):
                                            if _ * loader_num >= 50: break
                                            train(model, optimizer, data, loader, args.device, args.mode, base_times, non_blocking=non_blocking)
                                            train(model, optimizer, data, opt_loader, args.device, args.mode, opt_times, non_blocking=non_blocking)
                                        
                                        base_times, opt_times = np.array(base_times), np.array(opt_times)
                                        avg_base_times, avg_opt_times, base_all_times, opt_all_times = np.mean(base_times, axis=0), np.mean(opt_times, axis=0), np.sum(base_times, axis=1), np.sum(opt_times, axis=1)
                                        base_max_time, base_min_time = np.max(base_all_times), np.min(base_all_times)
                                        base_time, opt_time, opt_max_time, opt_min_time = np.sum(avg_base_times), np.sum(avg_opt_times), np.max(opt_all_times), np.min(opt_all_times)
                                        print(avg_base_times, avg_opt_times)
                                        ratio = 100 * (base_time - opt_time) / base_time
                                        print(f'base time: {base_time}, opt_time: {opt_time}, ratio: {ratio}')
                                        res = [cur_name, avg_base_times[0], avg_base_times[1], avg_base_times[2], avg_opt_times[0], avg_opt_times[1], avg_opt_times[2], base_max_time, base_min_time, opt_max_time, opt_min_time, ratio]
                                        print(','.join([str(r) for r in res]))
                                        tmp_data.append(res)
                                        
                                    except Exception as e:
                                        print(e.args)
                                        print("======")
                                        print(traceback.format_exc())
                        tab_data.extend(tmp_data)

        # pd.DataFrame(tab_data, columns=headers).to_csv(data_path)
        print(tabulate(tab_data, headers=headers, tablefmt='github'))


def do_failed():
    failed = """coauthor-physics_gcn_0.06_cluster, coauthor-physics_gcn_0.1_cluster, 
            coauthor-physics_gat_0.01_graphsage, 
            coauthor-physics_ggnn_0.01_cluster, coauthor-physics_ggnn_0.03_cluster, 
            flickr_gcn_None_graphsage, flickr_gcn_0.01_graphsage, flickr_gcn_0.03_graphsage, 
            flickr_gat_0.01_graphsage, flickr_ggnn_0.01_graphsage, flickr_ggnn_0.03_graphsage, 
            flickr_ggnn_0.06_graphsage, flickr_ggnn_0.1_graphsage, flickr_gaan_0.01_graphsage, 
            flickr_gaan_0.03_graphsage, flickr_gaan_0.06_graphsage, flickr_gaan_0.1_graphsage"""
            
    failed = [x.strip() for x in failed.split(', ')]
    for f in failed:
        xs = f.split('_')
        print(xs)
        data, model, rs, mode = xs
        run_all(file_name='sampling_training_final_v2', dir_name='sampling_training_final_v3', datasets=[data], models=[model], rses=[rs],
        modes=[mode], pin_memorys=[False], workers=[0], non_blockings=[False])
        
        
if __name__ == '__main__':
    import sys
    default_args = '--hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32'
    sys.argv = [sys.argv[0], '--device', 'cuda:1', '--num_workers', '0'] + default_args.split(' ')
    # small_datasets = ['reddit', 'yelp', 'amazon']
    small_datasets = ['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'flickr']
    # run_all(file_name='sampling_training_final_v2', dir_name='sampling_training_final_v2', datasets=small_datasets, models=['gcn', 'gat'], rses=[None],
    #      modes=['cluster', 'graphsage'], pin_memorys=[False], workers=[0], non_blockings=[False])
    run_all(file_name='sampling_training_tmp12', dir_name='sampling_training_final_v3', datasets=['coauthor-physics'], models=['gat'], rses=[None],
         modes=['cluster'], pin_memorys=[False], workers=[0], non_blockings=[False])


