import time
import traceback
import copy
import torch
import numpy as np
from collections import defaultdict
from neuroc_pygs.sec4_time.epoch_utils import train, infer
from neuroc_pygs.samplers.cuda_prefetcher import CudaDataLoader
from neuroc_pygs.options import get_args, build_dataset, build_subgraphloader, build_model_optimizer
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler

def build_train_loader(args, data, Cluster_Loader=ClusterLoader, Neighbor_Loader=NeighborSampler):
    if args.relative_batch_size:
        args.batch_size = int(data.x.shape[0] * args.relative_batch_size)
        args.batch_partitions = int(args.cluster_partitions * args.relative_batch_size)
    if args.mode == 'cluster':
        cluster_data = ClusterData(data, num_parts=args.cluster_partitions, recursive=False,
                                   save_dir=args.cluster_save_dir)
        train_loader = Cluster_Loader(cluster_data, batch_size=args.batch_partitions, shuffle=True,
                                     num_workers=args.num_workers, pin_memory=args.pin_memory)
    elif args.mode == 'graphsage': # sizes
        train_loader = Neighbor_Loader(data.edge_index, node_idx=None,
                                       sizes=args.sizes, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=args.pin_memory)
    return train_loader


# for exp_data in ['amazon-computers']:
exp_data = 'pubmed'
max_cnt = 50
for exp_model in ['gcn']:
    for exp_mode in ['cluster', 'graphsage']:
        # for rs in [0.01, 0.03, 0.06, 0.1, 0.25, 0.5]:
        # for e in [5, 10, 20, 50, 100, 150, 200]:
        for var in [16, 64, 256, 1024, 10240, 25600]:
            try:
                # exp_data = f'random_100k_{e}k'
                args = get_args()
                args.num_workers = 0
                args.dataset, args.model, args.mode = exp_data, exp_model, exp_mode
                args.gaan_hidden_dims, args.hidden_dims = var, var
                # args.relative_batch_size = rs
                print(args)
                data = build_dataset(args)
                train_loader = build_train_loader(args, data)
                subgraph_loader = build_subgraphloader(args, data)
                if args.mode == 'graphsage':
                    opt_train_loader = CudaDataLoader(copy.deepcopy(train_loader), device=args.device, sampler='graphsage', data=data)
                else:
                # if True:
                    opt_train_loader = CudaDataLoader(copy.deepcopy(train_loader), device=args.device)

                model, optimizer = build_model_optimizer(args, data)
                model = model.to(args.device)
                model.reset_parameters()

                df1, df2 = defaultdict(list), defaultdict(list)
                df1['cnt'], df2['cnt'] = [0], [0]
                df1['max_cnt'], df2['max_cnt'] = [max_cnt], [max_cnt]
                # warm up
                train(model, optimizer, data, copy.deepcopy(train_loader), args.device, args.mode)
                #
                t1 = time.time()
                for _ in range(100):
                    train(model, optimizer, data, train_loader, args.device, args.mode, non_blocking=False, df=df1)
                    if df1['cnt'][0] >= var:
                        break
                t2 = time.time()
                for _ in range(100):
                    train(model, optimizer, data, opt_train_loader, args.device, args.mode, non_blocking=False, df=df2, opt_flag=True)
                    # train(model, optimizer, data, opt_train_loader, args.device, args.mode, non_blocking=False, df=df2)
                    if df2['cnt'][0] >= max_cnt:
                        break
                t3 = time.time()
                baseline, opt = t2 - t1, t3 - t2
                print(f'model: {exp_model}, data: {exp_data}, baseline: {baseline}, opt: {opt}, ratio: {opt/baseline}')
                avg_sample, avg_move, avg_cal = np.mean(df1['sample']), np.mean(df1['move']), np.mean(df1['cal'])
                y, z = avg_sample / (avg_sample + avg_move + avg_cal), avg_move / (avg_sample + avg_move + avg_cal)
                exp_ratio = 1 / var + (var - 1) * max(y, z, 1 - y - z) / var
                print(f'Baseline: sample: {avg_sample}, move: {avg_move}, cal: {avg_cal}')
                print(f"Opt: sample: {np.mean(df2['sample'])}, move: {np.mean(df2['move'])}, cal: {np.mean(df2['cal'])}")
                print(f'y: {y}, z: {z}, exp_ratio: {exp_ratio}')
            except Exception as e:
                print(e.args)
                print(traceback.format_exc())
                continue
