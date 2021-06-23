import time
import copy
import torch
import numpy as np
from collections import defaultdict
from neuroc_pygs.sec4_time.epoch_utils import infer
from neuroc_pygs.samplers.cuda_prefetcher import CudaDataLoader
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler


def build_subgraphloader(args, data, Neighbor_Loader=NeighborSampler):
    if args.relative_batch_size:
        args.batch_size = int(data.x.shape[0] * args.relative_batch_size)
        args.batch_partitions = int(args.cluster_partitions * args.relative_batch_size)

    if args.infer_layer:
        subgraph_loader = Neighbor_Loader(data.edge_index, sizes=[-1], batch_size=args.infer_batch_size,
                                          shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    else:
        subgraph_loader = Neighbor_Loader(data.edge_index, sizes=[-1] * args.layers, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    return subgraph_loader


def run_batch_infer(models=['gcn'], datasets=['pubmed'], exp_vars=[1024]):
    for exp_model in models:
        for exp_data in datasets:
            for var in exp_vars:
                args = get_args()
                args.num_workers = 0
                args.dataset, args.model = exp_data, exp_model
                args.infer_batch_size = var
                print(args)
                data = build_dataset(args)
                subgraph_loader = build_subgraphloader(args, data)
                opt_subgraph_loader = CudaDataLoader(copy.deepcopy(subgraph_loader), device=args.device)

                model, optimizer = build_model_optimizer(args, data)
                model = model.to(args.device)
                model.reset_parameters()

                df1, df2 = defaultdict(list), defaultdict(list)
                df1['cnt'], df2['cnt'] = [0], [0]

                t1 = time.time()
                for _ in range(args.epochs):
                    infer(model, data, subgraph_loader, df_time=df1)
                    if df1['cnt'][0] >= 50:
                        break
                t2 = time.time()
                for _ in range(args.epochs):
                    infer(model, data, opt_subgraph_loader, df_time=df2)
                    if df2['cnt'][0] >= 50:
                        break
                t3 = time.time()
                baseline, opt = t2 - t1, t3 - t2
                print(f'model: {exp_model}, data: {exp_data}, baseline: {baseline}, opt: {opt}, ratio: {opt/baseline}')
                avg_sample, avg_move, avg_cal = np.mean(df1['sample']), np.mean(df1['move']), np.mean(df1['cal'])
                y, z = avg_sample / (avg_sample + avg_move + avg_cal), avg_move / (avg_sample + avg_move + avg_cal)
                cnt = len(df1['sample'])
                exp_ratio = 1 / cnt + (cnt - 1) * max(y, z, 1 - y - z) / cnt
                print(f'Baseline: sample: {avg_sample}, move: {avg_move}, cal: {avg_cal}')
                print(f"Opt: sample: {np.mean(df2['sample'])}, move: {np.mean(df2['move'])}, cal: {np.mean(df2['cal'])}")
                print(f'y: {y}, z: {z}, exp_ratio: {exp_ratio}')


def test_batch_infer():
    run_batch_infer(models=['gcn', 'gat', 'ggnn', 'gaan'], datasets=['flickr'], exp_vars=[1024])
    run_batch_infer(models=['gcn'], datasets=['pubmed', 'amazon-computers', 'flickr', 'reddit'], exp_vars=[1024])
    run_batch_infer(models=['gaan'], datasets=['amazon-computers'], exp_vars=[1024, 2048, 4096, 8192])


if __name__ == '__main__':
    test_batch_infer()