import os
import torch
import traceback
import numpy as np
import pandas as pd
from joblib import load
from collections import defaultdict
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
from neuroc_pygs.configs import EXP_DATASET, ALL_MODELS, EXP_RELATIVE_BATCH_SIZE, MODES, PROJECT_PATH
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer

memory_limit = {
    'gcn': 6.5, # 6,979,321,856
    'gat': 8  # 8,589,934,592
}
dir_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_motivation_datasets')

def train(model, data, train_loader, optimizer, args, df, cnt):
    model = model.to(args.device) # special
    device = args.device
    model.train()
    memory = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]
    torch.cuda.reset_max_memory_allocated(device)
    
    loader_iter, loader_num = iter(train_loader), len(train_loader)
    for i in range(loader_num):
        # task1
        optimizer.zero_grad()
        batch = next(loader_iter)
        # task2
        batch = batch.to(device)
        node, edge = batch.x.shape[0], batch.edge_index.shape[1]
        reg = load(dir_path + f'/{args.model}_linear_model_v1.pth')
        memory_pre = reg.predict([[node, edge]]) / 1024
        if memory_pre > memory_limit[args.model]:
            print(f'{node}, {edge}, {memory_pre * 1024 * 1024 * 1024}, pass')
            continue
        df['nodes'].append(node)
        df['edges'].append(edge)
        # task3
        logits = model(batch.x, batch.edge_index)
        loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        acc = model.evaluator(logits[batch.train_mask], batch.y[batch.train_mask])
        optimizer.step()
        print(f'batch {i}, train_acc: {acc:.4f}, train_loss: {loss.item():.4f}')
        df['memory'].append(torch.cuda.memory_stats(device)["allocated_bytes.all.peak"])
        torch.cuda.reset_max_memory_allocated(device)
        torch.cuda.empty_cache()
        cnt += 1
        if cnt >= 40:
            break
    return df, cnt


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


def run_one(file_name, args, model, data, optimizer):
    data = data.to('cpu')
    train_loader = build_train_loader(args, data)
    torch.cuda.reset_max_memory_allocated(args.device) # 避免dataloader带来的影响
    torch.cuda.empty_cache()
    print(file_name)
    base_memory = torch.cuda.memory_stats(args.device)["allocated_bytes.all.current"]
    print(f'base memory: {base_memory}')
    real_path = os.path.join(PROJECT_PATH, 'sec5_memory/exp_motivation', file_name) + '.csv'
    if os.path.exists(real_path):
        res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')
    else:
    # if True:
        try:
            print('start...')
            res = defaultdict(list)
            cnt = 0
            for _ in range(20):
                res, cnt = train(model, data, train_loader, optimizer, args, res, cnt)
                if cnt >= 20:
                    break
            pd.DataFrame(res).to_csv(real_path)
        except Exception as e:
            print(e.args)
            print("======")
            print(traceback.format_exc())
    peak_memory = list(map(lambda x: x / (1024 * 1024 * 1024), res['memory']))
    print(f'max: {max(peak_memory)}, min: {np.min(peak_memory)}, medium: {np.median(peak_memory)}, diff: {max(peak_memory)-min(peak_memory)}')
    return


def run_all(exp_datasets=EXP_DATASET, exp_models=ALL_MODELS, exp_modes=MODES, exp_relative_batch_sizes=EXP_RELATIVE_BATCH_SIZE):
    args = get_args()
    print(f"device: {args.device}")
    for exp_data in ['yelp', 'reddit']:
        args.dataset = exp_data
        data = build_dataset(args)
        print('build data success')
        for exp_model in ['gcn', 'gat']:
            args.model = exp_model
            data = data.to('cpu')
            model, optimizer = build_model_optimizer(args, data)
            print(model)
            args.mode = 'cluster'
            if True:
                re_bs = [175, 180, 185]
                for rs in re_bs:
                    args.batch_partitions = rs
                    file_name = '_'.join([args.dataset, args.model, str(rs), args.mode, 'linear_model_v1'])
                    run_one(file_name, args, model, data, optimizer)       
    return


def test_run_one():
    args = get_args()
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model = model.to(args.device)
    file_name = '_'.join([args.dataset, args.model, str(args.relative_batch_size), args.mode])
    run_one(file_name, args, model, data, optimizer)


if __name__ == '__main__':
    import sys
    default_args = '--hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32'
    sys.argv = [sys.argv[0], '--device', 'cuda:2', '--num_workers', '0'] + default_args.split(' ')
    run_all()