import os
import gc
import torch
import traceback
import numpy as np
import pandas as pd
from collections import defaultdict
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
from neuroc_pygs.configs import EXP_DATASET, ALL_MODELS, EXP_RELATIVE_BATCH_SIZE, MODES, PROJECT_PATH
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer, build_subgraphloader


def train(model, data, train_loader, optimizer, args, df, cnt):
    model = model.to(args.device) # special
    device, mode = args.device, args.mode
    model.train()

    all_loss, all_acc = 0, 0
    total_examples = 0
    loader_iter, loader_num = iter(train_loader), len(train_loader)
    for i in range(loader_num):
        # task1
        optimizer.zero_grad()
        batch = next(loader_iter)
        batch_size = batch.train_mask.sum().item()
        # task2
        batch = batch.to(device)
        df['nodes'].append(batch.x.shape[0])
        df['edges'].append(batch.edge_index.shape[1])
        # task3
        logits = model(batch.x, batch.edge_index)
        y = batch.y[batch.train_mask]
        loss = model.loss_fn(logits[batch.train_mask], y)
        loss.backward()
        acc = model.evaluator(logits[batch.train_mask], y)
        optimizer.step()
        # print(f'batch {i}, train_acc: {acc:.4f}, train_loss: {loss.item():.4f}')
        df['memory'].append(torch.cuda.memory_stats(device)["allocated_bytes.all.peak"])
        torch.cuda.reset_max_memory_allocated(device)
        torch.cuda.empty_cache()
        all_loss += loss.item() * batch_size
        all_acc += acc * y.size(0)
        total_examples += batch_size + y.size(0)
        cnt += 1
        if cnt >= 40:
            break
    return all_loss / total_examples, all_acc / total_examples, df, cnt


@torch.no_grad()
def infer(model, data, subgraphloader):
    model.eval()
    model.reset_parameters()
    y_pred = model.inference(data.x, subgraphloader) # 这里使用inference_cuda作为测试
    y_true = data.y.cpu()

    # accs, losses = [], []
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        # loss = model.loss_fn(y_pred[mask], y_true[mask])
        acc = model.evaluator(y_pred[mask], y_true[mask]) 
        # losses.append(loss.item())
        accs.append(acc)
    return accs


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


def run_one(file_name, args):
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model = model.to(args.device)
    
    train_loader = build_train_loader(args, data)
    subgraph_loader = build_subgraphloader(args, data)
    torch.cuda.reset_max_memory_allocated(args.device) # 避免dataloader带来的影响
    torch.cuda.empty_cache()
    print(file_name)
    base_memory = torch.cuda.memory_stats(args.device)["allocated_bytes.all.current"]
    print(f'base memory: {base_memory}')
    real_path = os.path.join(PROJECT_PATH, 'sec5_memory/exp_motivation', file_name) + '.csv'
    if os.path.exists(real_path):
        res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')
    else:
        try:
            print('start...')
            res = defaultdict(list)
            cnt = 0
            for _ in range(50):
                loss, acc, res, cnt = train(model, data, train_loader, optimizer, args, res, cnt)
                print(f'Epoch: {_:03d}, train loss: {loss:.6f}, train acc: {acc:.6f}')
                train_acc, val_acc, test_acc = infer(model, data, subgraph_loader)
                print(f'Epoch: {_:03d}, train acc: {train_acc:.6f}, val acc: {val_acc:.6f}, test_acc: {test_acc:.6f}')
                # if cnt >= 20:
                #     break
            exit(0)
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
        print('build data success')
        for exp_model in ['cluster_gcn', 'gat']:
            args.model = exp_model
            re_bs = [175, 180, 185]
            for rs in re_bs:
                args.batch_partitions = rs
                file_name = '_'.join([args.dataset, args.model, str(rs), args.mode, 'copy_v2'])
                run_one(file_name, args)
                gc.collect()           
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