import os
import gc
import torch
import time
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
dir_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_automl_datasets_diff')
dir_out = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp5_thesis_motivation')
ratio_dict = pd.read_csv(dir_path + '/regression_mape_res.csv', index_col=0)
linear_ratio_dict = pd.read_csv(dir_path + '/regression_linear_mape_res.csv', index_col=0)


def train(model, data, train_loader, optimizer, args, df, cnt):
    model = model.to(args.device) 
    device = args.device
    model.train()

    loader_iter, loader_num = iter(train_loader), len(train_loader)
    copy_loader = iter(train_loader)
    for i in range(loader_num):
        torch.cuda.reset_max_memory_allocated(device)
        torch.cuda.empty_cache()
        current_memory = torch.cuda.memory_stats(device)["allocated_bytes.all.current"]
        optimizer.zero_grad()
        batch = next(loader_iter)
        node, edge = batch.x.shape[0], batch.edge_index.shape[1]

        # 指定使用哪种内存开销预测模型
        if args.predict_model == 'linear_model':
            reg = load(dir_path + f'/{args.model}_{args.dataset}_{args.predict_model}_diff_v2.pth')
            memory_pre = reg.predict([[node, edge]])[0]
        else:
            reg = load(dir_path + f'/{args.model}_automl_diff_v2.pth')
        if args.predict_model == 'linear_model':
            memory_pre = reg.predict([[node, edge]])[0]
        else:
            paras_dict = model.get_hyper_paras()
            memory_pre = reg.predict([[node, edge] + [v for v in paras_dict.values()]])[0]
        if memory_pre / (1 - args.memory_ratio) + current_memory > memory_limit[args.model] * 1024 * 1024 * 1024:
            print(f'{node}, {edge}, {memory_pre}, pass')
            xt1 = time.time()
            batch = next(copy_loader)
            xt2 = time.time()
            df['resampling'].append(xt2-xt1)
        t2 = time.time()
        # task3
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        acc = model.evaluator(logits[batch.train_mask], batch.y[batch.train_mask])
        optimizer.step()
        st1 = time.time()
        df['nodes'].append(node)
        df['edges'].append(edge)
        memory = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]
        df['acc'].append(acc)
        df['loss'].append(loss.item())
        df['memory'].append(memory)
        st2 = time.time()
        df['overhead'].append(t2 - t1) # 增加了overhead的测量
        df['info'].append(st2 - st1)
        # print(f'batch {i}, train loss: {loss.item()}, train acc: {acc}')
        # print(f'batch {i}, nodes:{node}, edges: {edge}, predict: {memory_pre}-{memory_pre/(1 - args.memory_ratio) + current_memory}, real: {memory}')
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


def run_one(file_name, args):
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model = model.to(args.device)
    
    train_loader = build_train_loader(args, data)
    torch.cuda.reset_max_memory_allocated(args.device) # 避免dataloader带来的影响
    torch.cuda.empty_cache()
    print(file_name)
    base_memory = torch.cuda.memory_stats(args.device)["allocated_bytes.all.current"]
    print(f'base memory: {base_memory}')
    real_path = dir_out + '/' + file_name + '.csv'
    if os.path.exists(real_path):
        res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')
    else:
        try:
            print('start...')
            t1 = time.time()
            res = defaultdict(list)
            cnt = 0
            for _ in range(40):
                res, cnt = train(model, data, train_loader, optimizer, args, res, cnt)
                if cnt >= 40:
                    break
            # pd.DataFrame(res).to_csv(real_path)
            t2 = time.time()
            total_time, overhead, resampling, info = t2 - t1, np.sum(res['overhead']), np.sum(res['resampling']), np.sum(res['info'])
            predict_time = overhead - resampling
            print(f"total time: {total_time}, overhead: {overhead}, resampling: {resampling}, resampling cnt: {len(res['resampling'])}, info: {info}")
        except Exception as e:
            print(e.args)
            print("======")
            print(traceback.format_exc())
    peak_memory = list(map(lambda x: x / (1024 * 1024 * 1024), res['memory']))
    # print(f'max: {max(peak_memory)}, min: {np.min(peak_memory)}, medium: {np.median(peak_memory)}, diff: {max(peak_memory)-min(peak_memory)}')
    return


def run_all(predict_model='automl', exp_model='gcn', bias=0.001):
    args = get_args()
    args.predict_model = predict_model
    print(f"device: {args.device}")
    for exp_data in ['yelp', 'reddit']:
        args.dataset = exp_data
        print('build data success')
        args.model = exp_model
        if predict_model == 'linear_model':
            args.memory_ratio = linear_ratio_dict[exp_model][exp_data] + bias
        else:
            args.memory_ratio = ratio_dict[model][predict_model] + bias
        if exp_data == 'reddit' and exp_model == 'gat':
            re_bs = [170, 175, 180]
        else:
            re_bs = [175, 180, 185]
        for rs in re_bs:
            args.batch_partitions = rs
            file_name = '_'.join([args.dataset, args.model, str(rs), args.predict_model, str(int(100*args.memory_ratio)), 'mape_diff_v3'])
            t1 = time.time()
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
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
    default_args = '--hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32'
    sys.argv = [sys.argv[0], '--device', 'cuda:2', '--num_workers', '0'] + default_args.split(' ')
    for model in ['gat']:
        for predict_model in ['linear_model']:
            run_all(predict_model, model)
