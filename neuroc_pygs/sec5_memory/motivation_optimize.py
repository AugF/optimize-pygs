import os
import gc
import torch, time
import traceback
import numpy as np
import pandas as pd
from joblib import load
from collections import defaultdict
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
from neuroc_pygs.configs import EXP_DATASET, ALL_MODELS, EXP_RELATIVE_BATCH_SIZE, MODES, PROJECT_PATH
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer, build_train_loader
from neuroc_pygs.sec6_cutting.cutting_utils import BSearch
from neuroc_pygs.sec6_cutting.cutting_method import cut_by_importance_method, cut_by_random, get_pagerank, get_degree

memory_limit = {
    'gcn': 6.5, # 6,979,321,856
    'gat': 8  # 8,589,934,592
}


def train(model, data, train_loader, optimizer, args, df, cnt):
    model = model.to(args.device) # special
    device = args.device
    model.train()

    backup_loader_iter = iter(train_loader)
    loader_iter, loader_num = iter(train_loader), len(train_loader)

    for i in range(loader_num):
        torch.cuda.reset_max_memory_allocated(device)
        torch.cuda.empty_cache()
        current_memory = torch.cuda.memory_stats(device)["allocated_bytes.all.current"]
        
        optimizer.zero_grad()
        batch = next(loader_iter)

        # !!! 内存超限处理步骤
        t0 = time.time()
        # 重采样处理机制
        success, resampling_cnt = False, 0

        while resampling_cnt < args.resampling_cnt:
            node, edge = batch.x.shape[0], batch.edge_index.shape[1]
            if args.predict_model == 'linear_model':
                memory_pre = args.reg.predict([[node, edge]])[0]
            else:
                paras_dict = model.get_hyper_paras()
                memory_pre = args.reg.predict([[node, edge] + [v for v in paras_dict.values()]])[0]
            if memory_pre / (1 - args.tolerance) > memory_limit[args.model] * 1024 * 1024 * 1024 - current_memory:
                print(f'resampling..., nodes:{node}, edges: {edge}, predict: {memory_pre/(1 - args.tolerance) + current_memory}')
                try:
                    batch = next(backup_loader_iter)
                except StopIteration as e:
                    backup_loader_iter = iter(train_loader)
                    batch = next(backup_loader_iter)
                resampling_cnt += 1               
            else:
                print("success !!!")
                success = True
                break
        
        # 剪枝处理机制
        if not success:
            print("cutting...")
            cutting_nums = args.bsearch.get_cutting_nums(batch.x.shape[0], batch.edge_index.shape[1], args.tolerance, current_memory)
            if args.cutting_method == 'random':
                batch.edge_index = cut_by_random(batch.edge_index, cutting_nums, seed=int(args.cutting_way))
            else:
                batch.edge_index = cut_by_importance(batch.edge_index, cutting_nums, method=args.cutting_method, name=args.cutting_way)
        t1 = time.time()

        batch = batch.to(device)
        node, edge = batch.x.shape[0], batch.edge_index.shape[1]
        logits = model(batch.x, batch.edge_index)
        loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        acc = model.evaluator(logits[batch.train_mask], batch.y[batch.train_mask])
        optimizer.step()
        memory = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]
        
        # 收集样本
        df['nodes'].append(node)
        df['edges'].append(edge)
        df['acc'].append(acc)
        df['loss'].append(loss.item())
        df['memory'].append(memory)
        df['overhead'][0] += t1 - t0
        print(f'batch {i}, train loss: {loss.item()}, train acc: {acc}')
        cnt += 1
        if cnt >= 40:
            break
    return df, cnt


def run_one(file_name, args):
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model = model.to(args.device)
    
    train_loader = build_train_loader(args, data)
    torch.cuda.reset_max_memory_allocated(args.device) # 避免dataloader带来的影响
    torch.cuda.empty_cache()

    base_memory = torch.cuda.memory_stats(args.device)["allocated_bytes.all.current"]
    real_path = f'out_{args.predict_model}_res/' + file_name + '.csv'
    print(real_path)
    if os.path.exists(real_path):
        res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')
    else:
        try:
            # 面向超参数确定场景；这里其他超参数给定的，否则其他情况下需要说明所有超参数
            if args.predict_model == 'linear_model': 
                args.tolerance = float(open(f'out_linear_model_pth/{args.dataset}_{args.model}_{args.batch_partitions}_linear_model.txt').read()) + args.bias
                args.reg = load(f'out_linear_model_pth/{args.dataset}_{args.model}_{args.batch_partitions}_linear_model.pth')
                args.bsearch = BSearch(clf=args.reg, memory_limit=memory_limit[args.model])
            
            print('start...')
            st1 = time.time()
            res = defaultdict(list)
            res['overhead'] = [0] + [None] * 39
            cnt = 0
            for _ in range(40):
                res, cnt = train(model, data, train_loader, optimizer, args, res, cnt)
                if cnt >= 40:
                    break
            st2 = time.time()
            res['total_time'] = [st2 - st1] + [None] * 39
            pd.DataFrame(res).to_csv(real_path)
            peak_memory = list(map(lambda x: x / (1024 * 1024 * 1024), res['memory']))
            print(f'max: {max(peak_memory)}, min: {np.min(peak_memory)}, medium: {np.median(peak_memory)}, diff: {max(peak_memory)-min(peak_memory)}')
        except Exception as e:
            print(e.args)
            print("======")
            print(traceback.format_exc())
    return


def run_all(predict_model='linear_model', exp_model='gcn', bias=0.001):
    args = get_args()
    args.model = exp_model
    args.predict_model, args.bias = predict_model, bias
    # 设置默认的参数
    args.resampling_cnt, args.cutting_method, args.cutting_way = 10, 'random', 2
    print(f"device: {args.device}")

    if predict_model == 'random_forest': # 面向超参数不确定场景
        args.tolerance = float(open(f'out_random_forest_pth/{args.model}_random_forest.txt').read()) + args.bias
        args.reg = load(f'out_random_forest_pth/{args.model}_random_forest.pth')
        args.bsearch = BSearch(clf=args.reg, memory_limit=memory_limit[args.model])
    
    for exp_data in ['yelp', 'reddit']:
        args.dataset = exp_data
        print('build data success')
        if exp_data == 'reddit' and exp_model == 'gat':
            re_bs = [170, 175, 180]
        else:
            re_bs = [175, 180, 185]
        for rs in re_bs:
            args.batch_partitions = rs
            file_name = '_'.join([args.dataset, args.model, str(rs), args.predict_model])
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
    sys.argv = [sys.argv[0], '--device', 'cuda:0', '--num_workers', '0'] + default_args.split(' ')
    for predict_model in ['linear_model', 'random_forest']:
        for model in ['gcn', 'gat']:
            run_all(predict_model, model)
