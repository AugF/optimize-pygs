import os, sys
import gc
import torch
import traceback
import numpy as np
import pandas as pd
from collections import defaultdict
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
from neuroc_pygs.configs import EXP_DATASET, ALL_MODELS, EXP_RELATIVE_BATCH_SIZE, MODES, PROJECT_PATH
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer, build_train_loader


def train(model, data, train_loader, optimizer, args, df, cnt):
    model = model.to(args.device) # special
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
        
        # ！！！内存超限处理步骤
        # 重采样处理机制
        resampling_cnt = 0
        success = False
        while success and resampling_cnt < args.resampling_cnt:
            # 这里应该是具体的配置
            paras_dict = model.get_hyper_paras()
            if args.predict_model == 'linear_model':
                reg = load(f'out_linear_model_pth/{args.model}_{args.predict_model}.pth')
                memory_pre = reg.predict([[node, edge]])[0]
            else:
                reg = load(f'out_random_forest_pth/{args.model}_random_forest.pth')
                memory_pre = reg.predict([[node, edge] + [v for v in paras_dict.values()]])[0]
            if memory_pre / (1 - args)
                    
        # 剪枝处理机制
        if not success:
            cutting_nums = bsearch.get_cutting_nums(node, edge, real_memory_ratio, current_memory)
            if args.cutting_method == 'random':
                edge_index = cut_by_random(edge_index, cutting_nums, seed=int(args.cutting_way))
            else:
                edge_index = cut_by_importance_reverse(edge_index, cutting_nums, method=args.cutting_method, name=args.cutting_way)
        
        batch = batch.to(device)
        node, edge = batch.x.shape[0], batch.edge_index.shape[1]
        logits = model(batch.x, batch.edge_index)
        loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        acc = model.evaluator(logits[batch.train_mask], batch.y[batch.train_mask])
        optimizer.step()
        df['nodes'].append(node)
        df['edges'].append(edge)
        memory = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]
        # ?
        df['acc'].append(acc)
        df['loss'].append(loss.item())
        df['memory'].append(memory)
        df['diff_memory'].append(memory - current_memory)
        print(f'batch {i}, train loss: {loss.item()}, train acc: {acc}')
        print(f'batch {i}, nodes:{node}, edges: {edge}, memory: {memory}, diff_memory: {memory-current_memory}')
        cnt += 1
        if cnt >= max_cnt: 
            break
    return df, cnt


# 推理
@torch.no_grad()
def infer(model, data, subgraphloader):
    model.eval()
    model.reset_parameters()
    y_pred = model.inference_cuda(data.x, subgraphloader) # 这里使用inference_cuda作为测试
    y_true = data.y.cpu()

    accs, losses = [], []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        loss = model.loss_fn(y_pred[mask], y_true[mask])
        acc = model.evaluator(y_pred[mask], y_true[mask]) 
        losses.append(loss.item())
        accs.append(acc)
    return accs, losses


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
    real_path = dir_path + f'/{file_name}.csv'
    if os.path.exists(real_path):
        res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')
    else:
        try:
            print('start...')
            res = defaultdict(list)
            cnt = 0
            for _ in range(max_cnt):
                res, cnt = train(model, data, train_loader, optimizer, args, res, cnt)
                if cnt >= max_cnt:
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
    # for exp_data in ['reddit']:
        args.dataset = exp_data
        print('build data success')
        for exp_model in ['gcn', 'gat']:
            args.model = exp_model
            if exp_data == 'reddit' and exp_model == 'gat':
                re_bs = [170, 175, 180]
            else:
                re_bs = [175, 180, 185]
            for rs in re_bs:
                args.batch_partitions = rs
                file_name = '_'.join([args.dataset, args.model, str(rs), args.mode])
                run_one(file_name, args)
                gc.collect()           
    return


def collect_motivation_info():
    dir_path = os.path.join(PROJECT_PATH, 'sec5_memory/out_motivation_csv')
    max_cnt = 40
    default_args = '--hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32'
    sys.argv = [sys.argv[0], '--device', 'cuda:0', '--num_workers', '0'] + default_args.split(' ')
    run_all()


def build_linear_model_dataset():
    dir_path = os.path.join(PROJECT_PATH, 'sec5_memory/out_linear_model_csv')
    max_cnt = 120    
    default_args = '--hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32'
    sys.argv = [sys.argv[0], '--device', 'cuda:0', '--num_workers', '0'] + default_args.split(' ')
    run_all()
    
    # 185单独计算
    default_args = '--hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32 --model gat --data reddit --batch_partitions 185'
    sys.argv = [sys.argv[0], '--device', 'cuda:0', '--num_workers', '0'] + default_args.split(' ')
    args = get_args()
    file_name = '_'.join([args.dataset, args.model, str(args.batch_partitions), args.mode])
    run_one(file_name, args)


if __name__ == '__main__':
    collect_motivation_info()
    # build_linear_model_dataset()