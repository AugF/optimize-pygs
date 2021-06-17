import copy
import math
import sys, os
import time
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from tabulate import tabulate
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer, build_train_loader
from neuroc_pygs.sec4_time.epoch_utils import test_full
from neuroc_pygs.configs import PROJECT_PATH


datasets_maps = {
    'amazon-photo': 'amp',
    'pubmed': 'pub',
    'amazon-computers': 'amc',
    'coauthor-physics': 'cph',
    'flickr': 'fli',
    'com-amazon': 'cam'
}

dir_out = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_train_final')
def train(model, optimizer, data, loader, device, mode, discard_per, best_val_acc, final_test_acc, cnt, df=None, non_blocking=False):
    model.reset_parameters()
    model.train()
    loader_num, loader_iter, backup_loader_iter = len(loader), iter(loader), iter(loader)
    outilers_batches = np.random.choice(range(loader_num), int(loader_num * discard_per))
    for _ in range(loader_num):
        batch = next(loader_iter)
        if _ in outilers_batches:
            print(f'batch {_} resampling!!')
            batch = next(backup_loader_iter)
        batch = batch.to(device, non_blocking=non_blocking)
        batch_size = batch.train_mask.sum().item()
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()

        # 评估
        train_acc, val_acc, test_acc = test_full(model, data)
        if val_acc > best_val_acc:
            patience_step = 0
            best_val_acc = val_acc
            final_test_acc = test_acc
            best_model = copy.deepcopy(model)

        if df is not None:
            df['train_acc'].append(train_acc)
            df['val_acc'].append(val_acc)
            df['test_acc'].append(test_acc)
            df['best_val_acc'].append(best_val_acc)
            df['final_test_acc'].append(final_test_acc)   
        print(f"Batch: {cnt:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Best Val: {best_val_acc:.4f}, Test: {final_test_acc:.4f}")
        cnt += 1
        if cnt >= 40:
            break
    return best_val_acc, final_test_acc, cnt


def epoch(discard_per=0.01, df=None): 
    args = get_args()
    print(args)
    print(f'discard_per: {discard_per}')
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    loader = build_train_loader(args, data)
    model, data = model.to(args.device), data.to(args.device)
    best_val_acc = 0
    final_test_acc = 0
    best_model = None
    patience_step = 0
    cnt = 0
    for epoch in range(args.epochs): # 50
        best_val_acc, final_test_acc, cnt = train(model, optimizer, data, loader, args.device, args.mode, discard_per, best_val_acc, final_test_acc, cnt, df)
        if cnt >= 40:
            break
    return final_test_acc


headers = ['Model', 'Data', 'Per', 'Acc']
config_paras = pd.read_csv("/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec5_memory/exp_res/max_res.csv", index_col=0)


# 不同模型
def run(models, datasets):
    for model in models:
        for data in datasets:
            paras = str(config_paras.loc[datasets_maps[data], model]).split('_')
            if model in ["gcn", "ggnn"]:
                config_str = f"--hidden_dims {paras[0]}"
            elif model == "gat":
                config_str = f"--heads {paras[0]} --head_dims {paras[1]}"
            elif model == "gaan":
                config_str = f"--heads {paras[0]} --d_v {paras[1]} --d_a {paras[1]} --d_m {paras[1]} --hidden_dims {paras[2]}"
            

            for discard_per in [0, 0.01, 0.03, 0.06, 0.1, 0.2, 0.5]:
                real_path = dir_out + f'/{model}_{data}_{str(int(100*discard_per))}.csv'
                if os.path.exists(real_path):
                    continue
                dd = defaultdict(list)
                test_accs = []
                try:
                    for run in range(20):
                        sys.argv = [sys.argv[0], '--model', model, '--dataset', data, '--epoch', '40', '--device', 'cuda:0', '--seed', str(run)] + config_str.split(' ')
                        final_test_acc = epoch(discard_per)
                        test_accs.append(final_test_acc)
                except Exception as e:
                    print(e)
                dd[f'{model}_{data}_{discard_per}'] = test_accs
                pd.DataFrame(dd).to_csv(real_path)


if __name__ == '__main__':
    run(models=['gcn', 'gat'], datasets=['pubmed', 'coauthor-physics'])
