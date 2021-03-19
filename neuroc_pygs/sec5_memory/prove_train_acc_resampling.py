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


dir_out = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_train_res')
def train(model, optimizer, data, loader, device, mode, discard_per=0.01, non_blocking=False):
    model.reset_parameters()
    model.train()
    all_loss = []
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
        all_loss.append(loss.item() * batch_size)
    return np.sum(all_loss) / int(data.train_mask.sum())


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
    for epoch in range(args.epochs): # 50
        train(model, optimizer, data, loader, args.device, args.mode, discard_per)
        train_acc, val_acc, test_acc = test_full(model, data)
        if val_acc > best_val_acc:
            patience_step = 0
            best_val_acc = val_acc
            final_test_acc = test_acc
            best_model = copy.deepcopy(model)
        else:
            patience_step += 1
            if patience_step >= 50000:
                break
        # if epoch % 1 == 0:
        if df is not None:
            df['train_acc'].append(train_acc)
            df['val_acc'].append(val_acc)
            df['test_acc'].append(test_acc)
            df['best_val_acc'].append(best_val_acc)
            df['final_test_acc'].append(final_test_acc)   
        print(f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Best Val: {best_val_acc:.4f}, Test: {final_test_acc:.4f}")
    return final_test_acc


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
tab_data = []
headers = ['Model', 'Data', 'Per', 'Acc', 'Use time']
columns = ['train_acc', 'val_acc', 'test_acc', 'best_val_acc', 'final_test_acc']
small_datasets =  ['pubmed', 'coauthor-physics', 'amazon-computers', 'amazon-photo', 'flickr']

for data in small_datasets:
    for model in ['gcn', 'gat']:
        for discard_per in [0, 0.01, 0.03, 0.06, 0.1, 0.2, 0.5]:
            test_accs, use_times = [], []
            for run in range(3):
                real_path = dir_out + f'/{model}_{data}_{str(int(100*discard_per))}_{run}.csv'
                if os.path.exists(real_path):
                    test_accs.append(pd.read_csv(real_path, index_col=0).values[-1,-1])
                    use_times.append(0)
                    continue
                df = defaultdict(list)
                sys.argv = [sys.argv[0], '--model', model, '--dataset', data, '--epoch', '50', '--device', 'cuda:1']
                t1 = time.time()
                final_test_acc = epoch(discard_per, df=df)
                t2 = time.time()
                test_accs.append(final_test_acc); use_times.append(t2 - t1)
                pd.DataFrame(df).to_csv(real_path)
            res = [model, data, discard_per, np.mean(test_accs), np.mean(use_times)]
            print(res)
            tab_data.append(res)
print(tabulate(tab_data, headers=headers, tablefmt='github'))
pd.DataFrame(tab_data, columns=headers).to_csv(dir_out + '/prove_train_acc_small.csv')

# time不准确, 以后再说