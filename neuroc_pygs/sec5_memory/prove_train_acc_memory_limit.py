# memory_limit 待做!
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
from neuroc_pygs.sec4_time.epoch_utils import train_full, test_full
from neuroc_pygs.configs import PROJECT_PATH


dir_out = os.path.join(PROJECT_PATH, 'sec5_memory', 'out_train_data')

def train(model, optimizer, data, loader, device, mode, discard_per=0.01, df=None, non_blocking=False):
    model.reset_parameters()
    model.train()
    all_loss = []
    loader_num, loader_iter = len(loader), iter(loader)
    outilers_batches = np.random.choice(range(loader_num), int(loader_num * discard_per))
    for _ in range(loader_num):
        batch = next(loader_iter)
        # if _ in outilers_batches:
        #     continue
        batch = batch.to(device, non_blocking=non_blocking)
        batch_size = batch.train_mask.sum().item()
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
        all_loss.append(loss.item() * batch_size)
        if df is not None:
            df['nodes'].append(batch.x.shape[0])
            df['edges'].append(batch.edge_index.shape[1])
            df['memory'].append(torch.cuda.memory_stats(device)["allocated_bytes.all.peak"])
            torch.cuda.reset_max_memory_allocated(device)
            torch.cuda.empty_cache()
    return np.sum(all_loss) / int(data.train_mask.sum()), df


def epoch(discard_per=0.01): 
    args = get_args()
    print(args)
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    loader = build_train_loader(args, data)
    model, data = model.to(args.device), data.to(args.device)
    best_val_acc = 0
    final_test_acc = 0
    best_model = None
    patience_step = 0
    for epoch in range(args.epochs): # 50
        res = defaultdict(list)
        train(model, optimizer, data, loader, args.device, args.mode, discard_per=discard_per, df=res)
        peak_memory = list(map(lambda x: x / (1024 * 1024 * 1024), res['memory']))
        print(f'max: {max(peak_memory)}, min: {np.min(peak_memory)}, medium: {np.median(peak_memory)}, diff: {max(peak_memory)-min(peak_memory)}')
        train_acc, val_acc, test_acc = test_full(model, data)
        if val_acc > best_val_acc:
            patience_step = 0
            best_val_acc = val_acc
            final_test_acc = test_acc
            best_model = copy.deepcopy(model)
        else:
            patience_step += 1
            if patience_step >= 50:
                break
        # if epoch % 1 == 0:
        if True:
            print(f"Epoch: {epoch:03d}, Accuracy: Train: {train_acc:.4f}, Best Val: {best_val_acc:.4f}, Test: {final_test_acc:.4f}")
    return final_test_acc


def test_acc():
    headers = ['Model', 'Data', 'Per', 'Acc']
    small_datasets =  ['pubmed', 'coauthor-physics']
    for model in ['gcn', 'gat']:
        tab_data = []
        for data in small_datasets:
            for discard_per in [0.01, 0.05, 0.1, 0.15]:
                sys.argv = [sys.argv[0], '--model', model, '--dataset', data, '--epoch', '1000', '--device', 'cuda:0']
                final_test_acc = epoch()
                res = [model, data, discard_per, final_test_acc, t2 - t1]
                print(res)
                tab_data.append(res)
        print(tabulate(tab_data, headers=headers, tablefmt='github'))
        # pd.DataFrame(tab_data, columns=headers).to_csv(dir_out + f'/prove_train_acc_{model}.csv')


