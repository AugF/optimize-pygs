import copy
import math
import sys, os
import time
import torch
import numpy as np
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer
from neuroc_pygs.sec4_time.epoch_utils import train_full, test_full


dir_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec6_cutting/exp_cutting_model'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def epoch(): 
    args = get_args()
    print(args)
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model, data = model.to(args.device), data.to(args.device)
    best_val_acc = 0
    final_test_acc = 0
    best_model = None
    patience_step = 0
    for epoch in range(args.epochs): # 50
        train_full(model, data, optimizer)
        train_acc, val_acc, test_acc = test_full(model, data)
        if val_acc > best_val_acc:
            patience_step = 0
            best_val_acc = val_acc
            final_test_acc = test_acc
            best_model = copy.deepcopy(model)
        else:
            patience_step += 1
            if patience_step >= 100:
                break
        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Accuracy: Train: {train_acc:.4f}, Best Val: {best_val_acc:.4f}, Test: {final_test_acc:.4f}")
    torch.save(best_model.state_dict(), dir_path + f'/{args.model}_{args.dataset}_best_model_v0.pth')
    return final_test_acc


from tabulate import tabulate
tab_data = []
small_datasets =  ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr']
for model in ['gcn', 'gat']:
    for data in small_datasets:
        sys.argv = [sys.argv[0], '--model', model, '--dataset', data, '--epoch', '2000', '--device', 'cuda:1']
        final_test_acc = epoch()
        tab_data.append([model, data, final_test_acc])
print(tabulate(tab_data, headers=['Model', 'Data', 'Acc'], tablefmt='github'))