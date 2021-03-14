import copy
import math
import sys, os
import time
import torch
import numpy as np
from neuroc_pygs.options import get_args, build_dataset, build_model
from neuroc_pygs.sec4_time.epoch_utils import test_full
from neuroc_pygs.sec6_cutting.cutting_methods import cut_by_random, cut_by_importance


dir_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec6_cutting/exp_cutting_model'

args = get_args()
print(args)
data = build_dataset(args)
model = build_model(args, data)
model, data = model.to(args.device), data.to(args.device)

save_dict = torch.load(dir_path + f'/{args.model}_{args.dataset}_best_model_v0.pth')
model.load_state_dict(save_dict)

edges = data.edge_index.shape[1]
edge_index = data.edge_index
ways = ['random', 'degree1', 'degree2', 'pr1', 'pr2']

print(f'full, test_acc={test_full(model, data)[2]}')
for per in [0.01, 0.03, 0.06, 0.1, 0.2, 0.5]:
    cutting_nums = int(edges * per)
    random_ = cut_by_random(edge_index, cutting_nums=2)
    degree1 = cut_by_importance(
        edge_index, cutting_nums=2, method='degree', name='way1')
    degree2 = cut_by_importance(
        edge_index, cutting_nums=2, method='degree', name='way2')
    pr1 = cut_by_importance(edge_index, cutting_nums=2,
                            method='pagerank', name='way1')
    pr2 = cut_by_importance(edge_index, cutting_nums=2,
                            method='pagerank', name='way2')
    for i, cutting in enumerate([random_, degree1, degree2, pr1, pr2]):
        data.edge_index = cutting
        print(f'per={per}, {ways[i]}, test_acc={test_full(model, data)[2]}')
