import copy
import math
import sys, os
import time
import torch
import numpy as np
from tabulate import tabulate
from neuroc_pygs.options import get_args, build_dataset, build_model
from neuroc_pygs.sec4_time.epoch_utils import test_full
from neuroc_pygs.sec6_cutting.cutting_methods import cut_by_random, get_pagerank, get_degree, cut_by_importance_method


dir_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec6_cutting/exp_cutting_model'
tab_data = []
for exp_model in ['gcn', 'ggnn', 'gat', 'gaan']:
# for exp_model in ['gcn']:
    # for exp_data in ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics']:
    for exp_data in ['flickr']: 
        sys.argv = [sys.argv[0], '--device', 'cuda:0', '--model', exp_model, '--data', exp_data]
        args = get_args()
        print(args)
        data = build_dataset(args)
        model = build_model(args, data)
        model, data = model.to(args.device), data.to(args.device)

        save_dict = torch.load(dir_path + f'/{args.model}_{args.dataset}_best_model_v0.pth')
        model.load_state_dict(save_dict)

        edges = data.edge_index.shape[1]
        edge_index = data.edge_index.cpu()
        print(edge_index.shape)
        ways = ['random', 'degree3', 'degree4', 'pr3', 'pr4']

        t1 = time.time()
        degree = get_degree(edge_index)
        t2 = time.time()
        pr = get_pagerank(edge_index)
        t3 = time.time()
        acc = test_full(model, data)[2]
        print(f'full, test_acc={acc:.8f}, degree overhead: {t2 - t1}, pr overhead: {t3 - t2}')
        tab_data.append([exp_model, exp_data, 'full', 'none', acc])
        for per in [0.01, 0.03, 0.06, 0.1, 0.2, 0.5]:
            cutting_nums = int(edges * per)
            random_ = cut_by_random(edge_index, cutting_nums)
            # degree1 = cut_by_importance_method(
            #     edge_index, cutting_nums, method='degree', name='way1', degree=degree, pr=pr)
            # degree2 = cut_by_importance_method(
            #     edge_index, cutting_nums, method='degree', name='way2', degree=degree, pr=pr)
            degree3 = cut_by_importance_method(
                edge_index, cutting_nums, method='degree', name='way3', degree=degree, pr=pr)
            degree4 = cut_by_importance_method(
                edge_index, cutting_nums, method='degree', name='way4', degree=degree, pr=pr)
            # pr1 = cut_by_importance_method(edge_index, cutting_nums,
            #                         method='pagerank', name='way1', degree=degree, pr=pr)
            # pr2 = cut_by_importance_method(edge_index, cutting_nums,
            #                         method='pagerank', name='way2', degree=degree, pr=pr)
            pr3 = cut_by_importance_method(edge_index, cutting_nums,
                                    method='pagerank', name='way3', degree=degree, pr=pr)
            pr4 = cut_by_importance_method(edge_index, cutting_nums,
                                    method='pagerank', name='way4', degree=degree, pr=pr)
            for i, cutting in enumerate([random_, degree3, degree4, pr3, pr4]):
                print(cutting.shape)
                data.edge_index = cutting.to(args.device)
                acc = test_full(model, data)[2]
                print(f'per={per}, {ways[i]}, test_acc={acc:.8f}')
                tab_data.append([exp_model, exp_data, per, ways[i], acc])
            print(tabulate(tab_data, headers=['Model', 'Data', 'Per', 'Method', 'Acc'], tablefmt='github'))
