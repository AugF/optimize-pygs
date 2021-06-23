import sys
import time
import torch
import numpy as np
from tabulate import tabulate
from neuroc_pygs.options import get_args, build_dataset, build_model
from neuroc_pygs.sec4_time.epoch_utils import test_full
from neuroc_pygs.sec6_cutting.cutting_method import cut_by_random, get_pagerank, get_degree, cut_by_importance_method


dir_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec6_cutting/out_cutting_method_res'

for exp_model in ['gcn', 'gat']:
    for exp_data in ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics']:
        tab_data = []
        sys.argv = [sys.argv[0], '--device', 'cuda:0', '--model', exp_model, '--data', exp_data]
        args = get_args()
        print(args)
        data = build_dataset(args)
        model = build_model(args, data)
        model, data = model.to(args.device), data.to(args.device)

        # 这里使用的是训练好的模型，对推理进行模拟
        save_dict = torch.load(dir_path + f'/{args.model}_{args.dataset}_best_model.pth')
        model.load_state_dict(save_dict)

        edges = data.edge_index.shape[1]
        edge_index = data.edge_index.cpu()
        ways = ['random', 'degree1', 'degree2', 'pr1', 'pr2']

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
            random1 = cut_by_random(edge_index, cutting_nums, seed=2)
            degree1 = cut_by_importance_method(edge_index, cutting_nums, method='degree', name='way1', degree=degree, pr=pr)
            degree2 = cut_by_importance_method(edge_index, cutting_nums, method='degree', name='way2', degree=degree, pr=pr)
            pr1 = cut_by_importance_method(edge_index, cutting_nums, method='pagerank', name='way1', degree=degree, pr=pr)
            pr2 = cut_by_importance_method(edge_index, cutting_nums, method='pagerank', name='way2', degree=degree, pr=pr)
            for i, cutting in enumerate([random1, degree1, degree2, pr1, pr2]):
                data.edge_index = cutting.to(args.device)
                acc = test_full(model, data)[2]
                print(f'per={per}, {ways[i]}, test_acc={acc:.8f}')
                tab_data.append([exp_model, exp_data, per, ways[i], acc])
        print(tabulate(tab_data, headers=['Model', 'Data', 'Per', 'Method', 'Acc'], tablefmt='github'))
