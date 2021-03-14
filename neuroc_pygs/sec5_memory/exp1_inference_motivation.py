import torch
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from neuroc_pygs.options import get_args, build_dataset, build_subgraphloader, build_model
from neuroc_pygs.configs import PROJECT_PATH


@torch.no_grad()
def infer(model, data, subgraph_loader, args, split='test'):
    device, log_batch, log_batch_dir = args.device, args.log_batch, args.log_batch_dir
    model.eval()
    y_pred = model.inference(data.x, subgraph_loader, log_batch)
    y_true = data.y.cpu()

    mask = getattr(data, split + "_mask")
    acc = model.evaluator(y_pred[mask], y_true[mask]) 
    return acc


@torch.no_grad()
def test_full(model, data, split='test'): 
    model.eval()
    y_pred = model(data.x, data.edge_index)
    y_true = data.y.cpu()

    mask = getattr(data, split + "_mask")
    acc = model.evaluator(y_pred[mask], y_true[mask]) 
    return acc


import sys
sys.argv = [sys.argv[0], '--device', 'cuda:0']
args = get_args()
for exp_model in ['gcn', 'ggnn', 'gat', 'gaan']:
    for exp_data in ['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'flickr']:
        args.dataset =exp_data
        print(args)
        data = build_dataset(args)
        model = build_model(args, data)
        model.reset_parameters()
        model = model.to(args.device)
        acc = test_full(model, data.to(args.device))
        data = data.to('cpu')
        print('full', acc)
        for bs in [0.01, 0.03, 0.06, 0.1, 0.25, 0.5]:
            args.infer_batch_size = int(data.x.shape[0] * bs)
            subgraphloader = build_subgraphloader(args, data)
            torch.cuda.reset_max_memory_allocated(args.device)
            res = defaultdict(list)
            num_loader = len(subgraphloader) * args.layers
            model.reset_parameters()
            acc = infer(model, data, subgraphloader, args)
            print(f'batchsize{bs}', acc)
