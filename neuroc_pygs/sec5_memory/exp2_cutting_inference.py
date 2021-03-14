import time
import copy
import numpy as np
import pandas as pd
from collections import defaultdict
from neuroc_pygs.options import get_args, build_dataset, build_train_loader, build_subgraphloader, build_model_optimizer

import os, sys
import torch
import torch.nn.functional as F
from neuroc_pygs.configs import PROJECT_PATH


@torch.no_grad()
def infer(model, data, subgraphloader):
    model.eval()
    model.reset_parameters()
    y_pred = model.inference_cuda(data.x, subgraphloader)
    y_true = data.y.cpu()

    return model.evaluator(y_pred[data.test_mask], y_true[data.test_mask]) 


@torch.no_grad()
def test_full(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    logits, accs = F.log_softmax(out, dim=1), []

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs[2]


if __name__ == '__main__':
    default_args = '--hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32'
    sys.argv = [sys.argv[0], '--device', 'cuda:2', '--model', 'gcn', '--dataset', 'coauthor-physics'] + default_args.split(' ')
    args = get_args()
    data = build_dataset(args)
    subgraph_loader = build_subgraphloader(args, data)
    model, optimizer = build_model_optimizer(args, data)
    save_dict = torch.load(os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_inference_cutting',  f'{args.model}_{args.dataset}.pth'))
    model.load_state_dict(save_dict)
    model = model.to(args.device)
    # test_acc = infer(model, data, subgraph_loader)
    # print(test_acc)
    print(test_full(model, data.to(args.device)))