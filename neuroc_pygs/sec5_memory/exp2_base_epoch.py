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
def test_full(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    logits, accs = F.log_softmax(out, dim=1), []

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

@torch.no_grad()
def infer(model, data, subgraphloader):
    model.eval()
    model.reset_parameters()
    y_pred = model.inference_cuda(data.x, subgraphloader)
    y_true = data.y.cpu()

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        acc = model.evaluator(y_pred[mask], y_true[mask]) 
        accs.append(acc)
    return accs


def train(model, optimizer, data, loader, device, mode, best_val_acc, test_acc, patience_step, df, best_model, subgraph_loader):
    model.reset_parameters()
    model.train()
    loader_num, loader_iter = len(loader), iter(loader)
    for _ in range(loader_num):
        if mode == 'cluster':
            batch = next(loader_iter)
            batch = batch.to(device)
            batch_size = batch.train_mask.sum().item()
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            # nodes, edges = batch.x.shape[0], batch.edge_index.shape[1]
            # memory = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"] / (1024*1024*1024)
            # torch.cuda.reset_max_memory_allocated(device)
            # df['nodes'].append(nodes)
            # df['edges'].append(edges)
            # df['memory'].append(memory)
            # print(f'nodes={nodes}, edges={edges}, memory={memory}')
            # accs = test_full(model, data.to(device))
            accs = infer(model, data, subgraph_loader)
            if accs[1] > best_val_acc:
                patience_step = 0
                best_val_acc = accs[1]
                test_acc = accs[2]
                best_model = copy.deepcopy(model)
            else:
                patience_step += 1
                if patience_step >= 100:
                    break 
            print(f'Batch: {_:03d}, Loss: {loss.item():.8f}, Train: {accs[0]:.8f}, Val: {accs[1]:.8f}, Test: {accs[2]:.8f}, Best Val: {best_val_acc:.8f}, Best Test: {test_acc:.8f}')
    return best_val_acc, test_acc, patience_step, best_model


def epoch():
    args = get_args()
    print(args)
    data = build_dataset(args)
    train_loader = build_train_loader(args, data)
    subgraph_loader = build_subgraphloader(args, data)
    model, optimizer = build_model_optimizer(args, data)
    model = model.to(args.device)
    model.reset_parameters()
    best_val_acc, test_acc = 0, 0
    patience_step = 0
    best_model = None
    df = defaultdict(list)
    for epoch in range(5):
        best_val_acc, test_acc, patience_step, best_model  = train(model, optimizer, data, train_loader, args.device,
                            args.mode, best_val_acc, test_acc, patience_step, df, best_model, subgraph_loader)
        if patience_step >= 100:
            break
    # peak_memory = df['memory']
    # print(f'max: {max(peak_memory)}, min: {np.min(peak_memory)}, medium: {np.median(peak_memory)}, diff: {max(peak_memory)-min(peak_memory)}')
    torch.save(best_model.state_dict(), os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_inference_cutting',  f'{args.model}_{args.dataset}.pth')) 
    return



default_args = '--hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32'

for exp_model in ['gcn', 'gat']:
    for exp_data in ['coauthor-physics', 'flickr']:
        sys.argv = [sys.argv[0], '--model', exp_model, '--dataset', exp_data] + default_args.split(' ')
        epoch()
        # for exp_rs in [0.4]:
        #     sys.argv = [sys.argv[0], '--model', exp_model, '--dataset', exp_data, '--relative_batch_size', str(exp_rs)] + default_args.split(' ')
        #     epoch()