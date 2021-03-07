import torch
import os
import sys
import time 
import numpy as np

from tabulate import tabulate
from neuroc_pygs.samplers.prefetch_generator import BackgroundGenerator
from neuroc_pygs.samplers.data_prefetcher import DataPrefetcher
from neuroc_pygs.options import get_args, build_dataset, build_model, build_subgraphloader
from neuroc_pygs.configs import ALL_MODELS, MODES, EXP_DATASET, PROJECT_PATH, EXP_RELATIVE_BATCH_SIZE


@torch.no_grad()
def infer(model, data, subgraphloader, f_iter, split="val"):
    model.eval()
    y_pred = model.inference_base(data.x, subgraphloader, f_iter)
    y_true = data.y.cpu()

    mask = getattr(data, split + "_mask")
    loss = model.loss_fn(y_pred[mask], y_true[mask])
    acc = model.evaluator(y_pred[mask], y_true[mask]) 
    return acc, loss.item()


@torch.no_grad()
def infer_cuda(model, data, subgraphloader, f_iter, split="val"):
    model.eval()
    y_pred = model.inference_cuda(data.x, subgraphloader, f_iter)
    y_true = data.y.cpu()

    mask = getattr(data, split + "_mask")
    loss = model.loss_fn(y_pred[mask], y_true[mask])
    acc = model.evaluator(y_pred[mask], y_true[mask]) 
    return acc, loss.item()


sys.argv = [sys.argv[0], '--device', 'cuda:2', '--model', 'gaan']
args = get_args()
data = build_dataset(args)
model = build_model(args, data)
subgraph_loader = build_subgraphloader(args, data)

model = model.to(args.device)
res = infer(model, data, subgraph_loader, f_iter=lambda x: iter(x))
print(res)

res = infer(model, data, subgraph_loader, f_iter=lambda x: BackgroundGenerator(x))
print(res)

res = infer_cuda(model, data, subgraph_loader, 
    f_iter=lambda x: DataPrefetcher(iter(x), 'graphsage', args.device, data))
print(res)

res = infer_cuda(model, data, subgraph_loader, 
    f_iter=lambda x: DataPrefetcher(BackgroundGenerator(x), 'graphsage', args.device, data))
print(res)