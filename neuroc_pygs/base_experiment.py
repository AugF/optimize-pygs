import math
import time
import numpy as np
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer, build_train_loader, build_subgraphloader
from neuroc_pygs.sec4_time.epoch_utils import train_full, test_full, train, infer
from neuroc_pygs.samplers.cuda_prefetcher import CudaDataLoader

def trainer_full(args):
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model, data = model.to(args.device), data.to(args.device)
    for epoch in range(args.epochs): # 50
        train_full(model, data, optimizer)
        accs = test_full(model, data)
        print(f'Epoch: {epoch:03d}, Train: {accs[0]:.8f}, Val: {accs[1]:.8f}, Test: {accs[2]:.8f}')
    return


def trainer_sampling(args):
    data = build_dataset(args)
    train_loader = build_train_loader(args, data)
    subgraph_loader = build_subgraphloader(args, data)
    if args.opt_train_flag:
        train_loader = CudaDataLoader(train_loader, device=args.device)
    if args.opt_eval_flag:
        subgraph_loader = CudaDataLoader(subgraph_loader, args.device)

    model, optimizer = build_model_optimizer(args, data)
    model = model.to(args.device)
    model.reset_parameters()
    for _ in range(args.epochs):
        train(model, optimizer, data, train_loader, args.device, args.mode, non_blocking=False)
        accs, losses = infer(model, data, subgraph_loader)
        print(f'Epoch: {_:03d}, Train: {accs[0]:.8f}, Val: {accs[1]:.8f}, Test: {accs[2]:.8f}')
    return


if __name__ == '__main__':
    args = get_args()
    print(args)
    if args.mode == 'None':
        trainer_full(args)
    else:
        trainer_sampling(args)
