import sys
import time
import numpy as np
from collections import defaultdict
from neuroc_pygs.options import get_args, build_dataset, build_train_loader, build_subgraphloader, build_model_optimizer
from neuroc_pygs.sec4_time.epoch_utils import train, infer
from neuroc_pygs.samplers.cuda_prefetcher import CudaDataLoader


def epoch():
    args = get_args()
    print(f'base epoch, opt_train_flag:{args.opt_train_flag}, opt_eval_flag:{args.opt_eval_flag}')
    print(args)
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
    train_times, eval_times = [], []
    for _ in range(args.epochs):
        t1 = time.time()
        train(model, optimizer, data, train_loader, args.device, args.mode, non_blocking=False)
        t2 = time.time()
        accs, losses = infer(model, data, subgraph_loader)
        t3 = time.time()
        print(f'Epoch: {_:03d}, Train: {accs[0]:.8f}, Val: {accs[1]:.8f}, Test: {accs[2]:.8f}, Train Time: {t2-t1}, Val Time: {t3-t2}')
        train_times.append(t3 - t2); eval_times.append(t2 - t1)
    avg_train_time, avg_eval_time = np.mean(train_times), np.mean(eval_times)
    x = avg_eval_time / (avg_train_time + avg_eval_time)
    exp_ratio = 1 / args.epochs + max(x, 1-x) * (args.epochs - 1) / args.epochs
    print(f'Average train_time: {np.mean(train_times)}, eval_time: {np.mean(eval_times)}, x: {x}, exp_ratio: {exp_ratio}')
    return


def run():
    args = get_args()
    print(f'base epoch, opt_train_flag:{args.opt_train_flag}, opt_eval_flag:{args.opt_eval_flag}')
    print(args)
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
    train_times, eval_times = [], []
    for _ in range(args.epochs):
        st_time = time.time()
        train(model, optimizer, data, train_loader, args.device, args.mode, non_blocking=False, opt_flag=args.opt_flag)
        t0 = time.time()
        accs, losses = infer(model, data, subgraph_loader)
        ed_time = time.time()
        print(f'Epoch: {epoch:03d}, train_time: {t0 - st_time}, eval_time: {ed_time - t0}, all_time: {ed_time - st_time}')
        train_times.append(t0 - st_time); eval_times.append(ed_time - t0)
    avg_train_time, avg_eval_time = np.mean(train_times), np.mean(eval_times)
    x = avg_eval_time / (avg_train_time + avg_eval_time)
    exp_ratio = 1 / args.epochs + max(x, 1-x) * (args.epochs - 1) / args.epochs
    print(f'Average train_time: {np.mean(train_times)}, eval_time: {np.mean(eval_times)}, x: {x}, exp_ratio: {exp_ratio}')
    return


if __name__ == '__main__':
    epoch()
    # import sys
    # default_args = '--epochs 30 --hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32'
    # sys.argv = [sys.argv[0]] + default_args.split(' ') + ['--model', 'gcn', '--dataset', 'amazon-computers']
    # run()