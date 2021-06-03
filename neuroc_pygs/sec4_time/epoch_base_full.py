import math
import time
import numpy as np
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer
from neuroc_pygs.sec4_time.epoch_utils import train_full, test_full


def epoch(): # 50取平均，删除一些异常元素
    args = get_args()
    print(args)
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model, data = model.to(args.device), data.to(args.device)
    train_times, eval_times = [], []
    for epoch in range(args.epochs): # 50
        st_time = time.time()
        train_full(model, data, optimizer)
        t0 = time.time()
        if args.eval_per is not None: 
            time.sleep((t0 - st_time) * args.eval_per / (1 - args.eval_per))
        else:
            accs = test_full(model, data)
        ed_time = time.time()
        print(f'Epoch: {epoch:03d}, train_time: {t0 - st_time}, eval_time: {ed_time - t0}, all_time: {ed_time - st_time}')
        train_times.append(t0 - st_time); eval_times.append(ed_time - t0)
    avg_train_time, avg_eval_time = np.mean(train_times), np.mean(eval_times)
    x = avg_eval_time / (avg_train_time + avg_eval_time)
    exp_ratio = 1 / args.epochs + max(x, 1-x) * (args.epochs - 1) / args.epochs
    print(f'Average train_time: {np.mean(train_times)}, eval_time: {np.mean(eval_times)}, x: {x}, exp_ratio: {exp_ratio}')
    return

def run(): # 50取平均，删除一些异常元素
    args = get_args()
    print(args)
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model, data = model.to(args.device), data.to(args.device)
    tab_data = []
    for epoch in range(args.epochs):
        st_time = time.time()
        train_full(model, data, optimizer)
        t0 = time.time()
        if args.eval_per is not None: 
            time.sleep((t0 - st_time) * args.eval_per / (1 - args.eval_per))
        else:
            print('error')
            accs = test_full(model, data)
        ed_time = time.time()
        tab_data.append([t0 - st_time, ed_time - t0])
    tab_data = np.array(tab_data)
    tab_data = np.mean(tab_data, axis=0)
    print(tab_data)
    return

if __name__ == '__main__':
    epoch()
    # import sys
    # default_args = '--epochs 100 --hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32'
    # sys.argv = [sys.argv[0]] + default_args.split(' ') + ['--model', 'gcn', '--dataset', 'amazon-computers']
    # run()