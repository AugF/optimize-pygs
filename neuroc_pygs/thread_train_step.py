import torch
import time
import os.path as osp
import numpy as np

from neuroc_pygs.utils import BatchLogger
from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.samplers.thread_loader import ThreadLoader
from neuroc_pygs.train_step import train, test, infer
from neuroc_pygs.options import prepare_trainer


def opt_trainer(train_func=train, test_func=test, infer_func=infer): # 训练函数可定制化
    data, train_loader, subgraph_loader, model, optimizer, args = prepare_trainer()
    model = model.to(args.device)
    train_loader = ThreadLoader(loader=train_loader, sampler=args.mode, device=args.device, data=None if args.model == 'cluster' else data)
    # step1 fit
    best_val_acc = 0
    best_model = None
    for epoch in range(1):
        train(model, data, train_loader, optimizer, args) # data由训练负责
    train_loader.stop()
    return

def trainer(train_func=train, test_func=test, infer_func=infer): # 训练函数可定制化
    data, train_loader, subgraph_loader, model, optimizer, args = prepare_trainer()
    model = model.to(args.device)
    # step1 fit
    best_val_acc = 0
    best_model = None
    for epoch in range(1):
        train(model, data, train_loader, optimizer, args) # data由训练负责
    return
  

if __name__ == '__main__':
    # begin test
    t1 = time.time()
    trainer()
    t2 = time.time()
    opt_trainer()
    t3 = time.time()
    base_time, opt_time = t2 - t1, t3 - t2
    ratio = 100 * (base_time - opt_time) / base_time
    print(base_time, opt_time, ratio)
    exit(0)