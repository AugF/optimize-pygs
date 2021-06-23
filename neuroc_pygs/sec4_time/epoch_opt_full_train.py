import os
import torch
import time
from threading import Thread
from queue import Queue

from neuroc_pygs.sec4_time.epoch_utils import train_full
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer

def train(q1):
    args = get_args()
    print(args)
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model, data = model.to(args.device), data.to(args.device)

    for epoch in range(args.epochs):
        t1 = time.time()
        train_full(model, data, optimizer)
        t2 = time.time()
        save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'st_time': t1,
                'ed_train_time': t2
            }
        q1.put(save_dict)
    q1.put(None)


def save_file(q1):
    checkpoint_dir = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/exp_checkpoints'

    while True:
        save_dict = q1.get()
        if save_dict == None:
            return
        else:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(save_dict, os.path.join(checkpoint_dir, 'model_full_%d.pth' % save_dict['epoch']))


def run_new_train():
    q = Queue()
    t1 = Thread(target=train, args=(q,))
    t2 = Thread(target=save_file, args=(q,))
    t1.start()
    t2.start()
    t1.join()
    t2.join() 


def run_train():
    args = get_args()
    print(args)
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model, data = model.to(args.device), data.to(args.device)
    for epoch in range(args.epochs):
        t1 = time.time()
        train_full(model, data, optimizer)
        t2 = time.time()
        save_dict = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'st_time': t1,
            'ed_train_time': t2
        }
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(save_dict, os.path.join(args.checkpoint_dir, 'model_full_%d.pth' % epoch))


def test_train_overhead():
    args = get_args()
    # print(args)
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model, data = model.to(args.device), data.to(args.device)
    times = []
    for epoch in range(args.epochs):
        t1 = time.time()
        train_full(model, data, optimizer)
        t2 = time.time()
        save_dict = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'st_time': t1,
            'ed_train_time': t2
        }
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(save_dict, os.path.join(args.checkpoint_dir, 'model_full_%d.pth' % epoch))
        t3 = time.time()
        times.append(t3 - t2)
        # print('use time', t3 - t2)
    return sum(times) / len(times)


if __name__ == '__main__':
    # import sys
    # for m in [5, 10, 20, 50, 100, 200]:
    #     data = f'random_100k_{m}k'
    #     sys.argv = [sys.argv[0], '--dataset', data, '--epoch', str(50)]    
    #     res = test_train_overhead()
    #     print(data, res)
    run_train()