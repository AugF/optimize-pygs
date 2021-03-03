import os, sys
from threading import Thread
from queue import Queue

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from neuroc_pygs.utils import to


class AsynchronousLoader(object):
    # https://github.com/HenryJia/Lighter/blob/master/lighter/train/loaders.py
    def __init__(self, dataloader, device, sampler, queue_size = 1, **kwargs):
        self.device = device
        self.sampler = sampler
        self.queue_size = queue_size

        self.load_stream = torch.cuda.Stream(device = device)
        self.queue = Queue(maxsize = self.queue_size)

        # use PyTorch Loader 
        self.dataloader = dataloader
        self.idx = 0


    def load_loop(self): # The loop that will load into the queue in the background
        for i, sample in enumerate(self.dataloader):
            self.queue.put(self.load_instance(sample))


    def load_instance(self, sample): # Recursive loading for each instance based on torch.utils.data.default_collate
        with torch.cuda.stream(self.load_stream):
            if self.sampler == 'graphsage':
                batch_size, n_id, adjs = sample
                adjs = [adj.to(device=self.device, non_blocking=True) for adj in adjs]
                x, y = self.data.x[n_id], self.data.y[n_id[:batch_size]]
                x, y = x.cuda(self.device, non_blocking=True), y.cuda(self.device, non_blocking=True)
                self.batch = [batch_size, x, y, adjs]
            else:
                self.batch = to(sample, self.device, non_blocking=True)


    def __iter__(self):
        assert self.idx == 0, 'idx must be 0 at the beginning of __iter__. Are you trying to run the same instance more than once in parallel?'
        self.idx = 0
        self.worker = Thread(target = self.load_loop)
        #self.worker.setDaemon(True)
        self.worker.start()
        return self


    def __next__(self):
        # If we've reached the number of batches to return or the queue is empty and the worker is dead then exit
        if (not self.worker.is_alive() and self.queue.empty()) or self.idx >= len(self.dataloader):
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        else: # Otherwise return the next batch
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out


    def __len__(self):
        return len(self.dataloader)


if __name__ == '__main__':
    import time
    from neuroc_pygs.options import build_train_loader, get_args, build_dataset

    args = get_args()
    # args.dataset = 'amazon'
    # args.cluster_partitions = 150
    data = build_dataset(args)
    train_loader = build_train_loader(args, data)
    iter1 = iter(train_loader)
    iter2 = AsynchronousLoader(train_loader, args.device, args.mode)
    t1 = time.time()
    for _ in iter1: pass
    t2 = time.time()
    for _ in iter2: pass
    t3 = time.time()
    print(f'use time: {t2 - t1}, opt time: {t3 - t2}')