import torch
import queue as Queue
from threading import Thread
from torch_geometric.data import Data


class CudaDataLoader(object):
    """ 异步预先将数据从CPU加载到GPU中 """
    def __init__(self, loader, device, sampler='', data=None, to_flag=True, queue_size=2):
        self.device = device
        self.sampler = sampler
        self.data = data
        self.to_flag = to_flag
        self.queue_size = queue_size
        self.loader = loader

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue.Queue(maxsize=self.queue_size)

        self.idx = 0
        self.worker = Thread(target=self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()

    def load_loop(self):
        """ 不断的将cuda数据加载到队列里"""
        while True:
            for i, sample in enumerate(self.loader):
                # 内存超限处理机制
                if self.to_flag:
                    data = self.load_instance(sample)
                    if self.sampler == 'graphsage':
                        with torch.cuda.stream(self.load_stream):
                            batch_size, n_id, adjs = data
                            x, y = self.data.x[n_id], self.data.y[n_id[:batch_size]]
                            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                            data = [batch_size, n_id, adjs, x, y]
                else:
                    data = sample
                self.queue.put(data)

    def load_instance(self, sample):
        if torch.is_tensor(sample) or hasattr(sample, 'to'):
            with torch.cuda.stream(self.load_stream):
                return sample.to(self.device, non_blocking=True)
        elif sample is None or type(sample) in (str, int, float, bool):
            return sample
        elif isinstance(sample, dict):
            return {k: self.load_instance(v) for k, v in sample.items()}
        else:
            return [self.load_instance(s) for s in sample]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        elif self.idx >= len(self.loader):
            self.idx = 0
            raise StopIteration
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

    def next(self):
        return self.__next__()

    def __len__(self):
        return len(self.loader)






