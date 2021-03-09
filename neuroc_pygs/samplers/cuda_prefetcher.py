import torch
import queue as Queue
from threading import Thread
from torch_geometric.data import Data


class CudaDataLoader(object):
    """ 异步预先将数据从CPU加载到GPU中 """
    def __init__(self, loader, device, sampler='', data=None, queue_size=2):
        self.device = device
        self.sampler = sampler
        self.data = data
        self.queue_size = queue_size
        self.loader = loader

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue.Queue(maxsize=self.queue_size)

        self.idx = 0
        self.worker = Thread(target=self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()

    def load_loop(self):
        """ 不断的将cuda数据加载到队列里 """
        # The loop that will load into the queue in the background
        while True:
            for i, sample in enumerate(self.loader):
                if self.sampler == 'infer_sage':
                    self.queue.put(sample)
                else:
                    self.queue.put(self.load_instance(sample))

    def load_instance(self, sample):
        """ 将batch数据从CPU加载到GPU中 """
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
        # 加载线程挂了
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        # 一个epoch加载完了
        elif self.idx >= len(self.loader):
            self.idx = 0
            raise StopIteration
        # 下一个batch
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

    def next(self):
        return self.__next__()

    def __len__(self):
        return len(self.loader)





if __name__ == '__main__':
    import time
    from neuroc_pygs.options import build_train_loader, get_args, build_dataset
    from neuroc_pygs.samplers.prefetch_generator import BackgroundGenerator
    from neuroc_pygs.samplers.data_prefetcher_2 import TransferGenerator
    from torch_geometric.data import Data
    args = get_args()
    args.mode = 'graphsage'
    data = build_dataset(args)
    train_loader = build_train_loader(args, data)

    opt_loader = CudaDataLoader(train_loader, args.device, sampler=args.mode, data=data)
    print('start', args.device)
    t1 = time.time()
    for i, x in enumerate(train_loader):
        print(i)
        if i == 0:
            print('base')
    t2 = time.time()
    for i, x in enumerate(opt_loader):
        print(i)
        if i == 0:
            print('opt', x)
            # print('opt', Data.from_dict({li[0]:li[1] for li in x}))
    t3 = time.time()
    print(f'use time: {t2 - t1}, opt time: {t3 - t2}')

