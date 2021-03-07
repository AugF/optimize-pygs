import torch
import threading
import queue as Queue

class TransferGenerator(threading.Thread):
    def __init__(self, generator, device, max_prefetch=1):
        super().__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.stream = torch.cuda.Stream(device)
        self.device = device
        self.daemon = True
        self.start()
        self.exhausted = False


    def run(self):
        with torch.cuda.stream(self.stream):
            for item in self.generator:
                item = item.to(self.device)
                self.queue.put(item)
            self.queue.put(None)

    def next(self):
        if self.exhausted:
            raise StopIteration
        else:
            next_item = self.queue.get()
            if next_item is None:
                raise StopIteration
            return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


if __name__ == '__main__':
    import time
    from neuroc_pygs.options import build_train_loader, get_args, build_dataset
    args = get_args()
    data = build_dataset(args)
    train_loader = build_train_loader(args, data)

    iter1 = iter(train_loader)
    iter2 = TransferGenerator(iter(train_loader), args.device)
    print('start')
    t1 = time.time()
    for i in iter1:
        print(i)
    t2 = time.time()
    print('x')
    for i in iter2:
        print(i)
    t3 = time.time()
    print(f'use time: {t2 - t1}, opt time: {t3 - t2}')