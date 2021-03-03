import torch
from neuroc_pygs.utils import to


class DataPrefetcher():
    def __init__(self, loader, sampler, device, data=None):
        self.loader = iter(loader)
        self.sampler, self.device, self.data = sampler, device, data
        self.stream = torch.cuda.Stream(device=device)
        self.preload()

    def preload(self):
        t1 = time.time()
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        t2 = time.time()
        with torch.cuda.stream(self.stream):
            if self.sampler == 'graphsage':
                batch_size, n_id, adjs = self.batch
                adjs = [adj.to(device=self.device, non_blocking=True) for adj in adjs]
                x, y = self.data.x[n_id], self.data.y[n_id[:batch_size]]
                x, y = x.cuda(self.device, non_blocking=True), y.cuda(self.device, non_blocking=True)
                self.batch = [batch_size, x, y, adjs]
            else:
                self.batch = to(self.batch, self.device, non_blocking=True)
        t3 = time.time()
        print(f'try use time: {t2-t1}, else use time: {t3-t2}')

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


if __name__ == '__main__':
    import time
    from neuroc_pygs.options import build_train_loader, get_args, build_dataset
    args = get_args()
    args.dataset = 'amazon'
    args.cluster_partitions = 150
    data = build_dataset(args)
    train_loader = build_train_loader(args, data)

    st0 = time.time()
    iter1 = iter(train_loader)
    st1 = time.time()
    iter2 = DataPrefetcher(iter(train_loader), sampler=args.mode, device=args.device, data=data)
    st2 = time.time()
    print(f'use time: {st1 - st0}, opt_time: {st2 - st1}')

    t1 = time.time()
    for batch in iter1: 
        batch = batch.to(args.device)
        print(batch)

    t2 = time.time()
    for i in range(len(train_loader)):
        print(iter2.next())
    t3 = time.time()
    print(f'use time: {t2 - t1}, opt time: {t3 - t2}')


