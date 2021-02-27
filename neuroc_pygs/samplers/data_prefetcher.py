import torch
from neuroc_pygs.utils import to


class DataPrefetcher():
    def __init__(self, loader, sampler, device, data=None):
        self.loader = iter(loader)
        self.sampler, self.device, self.data = sampler, device, data
        self.stream = torch.cuda.Stream(device=device)
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            if self.sampler == 'graphsage':
                batch_size, n_id, adjs = self.batch
                adjs = [adj.to(device=self.device, non_blocking=True) for adj in adjs]
                x, y = self.data.x[n_id], self.data.y[n_id[:batch_size]]
                x, y = x.cuda(self.device, non_blocking=True), y.cuda(self.device, non_blocking=True)
                self.batch = [batch_size, x, y, adjs]
            else:
                self.batch = to(self.batch, self.device, non_blocking=True)
    
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch


if __name__ == '__main__':
    from neuroc_pygs.options import prepare_trainer
    for mode in ['cluster', 'graphsage']:
        data, cluster_loader, subgraph_loader, model, optimizer, args = prepare_trainer(mode=mode)
        loader_iter = DataPrefetcher(loader=cluster_loader, sampler=mode, device=args.device, data=data)
        print(loader_iter.next())

