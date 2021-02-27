from neuroc_pygs.utils import to
from neuroc_pygs.options import prepare_trainer



data, cluster_loader, subgraph_loader, model, optimizer, args = prepare_trainer(mode='cluster')
data, neighbor_loader, subgraph_loader, model, optimizer, args = prepare_trainer(mode='graphsage')

cluster_iter = iter(cluster_loader)
neighbor_iter = iter(neighbor_loader)

cluster_batch = cluster_iter.next()
neighbor_batch = neighbor_iter.next()

print(cluster_batch, dir(cluster_batch), hasattr(cluster_batch, 'to'), hasattr(neighbor_batch[2][0], 'to'))

from collections import Iterable
def To(batch):
    if hasattr(batch, 'to'):
        return to(batch, device=args.device)
    elif isinstance(batch, list):
        for i, k in enumerate(batch):
            batch[i] = To(k)
    elif isinstance(batch, Iterable):
        for k in batch:
            batch[k] = To(batch[k])
    return batch

cluster_batch = To(cluster_batch)
neighbor_batch = To(neighbor_batch)
print(cluster_batch)
print(neighbor_batch)