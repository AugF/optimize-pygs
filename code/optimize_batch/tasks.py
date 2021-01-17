import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler

from code.models.sage import SAGE
from code.optimize_batch.utils import get_args

args = get_args(description="ogbn_products_sage_cluster")

# step2. prepare data
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

dataset = PygNodePropPredDataset(name='ogbn-products', root="/home/wangzhaokang/wangyunpan/gnns-project/datasets")
split_idx = dataset.get_idx_split()
data = dataset[0]

for key, idx in split_idx.items():
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[idx] = True
    data[f'{key}_mask'] = mask

# step3. get dataloader
cluster_data = ClusterData(data, num_parts=args.num_partitions,
                            recursive=False, save_dir=dataset.processed_dir)

loader = ClusterLoader(cluster_data, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers)

model = SAGE(data.x.size(-1), args.hidden_channels, dataset.num_classes,
                args.num_layers, args.dropout).to(device)

# step4. set model and optimizer
model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def task1(loader):
    data = next(loader)
    if data.train_mask.sum() == 0: # task3
        return None
    return data

def task2(data, device):
    return data.to(device)
    
def task3(data, model, optimizer):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)[data.train_mask]
    y = data.y.squeeze(1)[data.train_mask]
    loss = F.nll_loss(out, y)
    loss.backward()
    optimizer.step()
    num_examples = data.train_mask.sum().item()
    acc = out.argmax(dim=-1).eq(y).sum()
    # loss, acc, num = float(), float(acc.item()), int(num_examples + y.size(0))
    # print(type(loss), type(acc), type(num))
    return model, optimizer, loss.item(), acc.item(), num_examples + y.size(0)


def run():
    global model
    import time
    print("in", model.convs[0].lin_l.weight)
    t1 = time.time()
    loader_iter = iter(loader)
    data_cpu = task1(loader_iter)
    data_gpu = task2(data_cpu, device);
    res = task3(data_gpu, model, optimizer)
    model = res[0]
    t2 = time.time()
    print("use time", t2 - t1)
    print("task1", data_cpu)
    print("task2", data_gpu)
    print("task2", res)
    print("out", model.convs[0].lin_l.weight)
    
# run()