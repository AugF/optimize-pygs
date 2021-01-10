import time
from threading import Thread
from queue import Queue
import torch
import torch.nn.functional as F
import numpy as np

from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from code.models.sage import SAGE
from code.optimize_batch.utils import get_args, train_base, MyThread

    
def train_next(model, loader, optimizer, device):
    model.train()
    num = len(loader)
    
    def task1(q1):
        loader_iter = iter(loader)
        for i in range(num):
            data = next(loader_iter)
            q1.put(data)
    
    def task2(q1, q2):
        for i in range(num):
            data = q1.get()
            data = data.to(device)
            q2.put(data)
        
    def task3(q2):
        total_loss = total_examples = total_correct = 0
        for i in range(num):
            data = q2.get()
            if data.train_mask.sum() == 0: # task3
                continue
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)[data.train_mask]
            y = data.y.squeeze(1)[data.train_mask]
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()

            num_examples = data.train_mask.sum().item()
            total_loss += loss.item() * num_examples
            total_examples += num_examples

            total_correct += out.argmax(dim=-1).eq(y).sum().item()
            total_examples += y.size(0)
        return total_loss / total_examples, total_correct / total_examples

    q1, q2 = Queue(), Queue()
    job1 = Thread(target=task1, args=(q1,))
    job2 = Thread(target=task2, args=(q1, q2, ))
    job3 = MyThread(target=task3, args=(q2, ))
    
    job1.start()
    job2.start()
    job3.start()
    
    job1.join()
    job2.join()
    job3.join()
    return job3.get_result()


# ---- begin ----
# step1. get args
args = get_args(description="ogbn_products_sage_cluster")
print(args)
device = f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu'
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if 'cuda' in device:
    torch.cuda.manual_seed(args.seed)

device = torch.device(device)
dataset = PygNodePropPredDataset(name='ogbn-products', root="/home/wangzhaokang/wangyunpan/gnns-project/datasets")
split_idx = dataset.get_idx_split()
data = dataset[0]

for key, idx in split_idx.items():
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[idx] = True
    data[f'{key}_mask'] = mask


model = SAGE(data.x.size(-1), args.hidden_channels, dataset.num_classes,
                args.num_layers, args.dropout).to(device)

# 开始测试性能: 
print(f"ogbn_products: node, edge = {data.num_nodes}, {data.num_edges}")
cluster_batchs = [15, 45, 90, 150, 375, 750]
xticklabels = ['1%', '3%', '6%', '10%', '25%', '50%']

method_names = ['base_model', 'pipeline_thread']

# for i, bs in enumerate(cluster_batchs):
if True:
    cluster_data = ClusterData(data, num_parts=args.num_partitions,
                                recursive=False, save_dir=dataset.processed_dir)

    loader = ClusterLoader(cluster_data, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)
    cluster_data = ClusterData(data, num_parts=1500,
                                recursive=False, save_dir=dataset.processed_dir)

    loader = ClusterLoader(cluster_data, batch_size=bs,
                            shuffle=True, num_workers=args.num_workers)
    for j, train_method in enumerate([test_base, test_next]):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        st1 = time.time()
        loss, train_acc = train_method(model, loader, optimizer, device)
        print(f'loss: {loss:.4f}, train_acc: {train_acc:.4f}')
        st2 = time.time()
        print(f"method={method_names[j]}, relative batch size={xticklabels[i]}, use time: {st2 - st1}s")

