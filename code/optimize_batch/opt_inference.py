'''
该文件用来测试基于采样的并行化
https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_products_sage.py master: 8a57480
report acc: 0.7870 ± 0.0036
rank: 12
2020-10-27
'''

import os.path as osp
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.data import NeighborSampler

from code.models.sage import SAGE
from code.utils.logger import Logger


parser = argparse.ArgumentParser(description='Neighborsampling(SAGE)')
parser.add_argument('--train_device', type=int, default=0)
parser.add_argument('--eval_device', type=str, default="cpu")
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()
print(args)

dataset = PygNodePropPredDataset('ogbn-products', root="/home/wangzhaokang/wangyunpan/gnns-project/datasets")
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0]

train_idx = split_idx['train']
train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                               sizes=[15, 10, 5], batch_size=1024,
                               shuffle=True, num_workers=12)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=4096, shuffle=False,
                                  num_workers=12)

train_device = f'cuda:{args.train_device}' if torch.cuda.is_available() else 'cpu'
model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)

model = model.to(train_device)


def train(model, device):
    model.train()

    print(torch.cuda.memory_stats(device)["allocated_bytes.all.peak"])
    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        print(torch.cuda.memory_stats(device)["allocated_bytes.all.peak"])
        # pbar.update(batch_size)

    # pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)
    print(torch.cuda.memory_stats(device)["allocated_bytes.all.peak"])
    
    return loss, approx_acc


# 

@torch.no_grad()
def test(model, device):
    model.eval()
    print(torch.cuda.memory_stats(device)["allocated_bytes.all.peak"])
    out = model.inference(x, subgraph_loader, device)
    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    print(torch.cuda.memory_stats(device)["allocated_bytes.all.peak"])
    return train_acc, val_acc, test_acc

# test
# result = test()
# train_acc, val_acc, test_acc = result
# print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
#     f'Test: {test_acc:.4f}')