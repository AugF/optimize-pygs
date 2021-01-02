'''
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

from models.sage import SAGE
from utils.logger import Logger


parser = argparse.ArgumentParser(description='Neighborsampling(SAGE)')
parser.add_argument('--device', type=int, default=0)
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

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)
model = model.to(device)

x = data.x.to(device)
y = data.y.squeeze().to(device)

def train(epoch):
    model.train()

    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

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
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

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

    return train_acc, val_acc, test_acc


test_accs = []
for run in range(args.runs):
    print('')
    print(f'Run {run:02d}:')
    print('')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    best_val_acc = final_test_acc = 0
    for epoch in range(1, 1 + args.epochs):
        loss, acc = train(epoch)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')

        if epoch > 5:
            result = test()
            train_acc, val_acc, test_acc = result
            logger.add_result(run, result)
            print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
    test_accs.append(final_test_acc)
    logger.print_statistics(run)
    
logger.print_statistics()
test_acc = torch.tensor(test_accs)
print('============================')
print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')