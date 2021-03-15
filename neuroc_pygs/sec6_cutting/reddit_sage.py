# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/reddit.py
import os, copy, sys
import time
import argparse
import os.path as osp
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from neuroc_pygs.utils import get_dataset
from neuroc_pygs.configs import PROJECT_PATH


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser(description='OGBN-Products (Cluster-GCN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--infer_batch_size', type=int, default=1024)
    args = parser.parse_args()
    return args


def prepare_data(args):
    dataset = get_dataset('reddit', normalize_features=True)
    data = dataset[0]

    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                sizes=[25, 10], batch_size=1024, shuffle=True,
                                num_workers=12)
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                    batch_size=args.infer_batch_size, shuffle=False,
                                    num_workers=12)
    return dataset, train_loader, subgraph_loader


def prepare_model_optimizer(dataset):
    model = SAGE(dataset.num_features, 256, dataset.num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return model, optimizer


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGE, self).__init__()

        self.num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, subgraph_loader, df=None):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

                if i == 0:
                    if df is not None:
                        df['nodes'].append(size[0])
                        df['edges'].append(edge_index.shape[1])
                        df['memory'].append(torch.cuda.memory_stats(device)["allocated_bytes.all.peak"])
                    torch.cuda.reset_max_memory_allocated(device)
                    torch.cuda.empty_cache()

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def train(epoch, model, data, x, y, train_loader, optimizer):
    model.train()

    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:

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
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc


@torch.no_grad()
def test(model, data, x, y, subgraph_loader, df=None):
    model.eval()

    out = model.inference(x, subgraph_loader, df)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

    return results


def fit(model, optimizer, train_loader, data, subgraph_loader):
    x = data.x.to(device)
    y = data.y.squeeze().to(device)
    best_model = None 
    best_val_acc = 0
    final_test_acc = 0

    for epoch in range(1, 11):
        loss, acc = train(epoch, model, data, x, y, train_loader, optimizer)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        train_acc, val_acc, test_acc = test(model, data, x, y, subgraph_loader)
        print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            f'Test: {test_acc:.4f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc
            best_model = copy.deepcopy(model)
                    
    torch.save(best_model.state_dict(), os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_res',  'reddit_sage.pth'))


def run_fit():
    args = get_args()
    dataset, train_loader, subgraph_loader = prepare_data(args)
    model, optimizer = prepare_model_optimizer(dataset)
    data = dataset[0]
    fit(model, optimizer, train_loader, data, subgraph_loader)


def run_test():
    from collections import defaultdict
    args = get_args()
    print(args)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    dataset, train_loader, subgraph_loader = prepare_data(args)
    model, optimizer = prepare_model_optimizer(dataset)
    data = dataset[0]
    x = data.x.to(device)
    y = data.y.squeeze().to(device)
    save_dict = torch.load(os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_res',  'reddit_sage.pth'))
    model.load_state_dict(save_dict)
    df = defaultdict(list)
    test_accs = []
    times = []
    for _ in range(40):
        if _ * len(subgraph_loader) >= 40:
            break
        t1 = time.time()
        train_acc, val_acc, test_acc = test(model, data, x, y, subgraph_loader, df)
        t2 = time.time()
        test_accs.append(test_acc)
        times.append(t2 - t1)
    test_accs = np.array(test_accs)
    times = np.array(times)
    print(f'{test_accs.mean():.6f} ± {test_accs.std():.6f}')
    print(f'{times.mean():.6f} ± {times.std():.6f}')
    pd.DataFrame(df).to_csv(os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_res', f'reddit_sage_{args.infer_batch_size}_v1.csv'))
    peak_memory = list(map(lambda x: x / (1024 * 1024 * 1024), df['memory']))
    print(f'max: {max(peak_memory)}, min: {np.min(peak_memory)}, medium: {np.median(peak_memory)}, diff: {max(peak_memory)-min(peak_memory)}')
    return test_accs, times


if __name__ == '__main__':
    tab_data = []
    for bs in [1024, 2048, 4096, 8192, 16384]:
        sys.argv = [sys.argv[0], '--infer_batch_size', str(bs), '--device', '1']
        test_accs, times = run_test()
        tab_data.append([str(bs)] + list(test_accs) + list(times))
    pd.DataFrame(tab_data).to_csv(os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_res', f'reddit_sage_acc.csv'))

