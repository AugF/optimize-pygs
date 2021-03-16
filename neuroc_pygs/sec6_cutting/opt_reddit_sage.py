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
from joblib import load
from neuroc_pygs.sec6_cutting.cutting_practise import BSearch
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from neuroc_pygs.utils import get_dataset
from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.sec6_cutting.cutting_methods import cut_by_importance


dir_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec6_cutting/exp_res'
reg = load(dir_path + '/reddit_sage_linear_model_v1.pth')
memory_limit = 2.3 * 1024 * 1024 * 1024
bsearch = BSearch(clf=reg, memory_limit=memory_limit, ratio=0)

def get_args():
    parser = argparse.ArgumentParser(description='OGBN-Products (Cluster-GCN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--infer_batch_size', type=int, default=1024)
    parser.add_argument('--cutting_method', type=str, default='degree')
    parser.add_argument('--cutting_way', type=str, default='way3')
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


def prepare_model_optimizer(dataset, device):
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

    def inference(self, x_all, subgraph_loader, args, df=None):
        device = args.device
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj
                
                node, edge = size[0], edge_index.shape[1]
                if i == 0:
                    memory_pre = reg.predict([[node, edge]])[0]
                    while memory_pre > memory_limit:
                        print(f'{node}, {edge}, {memory_pre}, begin cutting')
                        cutting_nums = bsearch.get_proper_edges(nodes=node, edges=edge)
                        edge_index = cut_by_importance(edge_index, cutting_nums, method=args.cutting_method, name=args.cutting_way)
                        memory_pre = reg.predict([[node, edge]])[0]
                
                edge_index = edge_index.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

                if i == 0:
                    if df is not None:
                        df['nodes'].append(node)
                        df['edges'].append(edge)
                        memory = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]
                        df['memory'].append(memory)
                        print(f'nodes={node}, edge={edge}, predict: {memory_pre}, real: {memory}')
                    torch.cuda.reset_max_memory_allocated(device)
                    torch.cuda.empty_cache()

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def train(epoch, model, data, x, y, train_loader, optimizer, device):
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
def test(model, data, x, y, subgraph_loader, args, df=None):
    model.eval()

    out = model.inference(x, subgraph_loader, args, df)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

    return results


def fit(model, optimizer, train_loader, data, subgraph_loader, args):
    x = data.x.to(args.device)
    y = data.y.squeeze().to(args.device)
    best_model = None 
    best_val_acc = 0
    final_test_acc = 0

    for epoch in range(1, 11):
        loss, acc = train(epoch, model, data, x, y, train_loader, optimizer, args.device)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        train_acc, val_acc, test_acc = test(model, data, x, y, subgraph_loader, args)
        print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            f'Test: {test_acc:.4f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc
            best_model = copy.deepcopy(model)
                    
    torch.save(best_model.state_dict(), os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_res',  'reddit_sage.pth'))


def run_fit():
    args = get_args()
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    dataset, train_loader, subgraph_loader = prepare_data(args)
    model, optimizer = prepare_model_optimizer(dataset, args.device)
    data = dataset[0]
    fit(model, optimizer, train_loader, data, subgraph_loader, args.device)


def run_test():
    from collections import defaultdict
    args = get_args()
    print(args)
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    dataset, train_loader, subgraph_loader = prepare_data(args)
    model, optimizer = prepare_model_optimizer(dataset, args.device)
    data = dataset[0]
    x = data.x.to(args.device)
    y = data.y.squeeze().to(args.device)
    save_dict = torch.load(os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_res',  'reddit_sage.pth'))
    model.load_state_dict(save_dict)
    df = defaultdict(list)
    test_accs = []
    times = []
    for _ in range(40):
        if _ * len(subgraph_loader) >= 40:
            break
        t1 = time.time()
        train_acc, val_acc, test_acc = test(model, data, x, y, subgraph_loader, args, df)
        t2 = time.time()
        test_accs.append(test_acc)
        times.append(t2 - t1)
    test_accs = np.array(test_accs)
    times = np.array(times)
    print(f'{test_accs.mean():.6f} ± {test_accs.std():.6f}')
    print(f'{times.mean():.6f} ± {times.std():.6f}')
    pd.DataFrame(df).to_csv(os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_opt_res', f'reddit_sage_{args.infer_batch_size}_v1.csv'))
    peak_memory = list(map(lambda x: x / (1024 * 1024 * 1024), df['memory']))
    print(f'max: {max(peak_memory)}, min: {np.min(peak_memory)}, medium: {np.median(peak_memory)}, diff: {max(peak_memory)-min(peak_memory)}')
    return test_accs, times


if __name__ == '__main__':
    for cutting in ['degree_way3', 'degree_way4', 'pagerank_way3', 'pagerank_way4']:
        method, way = cutting.split('_')
        tab_data = []
        for bs in [1024, 2048, 4096, 8192]:
            sys.argv = [sys.argv[0], '--infer_batch_size', str(bs), '--device', '1', '--cutting_method', method, '--cutting_way', way]
            test_accs, times = run_test()
            tab_data.append([str(bs)] + list(test_accs) + list(times))
        pd.DataFrame(tab_data).to_csv(os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_opt_res', f'reddit_sage_linear_model_{cutting}_v2.csv'))
