'''
https://github.com/Chillee/ogb_baselines/tree/master/ogbn_products master: 25e793b
report acc: 0.7971 ± 0.0042
rank: 6
date: 2020-10-27 
'''
import os, sys, time
import argparse

import torch
import copy
import numpy as np
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm
from joblib import load
from neuroc_pygs.sec6_cutting.cutting_practise import BSearch

from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.sec6_cutting.cutting_methods import cut_by_importance



dir_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec6_cutting/exp_res'
reg = load(dir_path + '/cluster_gcn_linear_model_v1.pth')
memory_limit = 2.3 * 1024 * 1024 * 1024
bsearch = BSearch(clf=reg, memory_limit=memory_limit, ratio=0)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()
        self.inProj = torch.nn.Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.linear = torch.nn.Linear(hidden_channels, out_channels)
        self.weights = torch.nn.Parameter(torch.randn((len(self.convs))))
        self.dropout = dropout

    def reset_parameters(self):
        self.inProj.reset_parameters()
        self.linear.reset_parameters()
        torch.nn.init.normal_(self.weights)
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.inProj(x)
        inp = x
        x = F.relu(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x + 0.2*inp
 
        return torch.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, args, df=None):
        device = args.device
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')
        
        x_all = self.inProj(x_all.to(device))
        x_all = x_all.cpu()
        inp = x_all
        x_all = F.relu(x_all)
        out = []
        for i, conv in enumerate(self.convs):
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
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
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
            if i != len(self.convs) - 1:
                x_all = x_all + 0.2*inp
        pbar.close()

        return x_all

def train(model, loader, optimizer, device):
    model.train()

    total_loss = total_examples = 0
    total_correct = total_examples = 0
    for data in loader:
        data = data.to(device)
        if data.train_mask.sum() == 0:
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


@torch.no_grad()
def test(model, data, evaluator, subgraph_loader, args, df=None):
    model.eval()

    out = model.inference(data.x, subgraph_loader, args, df)

    y_true = data.y
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[data.train_mask],
        'y_pred': y_pred[data.train_mask]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[data.valid_mask],
        'y_pred': y_pred[data.valid_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[data.test_mask],
        'y_pred': y_pred[data.test_mask]
    })['acc']

    return train_acc, valid_acc, test_acc

def process_adj(data):
    N = data.num_nodes
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row, col = data.edge_index

    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    return adj


def fit(model, data, loader, subgraph_loader, evaluator, optimizer, args):
    best_val_acc = 0
    final_test_acc = 0
    best_model = None
    for epoch in range(1, 1 + args.epochs):
        loss, train_acc = train(model, loader, optimizer, args.device)
        if epoch % args.log_steps == 0:
            print(f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Approx Train Acc: {train_acc:.4f}')

        if epoch > 19 and epoch % args.eval_steps == 0:
            train_acc, valid_acc, test_acc = test(model, data, evaluator, subgraph_loader, args)
            print(f'Epoch: {epoch:02d}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}% '
                    f'Test: {100 * test_acc:.2f}%')

            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                final_test_acc = test_acc
                best_model = copy.deepcopy(model)

    torch.save(best_model.state_dict(), os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_res',  'cluster_gcn.pth'))



def get_args():
    parser = argparse.ArgumentParser(description='OGBN-Products (Cluster-GCN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_partitions', type=int, default=15000)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--infer_batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--cutting_method', type=str, default='degree')
    parser.add_argument('--cutting_way', type=str, default='way3')
    args = parser.parse_args()
    return args


def prepare_data(args):
    root = '/mnt/data/wangzhaokang/wangyunpan/datasets'
    dataset = PygNodePropPredDataset(name='ogbn-products', root=root)
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    # Convert split indices to boolean masks and add them to `data`.
    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask
    args.num_classes = dataset.num_classes
    args.processed_dir = dataset.processed_dir
    return args, data


def prepare_loader(args, data):
    cluster_data = ClusterData(data, num_parts=args.num_partitions,
                               recursive=False, save_dir=args.processed_dir)

    loader = ClusterLoader(cluster_data, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)
    return loader


def run_fit():
    from collections import defaultdict
    args = get_args()
    print(args)
    args, data = prepare_data(args)
    loader = prepare_loader(args, data)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(device)
    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                      batch_size=args.infer_batch_size, shuffle=False,
                                      num_workers=args.num_workers)
    model = GCN(data.x.size(-1), args.hidden_channels, args.num_classes,
                 args.num_layers, args.dropout).to(args.device)

    evaluator = Evaluator(name='ogbn-products')
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    fit(model, data, loader, subgraph_loader, evaluator, optimizer, args)
    

def run_test():
    from collections import defaultdict
    args = get_args()
    print(args)
    args, data = prepare_data(args)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    args.device = device
    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                      batch_size=args.infer_batch_size, shuffle=False,
                                      num_workers=args.num_workers)

    model = GCN(data.x.size(-1), args.hidden_channels, args.num_classes,
                 args.num_layers, args.dropout).to(args.device)

    evaluator = Evaluator(name='ogbn-products')
   
    model.reset_parameters()
    save_dict = torch.load(os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_res',  'cluster_gcn.pth'))
    model.load_state_dict(save_dict)

    df = defaultdict(list)
    test_accs = []
    times = []
    for _ in range(40):
        if _ * len(subgraph_loader) >= 40:
            break
        t1 = time.time()
        train_acc, valid_acc, test_acc = test(model, data, evaluator, subgraph_loader, args, df)
        t2 = time.time()
        test_accs.append(test_acc)
        times.append(t2 - t1)
    test_accs = np.array(test_accs)
    times = np.array(times)
    print(f'{test_accs.mean():.6f} ± {test_accs.std():.6f}')
    print(f'{times.mean():.6f} ± {times.std():.6f}')
    pd.DataFrame(df).to_csv(os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_res', f'cluster_gcn_{args.infer_batch_size}_v1.csv'))
    peak_memory = list(map(lambda x: x / (1024 * 1024 * 1024), df['memory']))
    print(f'max: {max(peak_memory)}, min: {np.min(peak_memory)}, medium: {np.median(peak_memory)}, diff: {max(peak_memory)-min(peak_memory)}')
    return test_accs, times


if __name__ == "__main__":
    for cutting in ['degree_way3', 'degree_way4', 'pagerank_way3', 'pagerank_way4']:
        method, way = cutting.split('_')
        tab_data = []
        for bs in [1024, 2048, 4096, 8192]:
            sys.argv = [sys.argv[0], '--infer_batch_size', str(bs), '--device', '2', '--cutting_method', method, '--cutting_way', way]
            test_accs, times = run_test()
            tab_data.append([str(bs)] + list(test_accs) + list(times))
        pd.DataFrame(tab_data).to_csv(os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_opt_res', f'cluster_gcn_linear_model_{cutting}_v2.csv'))
