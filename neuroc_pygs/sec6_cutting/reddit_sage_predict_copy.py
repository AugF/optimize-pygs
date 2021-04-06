# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/reddit.py
import os, copy, sys
import time, gc
import argparse
import os.path as osp
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from joblib import load
from neuroc_pygs.utils import get_dataset
from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.sec6_cutting.cutting_methods import cut_by_importance_reverse, cut_by_random, get_degree, get_pagerank
from neuroc_pygs.sec6_cutting.cutting_utils import BSearch


dir_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec6_cutting/exp_diff_res'
reg = load(dir_path + '/reddit_sage_linear_model_v0.pth')
memory_ratio = pd.read_csv(dir_path + '/regression_mape_res.csv', index_col=0)['reddit_sage']['mape'] - 0.05
memory_limit = 3 * 1024 * 1024 * 1024 # 3221225472
bsearch = BSearch(clf=reg, memory_limit=memory_limit)
print(f'memory_ratio: {memory_ratio}, memory_limit: {memory_limit}')
pr, degree = None, None

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
        # pbar = tqdm(total=x_all.size(0) * self.num_layers)
        # pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            first_flag = True
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj
                if i == 0:
                    torch.cuda.reset_max_memory_allocated(device)
                    torch.cuda.empty_cache()
                    current_memory = torch.cuda.memory_stats(device)["allocated_bytes.all.current"]

                    node, edge = size[0], edge_index.shape[1]
                    memory_pre = reg.predict([[node, edge]])[0]
                    if first_flag:
                        real_memory_ratio = memory_ratio + 0.2
                        first_flag = False
                    else:
                        real_memory_ratio = memory_ratio
                    predict_peak = memory_pre / (1 - real_memory_ratio) + current_memory
                    if predict_peak > memory_limit:
                        print(f'{node}, {edge}, {predict_peak}, begin cutting')
                        st1 = time.time()
                        cutting_nums = bsearch.get_cutting_nums(node, edge, real_memory_ratio, current_memory)
                        print(f'cutting {cutting_nums} edges...')
                        if args.cutting_method == 'random':
                            edge_index = cut_by_random(edge_index, cutting_nums, seed=int(args.cutting_way))
                        else:
                            edge_index = cut_by_importance_method(edge_index, cutting_nums, method=args.cutting_method, name=args.cutting_way, degree=degree[n_id], pr=pr[n_id])
                        st2 = time.time()
                        print(f'cutting use time {st2 - st1}s')
                        
                edge_index = edge_index.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                # pbar.update(batch_size)

                if i == 0:
                    if df is not None:
                        node, edge = size[0], edge_index.shape[1]
                        memory = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]
                        df['nodes'].append(node)
                        df['edges'].append(edge)
                        df['memory'].append(memory)
                        df['diff_memory'].append(memory - current_memory)
                        df['predict'].append(memory_pre)
                        df['predict_peak'].append(predict_peak)
                        print(f'nodes={node}, edge={edge}, predict: {memory_pre}-{predict_peak}, real: {memory}, diff: {memory - current_memory}, device: {device}')

            x_all = torch.cat(xs, dim=0)

        # pbar.close()

        return x_all


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


def run_test():
    global pr, degree
    args = get_args()
    print(args)
    real_path = os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_diff_res', f'reddit_sage_{args.infer_batch_size}_opt_{args.cutting_method}_{args.cutting_way}_reverse_v2.csv')
    test_accs = []
    times = []
    print(real_path)
    if os.path.exists(real_path):
        return test_accs, times
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    args.device = device
    dataset, train_loader, subgraph_loader = prepare_data(args)
    model, optimizer = prepare_model_optimizer(dataset, device)
    data = dataset[0]
    pr, degree = get_pagerank(data.edge_index), get_degree(data.edge_index)
    
    x = data.x.to(device)
    y = data.y.squeeze().to(device)
    save_dict = torch.load(os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_diff_res',  'reddit_sage_best_model.pth'))
    model.load_state_dict(save_dict)
    df = defaultdict(list)
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
    pd.DataFrame(df).to_csv(real_path)
    peak_memory = list(map(lambda x: x / (1024 * 1024 * 1024), df['memory']))
    print(f'max: {max(peak_memory)}, min: {np.min(peak_memory)}, medium: {np.median(peak_memory)}, diff: {max(peak_memory)-min(peak_memory)}')
    return test_accs, times


if __name__ == '__main__':
    for bs in [8700, 8800, 8900]:
        tab_data = []
        for cutting in ['random_2', 'degree_way1', 'degree_way2', 'pr_way1', 'pr_way2']:
            method, way = cutting.split('_')
            sys.argv = [sys.argv[0], '--infer_batch_size', str(bs), '--device', '1', '--cutting_method', method, '--cutting_way', way]
            test_accs, times = run_test()
            tab_data.append([str(bs), cutting] + list(test_accs) + list(times))
            gc.collect()
        pd.DataFrame(tab_data).to_csv(os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_opt_res', f'reddit_sage_opt_{bs}_reverse_v2.csv'))
