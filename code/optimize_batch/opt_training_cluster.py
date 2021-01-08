'''
https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/cluster_gcn.py master: cf066f9
report acc: 0.7897 ± 0.0033
rank: 11
2020-10-27
'''
import torch
from tqdm import tqdm
import torch.nn.functional as F

from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
from torch_geometric.nn import SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from code.models.sage import SAGE
from code.optimize_batch.utils import get_args

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



# ---- begin ----
# step1. get args
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

# step5. train
loss, train_acc = train(model, loader, optimizer, device)
print(f'loss: {loss:.4f}, train_acc: {train_acc:.4f}')

