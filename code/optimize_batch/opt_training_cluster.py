import time
import torch

from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from code.models.sage import SAGE
from code.optimize_batch.utils import get_args, train_base

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
loss, train_acc = train_base(model, loader, optimizer, device)
print(f'loss: {loss:.4f}, train_acc: {train_acc:.4f}')

