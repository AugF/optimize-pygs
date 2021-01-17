import time

import torch
import torch.nn.functional as F
import numpy as np

from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from code.models.sage import SAGE
from code.optimize_batch.utils import get_args

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

cluster_data = ClusterData(data, num_parts=args.num_partitions,
                            recursive=False, save_dir=dataset.processed_dir)

loader = ClusterLoader(cluster_data, batch_size=1500,
                        shuffle=True, num_workers=args.num_workers)

model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)



