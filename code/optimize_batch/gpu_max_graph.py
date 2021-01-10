"""
测试sampler, to, training三部分在资源受限下最多花的内存
以ogbn_products为例
"""

dataset_name = 'ogbn-products'
dataset = PygNodePropPredDataset(name=dataset_name, root="/home/wangzhaokang/wangyunpan/gnns-project/datasets")
split_idx = dataset.get_idx_split()
data = dataset[0]
print(f"{dataset_name}: node, edge = {data.num_nodes}, {data.num_edges}")

for key, idx in split_idx.items():
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[idx] = True
    data[f'{key}_mask'] = mask

model = SAGE(data.x.size(-1), args.hidden_channels, dataset.num_classes,
                args.num_layers, args.dropout).to(device)

cluster_data = ClusterData(data, num_parts=args.num_partitions,
                            recursive=False, save_dir=dataset.processed_dir)

# 测试什么规模下的数据最大
loader = ClusterLoader(cluster_data, batch_size=1500,
                        shuffle=True, num_workers=args.num_workers)

model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
train_base(model, loader, optimizer, device)
