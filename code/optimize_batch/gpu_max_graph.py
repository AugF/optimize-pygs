"""
测试sampler, to, training三部分在资源受限下最多花的内存
以ogbn_products为例
"""

dataset_name = 
dataset = PygNodePropPredDataset(name='ogbn-products', root="/home/wangzhaokang/wangyunpan/gnns-project/datasets")
split_idx = dataset.get_idx_split()
data = dataset[0]
print(f"ogbn_products: node, edge = {data.num_nodes}, {data.num_edges}")

cluster_data = ClusterData(data, num_parts=args.num_partitions,
                            recursive=False, save_dir=dataset.processed_dir)

loader = ClusterLoader(cluster_data, batch_size=1500,
                        shuffle=True, num_workers=args.num_workers)
