from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler


def get_train_loader(dataset, sampler_name, **sampler_args): # sampler_args
    """封装采样方法"""
    train_loader = None
    if sampler_name == "cluster_sampler":
        for arg in ['num_parts', 'batch_size', 'shuffle', 'num_workers']:
            assert arg in sampler_args
        cluster_data = ClusterData(dataset[0], num_parts=sampler_args['num_parts'], recursive=False,
                            save_dir=dataset.processed_dir)
        train_loader = ClusterLoader(cluster_data, batch_size=sampler_args['batch_size'],
                                     shuffle=sampler_args['shuffle'], num_workers=sampler_args['num_workers'])
    elif sampler_name == "neighbor_sampler":
        for arg in ['node_idx', 'sizes', 'batch_size', 'shuffle', 'num_workers']:
            assert arg in sampler_args
        train_loader = NeighborSampler(dataset[0].edge_index, node_idx=sampler_args['node_idx'],
                               sizes=sampler_args['sizes'], batch_size=sampler_args['batch_size'],
                               shuffle=sampler_args['shuffle'], num_workers=sampler_args['num_workers'])
    return train_loader


from ogb.nodeproppred import PygNodePropPredDataset
dataset = PygNodePropPredDataset('ogbn-products', root="/home/wangzhaokang/wangyunpan/gnns-project/datasets")
train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                               sizes=[15, 10, 5], batch_size=1024,
                               shuffle=True, num_workers=12)