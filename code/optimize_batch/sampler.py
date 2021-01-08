from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler


def get_train_loader(sampler_name, sampler_args, shuffle=True): # sampler_args
    """"""
    if sampler_name == "cluster_sampler":
        cluster_data = ClusterData(data, num_parts=args['cluster_partitions, recursive=False,
                            save_dir=dataset.processed_dir)
        train_loader = ClusterLoader(cluster_data, batch_size=batch_partitions, shuffle=True,
                                num_workers=args.num_workers)
    elif sampler_name == "neighbor_sampler":
        pass
    else:
        return
sampler_args = {
    'node_idx': train_idx,
    'data'
}

train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                               sizes=[15, 10, 5], batch_size=1024,
                               shuffle=True, num_workers=12)