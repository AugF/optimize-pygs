import torch
from optimize_pygs.data.cluster import ClusterData, ClusterLoader
from optimize_pygs.loaders import BaseSampler, register_sampler


@register_sampler("cluster")
class ClusterSampler(BaseSampler):
    @staticmethod
    def add_args(parser):
        """Add sampler-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--sampler_data", type=torch.utils.data.Dataset)
        parser.add_argument("--recursive", type=bool, default=False)
        parser.add_argument("--shuffle", type=bool, default=True)
        parser.add_argument("--num-parts", type=int, default=1500)
        parser.add_argument("--batch_size", type=int, default=20)
        parser.add_argument("--num-workers", type=int, default=40)
        # fmt: on
    
    @classmethod
    def build_sampler_from_args(cls, args):
        return cls(
            # ClusterData
            args.sampler_data,
            args.num_parts,
            args.recursive,
            # ClusterLoader
            args.batch_size,
            args.shuffle,
            num_workers=args.num_workers
        )

    def __init__(self, dataset, num_parts, recursive, batch_size, shuffle, num_workers=0):
        cluster_data = ClusterData(dataset[0], num_parts=num_parts, 
        recursive=recursive, save_dir=dataset.processed_dir)
        loader = ClusterLoader(cluster_data, batch_size=batch_size, 
                    shuffle=shuffle, num_workers=num_workers)
        super().__init__(loader)
    
    def get_next_batch(self): # batch is a data
        batch = next(self.iter)
        return batch