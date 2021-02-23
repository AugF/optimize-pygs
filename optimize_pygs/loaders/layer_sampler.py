import torch
from optimize_pygs.data.neighbor import NeighborSampler
from optimize_pygs.loaders import BaseSampler, register_sampler


@register_sampler("graphsage")
class NeighborSampler(BaseSampler):
    @staticmethod
    def add_args(parser):
        """Add sampler-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--data", type=torch.utils.data.Dataset)
        parser.add_argument("--sizes", type=list, default=[25, 10])
        parser.add_argument("--node_idx", type=str, default=None)
        parser.add_argument("--recursive", type=bool, default=False)
        parser.add_argument("--shuffle", type=bool, default=True)
        parser.add_argument("--batch_size", type=int, default=20)
        parser.add_argument("--num-workers", type=int, default=0)
        # fmt: on

    @classmethod
    def build_sampler_from_args(cls, args):
        return cls(
            args.data,
            args.node_idx,
            args.sizes,
            args.batch_size,
            args.shuffle,
            num_workers=args.num_workers
        )

    def __init__(self, data, node_idx, sizes, batch_size, shuffle, num_workers):
        self.data = data
        loader = NeighborSampler(data, node_idx=node_idx, sizes=sizes,
                    batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        super().__init__(loader)
    
    def get_next_batch(self):
        batch_size, n_id, adjs = next(self.iter)
        return {
            'batch_size': batch_size,
            'x': self.data.x[n_id],
            'edge_index': adjs,
            'y': self.data.y[n_id[:batch_size]],
        }