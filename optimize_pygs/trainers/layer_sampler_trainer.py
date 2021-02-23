import torch
from optimize_pygs.loaders import BaseSampler
from optimize_pygs.trainers.sampled_trainer import SampledTrainer


class LayerSampledTrainer(SampledTrainer):
    @staticmethod
    def add_args(parser):
        """Add trainer-specific arguments to the parser."""
        pass
    
    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def __init__(self, args): # must args
        super().__init__(args)
    
    def _train_step(self, model, data):
        """
        {'batch_size':, 'x':, 'edge_index':, 'y':}
        """
        batch_size, x, adjs, y = data['batch_size'], data['x'], data['adjs'], data['y']
        self.optimizer.zero_grad()
        logits = model(x, adjs)
        loss = model.loss_fn(logits, y)
        loss.backward()
        self.optimizer.step()
        acc = model.evaluator(logits, y) / batch_size
        return acc, loss.item()
                
    @torch_no_grad
    def _test_step(self, model, data, split):
        batch_size, x, adjs, y = data['batch_size'], data['x'], data['adjs'], data['y']
        logits = model(x, adjs)
        loss = model.loss_fn(logits, y)
        acc = model.evaluator(logits, y) / batch_size
        return acc, loss.item()