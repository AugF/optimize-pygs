import torch
from optimize_pygs.loaders import BaseSampler
from optimize_pygs.trainers.sampled_trainer import SampledTrainer


class GraphSampledTrainer(SampledTrainer):
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
        self.optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = model.loss_fn(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        acc = model.evaluator(logits[data.train_mask], data.y[data.train_mask])
        return acc, loss.item()    

    @torch_no_grad
    def _test_step(self, model, data, split): #??
        logits = model(data.x, data.edge_index)
        mask = getattr(data, split + "_mask")
        loss = model.loss_fn(logits[mask], data.y[mask])
        loss.backward()
        acc = model.evaluator(logits[mask], data.y[mask])
        return acc, loss.item()    