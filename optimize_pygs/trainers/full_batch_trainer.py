import torch
import tqdm
import numpy as np
from optimize_pygs.trainers import BaseTrainer, register_trainer


@register_trainer("full")
class FullBatchTrainer(BaseTrainer):
    @staticmethod
    def add_args(parser):
        """Add trainer-specific arguments to the parser."""
        pass
    
    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def __init__(self, args): # must args
        super().__init__(args)

    def train_step(self, model, data, loader=None):
        assert loader == None
        self.optimizer.zero_grad()
        logits = model(data)
        loss = model.loss_fn(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        acc = model.evaluator(logits[data.train_mask], data.y[data.train_mask])
        return acc, loss.item()    

    def test_step(self, model, data, split, loader=None):
        assert loader == None
        with torch.no_grad():
            logits = model.predict(data)
            mask = getattr(data, split + "_mask") # 魔法方法
            loss = model.loss_fn(logits[mask], data.y[mask])
            acc = model.evaluator(logits[mask], data.y[mask]) 
        return acc, loss.item()
    
            
