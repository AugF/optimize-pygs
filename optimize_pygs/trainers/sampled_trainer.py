import torch
import numpy as np
from optimize_pygs.trainers.base_trainer import BaseTrainer


class SampledTrainer(BaseTrainer):
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
        model.train()
        all_acc = []
        all_loss = []
        num_batches = loader.get_num_batches()
        loader.reset_iter()
        for i in range(num_batches):
            batch = loader.get_next_batch()
            acc, loss = self._train_step(model, batch)
            all_acc.append(acc)
            all_loss.append(loss)
        return np.mean(all_acc), np.mean(all_loss)

    def test_step(self, model, data, split, loader=None):
        with torch.no_grad():
            model.eval()
            all_acc = []
            all_loss = []
            num_batches = loader.get_num_batches()
            loader.reset_iter()
            for i in range(num_batches):
                batch = loader.get_next_batch()
                acc, loss = self._test_step(model, batch, split)
                all_acc.append(acc)
                all_loss.append(loss)
            return np.mean(all_acc), np.mean(all_loss)  
    
    def infer_step(self, model, data, split, loader=None):
        with torch.no_grad():
            model.eval()
            y_pred = model.inference(data.x, loader)
            y_true = data.y.cpu()
            
            mask = getattr(data, split + "_mask")
            loss = model.loss_fn(y_pred[mask], y_true[mask])
            acc = model.evaluator(y_pred[mask], y_true[mask]) 
                
        return acc, loss

    def _train_step(self, model, data):
        return NotImplementedError
    
    def _test_step(self, model, data, split):
        return NotImplementedError
            
