import torch
import tqdm
import numpy as np
from optimize_pygs.loaders import BaseSampler
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
            all_loss.append(loss.item())
        return np.mean(all_acc), np.mean(all_loss)

    @torch_no_grad
    def test_step(self, model, data, split, loader=None):
        model.eval()
        all_acc = []
        all_loss = []
        num_batches = loader.get_num_batches()
        loader.reset_iter()
        for i in range(num_batches):
            batch = loader.get_next_batch()
            acc, loss = self._test_step(model, batch, split)
            all_acc.append(acc)
            all_loss.append(loss.item())
        return np.mean(all_acc), np.mean(all_loss)  
    
    @torch_no_grad()
    def infer_step(self, model, data, loader=None):
        model.eval()
        y_pred = model.inference(data.x, loader)
        y_true = data.y.cpu()

        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            correct = y_pred[mask].eq(y_true[mask]).sum().item()
            accs.append(correct / mask.sum().item())
        return accs

    def _train_step(self, model, data):
        return NotImplementedError
    
    def _test_step(self, model, data, split):
        return NotImplementedError
            
