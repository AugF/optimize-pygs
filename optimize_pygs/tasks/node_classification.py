import argparse
import copy
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from optimize_pygs.datasets import build_dataset
from optimize_pygs.models import build_model
from . import BaseTask, register_task


@register_task("node classification")
class NodeClassification(BaseTask):
    """Node classification task."""
    
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add task-specific arguments to the parser"""
        
    def __init__(
        self,
        args,
        dataset=None,
        model=None
    ):
        super(NodeClassification, self).__init__(args)
        
        self.args = args
        self.model_name = args.module()
        
        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]
        dataset = build_dataset(args) if dataset is None else dataset

        self.dataset = dataset
        self.data = dataset[0]
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        args.num_nodes = dataset.data.x.shape[0]
        
        self.model = build_model(args) if model is None else model
        self.model.set_device(self.device)
        
        self.set_loss_fn(dataset)
        self.set_evaluator(dataset)
        
        self.trainer = (
            self.model.get_trainer(NodeClassification, self.args)(self.args)
            if self.model.get_trainer(NodeClassification, self.args)
            else None
        ) # 忽略其不存在的情况
        
    def train(self):
        if self.trainer:
            result = self.trainer.fit(self.model, self.dataset)
            if issubclass(type(result), torch.nn.Module):
                self.model = result
            else:
                return result    
        else:
            epoch_iter = tqdm(range(self.max_epoch))
            patience = 0
            best_score = 0
            best_loss = np.inf
            max_score = 0
            min_loss = np.inf
            best_model = copy.deepcopy(self.model)
            for epoch in epoch_iter:
                self._train_step()
                train_acc, _ = self._test_step(split="train")
                val_acc, val_loss = self._test_step(split="val")
                epoch_iter.set_description(f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}")
                if val_loss <= min_loss or val_acc >= max_score:
                    if val_loss <= best_loss:  # and val_acc >= best_score:
                        best_loss = val_loss
                        best_score = val_acc
                        best_model = copy.deepcopy(self.model)
                    min_loss = np.min((min_loss, val_loss))
                    max_score = np.max((max_score, val_acc))
                    patience = 0
                else:
                    patience += 1
                    if patience == self.patience:
                        epoch_iter.close()
                        break
            print(f"Valid accurracy = {best_score: .4f}")
            self.model = best_model
        test_acc, _ = self._test_step(split="test")
        val_acc, _ = self._test_step(split="val")
        print(f"Test accuracy = {test_acc:.4f}")
        return dict(Acc=test_acc, ValAcc=val_acc)

    def _train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.node_classification_loss(self.data).backward()
        self.optimizer.step()

    def _test_step(self, split="val", logits=None):
        self.model.eval()
        with torch.no_grad():
            logits = logits if logits else self.model.predict(self.data)
        if split == "train":
            mask = self.data.train_mask
        elif split == "val":
            mask = self.data.val_mask
        else:
            mask = self.data.test_mask
        loss = self.loss_fn(logits[mask], self.data.y[mask])
        metric = self.evaluator(logits[mask], self.data.y[mask])
        return metric, loss