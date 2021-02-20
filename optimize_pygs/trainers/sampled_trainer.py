from typing import Optional, Type, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np 
import copy 
import tqdm

from .base_trainer import BaseTrainer
from optimize_pygs.utils import add_remaining_self_loops
from optimize_pygs.data.neighbor import NeighborSampler
from optimize_pygs.data.cluster import ClusterData, ClusterLoader

class SampledTrainer(BaseTrainer):
    @abstractmethod
    def fit(self, model, dataset):
        raise NotImplementedError
    
    @abstractmethod
    def _train_step(self):
        pass
    
    @abstractmethod
    def _test_step(self, split="val"):
        pass
    
    def __init__(self, args):
        self.device = "cpu" if not torch.cuda.is_availabel() or args.cpu else args.device_id[0]
        self.patience = args.patience 
        self.max_epoch = args.max_epoch
        self.lr = args.lr
        self.weight_decay = args.weight_decay
    
    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def train(self):
        epoch_iter = tqdm(range(self.max_epoch))
        patience = 0
        best_score = 0
        # best_loss = np.inf
        max_score = 0
        min_loss = np.inf
        best_model = copy.deepcopy(self.model)
        for epoch in epoch_iter:
            self.__train_step()
            train_acc, _ = self._test_step(split='train')
            val_acc, val_loss = self._test_step(split="val")
            epoch_iter.set_decription(f"Epoch: {epoch: 03d}, Train: {train_acc: .4f}, Val: {val_acc: .4f}")
            if val_loss <= min_loss or val_acc >= max_score:
                if val_acc >= best_score:
                    # best_loss = val_loss
                    best_score = val_acc
                    best_model = copy.deepcopy(self.model)
            else:
                patience += 1
                if patience == self.patience:
                    self.model = best_model
                    epoch_iter.close()
                    break
        return best_model
    

class NeighborSamplerTrainer(SampledTrainer):
    model: torch.nn.Module 
    
    def __init__(self, args):
        super(NeighborSamplerTrainer, self).__init__(args)
        self.hidden_size = args.hidden_size
        self.sample_size = args.sample_size
        self.batch_size = args.batch_size
        self.num_workers = 4 if not hasattr(args, "num_workers") else args.num_workers
        self.eval_per_epoch = 5
        self.patience = self.patience // self.eval_per_epoch
        
        self.device = "cpu" if not torch.cuda.is_availabel() or args.cpu else args.device_id[0]
        self.loss_fn, self.evaluator = None, None
    
    def fit(self, model, dataset):
        self.data = dataset[0]
        self.data.edge_index, _ = add_remaining_self_loops(self.data.edge_index)
        if hasattr(self.data, "edge_index_train"):
            self.data.edge_index_train, _ = add_remaining_self_loops(self.data.edge_index)
        # evaluator, loss_fn绑定到了具体的数据集上
        self.evaluator = dataset.get_evaluator()
        self.loss_fn = dataset.get_loss_fn()
        
        # 这里直接使用了train_sampler, 因为本文中涉及到的只是这两个sampler, 所以不需要更多的
        self.train_loader = NeighborSampler(
            
        )
        self.test_loader = NeighborSampler(
            
        )
        
        self.model = model.to(self.device)
        self.model.set_data_device(self.device)
        # 这里的optimizer固定，没有开放给用户
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_model = self.train()
        self.model = best_model
        acc, loss = self._test_step()
        acc, loss = self._test_step()
        return dict(Acc=acc["test"], ValAcc=acc["val"])

    def train(self):
        epoch_iter = tqdm(range(self.max_epoch))
        patience = 0
        max_score = 0
        min_loss = np.inf
        best_model = copy.deepcopy(self.model)
        for epoch in epoch_iter:
            self._train_step()
            if (epoch + 1) % self.eval_per_epoch == 0:
                acc, loss = self._test_step()
                train_acc = acc["train"]
                val_acc = acc["val"]
                val_loss = loss["val"]
                epoch_iter.set_description(f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}")
                if val_loss <= min_loss or val_acc >= max_score:
                    if val_loss <= min_loss:
                        best_model = copy.deepcopy(self.model)
                    min_loss = np.min((min_loss, val_loss))
                    max_score = np.max((max_score, val_acc))
                    patience = 0
                else:
                    patience += 1
                    if patience == self.patience:
                        self.model = best_model
                        epoch_iter.close()
                        print("early stopping !!!")
                        break
        return best_model
    
    def _train_step(self):
        self.model.train()
        # loader的不同之处，可以进行抽象和封装
        for target_id, n_id, adjs in self.train_loader:
            self.optimizer.zero_grad()
            x_src = self.data.x[n_id].to(self.device)
            y = self.data.y[target_id].to(self.device)
            # TODO: 检查这里的问题，是否符合全文的设计
            loss = self.model.node_classification_loss(x_src, adjs, y) 
            loss.backward()
            self.optimizer.step()
    
    def _test_step(self, split="val"):
        self.model.eval()
        masks = {"train": self.data.train_mask, "val": self.data.val_mask, "test": self.data.test_mask}
        with torch.no_grad(): # way1
            logits = self.model.inference(self.data.x, self.test_loader)
            
        loss = {key: self.loss_fn(logits[val], self.data.y[val]) for key, val in masks.items()}
        acc = {key: self.evaluator(logits[val], self.data.y[val]) for key, val in masks.items()}
        return acc, loss
    
    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)