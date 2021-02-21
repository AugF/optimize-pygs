from typing import Optional, Type, Any

import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model_from_args(cls, args):
        raise NotImplementedError("Models must implement the build_model_from_args")
     
    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = "" # 由用户指定
        self.loss_fn = None # 绑定到dataset上, test_step会用到
        self.evaluator = None
    
    def forward(self, *args):
        raise NotImplementedError
    
    def predict(self, data):
        return self.forward(data.x, data.edge_index)
    
    def node_classification_loss(self, data, mask=None):
        if mask is None:
            mask = data.train_mask
        edge_index = data.edge_index_train if hasattr(data, "edge_index_train") and self.training else data.edge_index # inductive?
        pred = self.forward(data.x, edge_index)
        return self.loss_fn(pred[mask], data.y[mask])
    
    def graph_classification_loss(self, batch):
        pred = self.forward(batch)
        return self.loss_fn(pred, batch.y)
    
    def set_device(self, device):
        self.device = device
    
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
    
    def set_evaluator(self, evaluator):
        self.evaluator = evaluator