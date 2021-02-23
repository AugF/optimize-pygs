import torch
from optimize_pygs.trainers.sampled_trainer import SampledTrainer
from optimize_pygs.trainers import BaseTrainer, register_trainer


@register_trainer("graph")
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
        data = data.to(self.device)
        self.optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = model.loss_fn(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        acc = model.evaluator(logits[data.train_mask], data.y[data.train_mask])
        return acc, loss.item()    

    def _test_step(self, model, data, split): # test_step是layer, TODO: 这里设计的不好
        with torch.no_grad():
            batch_size, x, adjs, y = data['batch_size'], data['x'], data['adjs'], data['y']
            
            adjs = [adj.to(self.device) for adj in adjs] 
            x, y = x.to(self.device), y.to(self.device)
            
            logits = model(x, adjs)
            loss = model.loss_fn(logits, y)
            acc = model.evaluator(logits, y) / batch_size
        return acc, loss.item() 