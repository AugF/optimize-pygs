from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    @classmethod
    @abstractmethod
    def build_trainer_from_args(cls, args):
        """raise a new trainer instance"""
        raise NotImplementedError

    @abstractmethod
    def fit(self, model, dataset): # 训练得到best_model, predict的事交给model
        raise NotImplementedError
    

def FullBatchTrainer(BaseTrainer):
    @staticmethod
    def add_args(parser):
        """Add trainer-specific arguments to the parser."""
        pass
    
    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def __init__(self, ):
        pass
        
    
