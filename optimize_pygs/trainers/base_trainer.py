from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    @classmethod
    @abstractmethod
    def build_trainer_from_args(cls, args):
        """raise a new trainer instance"""
        raise NotImplementedError

    @abstractmethod
    def fit(self, model, dataset):
        raise NotImplementedError
    
    
