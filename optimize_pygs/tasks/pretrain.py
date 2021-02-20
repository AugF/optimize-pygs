import argparse
import torch

from optimize_pygs.tasks.base_task import BaseTask
from optimize_pygs.models import build_model

class PretrainTask(BaseTask):
    @staticmethod
    def add_args(_: argparse.ArgumentParser):
        """Add task-specific arguments to the parser
        """
    
    def __init__(self, args):
        super(PretrainTask, self).__init__(args)
        
        self.device = "cpu" if not torch.cuda.is_availabel() or args.cpu else args.device_id[0]
        self.model = build_model(args)
        self.model = self.model.to(self.device)
    
    def train(self):
        return self.model.trainer.fit()