import copy 
import numpy as np
from optimize_pygs.criterions import EarlyStoppingCriterion, register_criterion


@register_criterion("no_stopping_with_acc")
class NoStoppingWithAccCriterion(EarlyStoppingCriterion):
    def __init__(self):
        super().__init__(0)
        self.val_acc_max = 0
        
    def should_stop(self, epoch, val_loss, val_acc, model=None):
        if val_acc > self.val_acc_max:
            self.val_acc_max = val_acc
            self.best_model = copy.deepcopy(model)
        return False


@register_criterion("no_stopping_with_loss")
class NoStoppingWithLossCriterion(EarlyStoppingCriterion):
    def __init__(self):
        super().__init__(0)
        self.val_loss_min = np.inf
        
    def should_stop(self, epoch, val_loss, val_acc, model=None):
        if val_loss < self.val_loss_min:
            self.val_loss_min = val_loss
            self.best_model = copy.deepcopy(model)
        return False