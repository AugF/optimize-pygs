import copy 
import numpy as np
from optimize_pygs.criterions import EarlyStoppingCriterion, register_criterion


@register_criterion("gcn")
class GCNCriterion(EarlyStoppingCriterion):
    def __init__(self, patience=50):
        super().__init__(patience)
        self.val_losses = []

    def should_stop(self, epoch, val_loss, val_acc, model=None):
        self.val_losses.append(val_loss)

        if epoch >= self.patience and self.val_losses[-1] > np.mean(
            self.val_losses[-(self.patience + 1):-1]):
            self.best_model = copy.deepcopy(model.state_dict())
            return True
        return False

    def reset(self):
        self.val_losses = []


@register_criterion("gat")
class GATCriterion(EarlyStoppingCriterion):
    def __init__(self, patience=50):
        super().__init__(patience)
        self.val_acc_max = 0.0
        self.val_loss_min = np.inf
        self.patience_step = 0

    def should_stop(self, epoch, val_loss, val_acc, model=None):
        if val_acc >= self.val_acc_max or val_loss <= self.val_loss_min:
            # either val accuracy or val loss improved
            self.val_acc_max = np.max((val_acc, self.val_acc_max))
            self.val_loss_min = np.min((val_loss, self.val_loss_min))
            self.patience_step = 0
            self.best_model = copy.deepcopy(model.state_dict())
        else:
            self.patience_step += 1

        return self.patience_step >= self.patience

    def reset(self):
        super().reset()
        self.val_acc = 0.0
        self.val_loss_min = np.inf
        self.patience_step = 0


@register_criterion("kdd")
class KDDCriterion(EarlyStoppingCriterion):
    def __init__(self, patience=50):
        super().__init__(patience)
        self.val_loss_min = np.inf
        self.patience_step = 0

    def should_stop(self, epoch, val_loss, val_acc, model=None):
        # only pay attention to validation loss
        if val_loss <= self.val_loss_min:
            # val loss improved
            self.val_loss_min = np.min((val_loss, self.val_loss_min))
            self.patience_step = 0
            self.best_model = copy.deepcopy(model.state_dict())
        else:
            self.patience_step += 1

        return self.patience_step >= self.patience

    def reset(self):
        super().reset()
        self.val_loss_min = np.inf
        self.patience_step = 0


@register_criterion("gat_with_tolerance")
class GATCriterionWithTolerance(GATCriterion):
    def __init__(self, patience=50, tolerance=20):
        super().__init__(patience)
        self.tolerance = tolerance

    def should_stop(self, epoch, val_loss, val_acc, model=None):
        if val_acc >= self.val_acc_max or val_loss <= self.val_loss_min:
            # either val accuracy or val loss improved, so we have a new best state
            self.val_acc_max = np.max((val_acc, self.val_acc_max))
            self.val_loss_min = np.min((val_loss, self.val_loss_min))
            self.best_step = copy.deepcopy(model.state_dict())

            # But only reset patience if accuracy or loss improved by a certain degree. This avoids long-running
            # convergence processes like for the LabelPropagation algorithm.
            if val_acc >= self.val_acc_max + self.tolerance or val_loss <= self.val_loss_min - self.tolerance:
                self.patience_step = 0
            else:
                self.patience_step += 1
        else:
            self.patience_step += 1

        return self.patience_step >= self.patience