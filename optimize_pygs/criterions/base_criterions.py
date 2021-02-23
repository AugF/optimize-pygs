import copy


class EarlyStoppingCriterion(object):
    def __init__(self, patience):
        self.patience = patience
        self.best_model = None

    def should_stop(self, epoch, val_loss, val_acc, model=None):
        raise NotImplementedError

    def after_stopping(self):
        pass

    def reset(self):
        pass
    
    def get_best_model(self):
        return self.best_model

    

