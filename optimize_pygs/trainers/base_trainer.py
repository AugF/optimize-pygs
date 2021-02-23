import torch
from tqdm import tqdm

from optimize_pygs.criterions import build_criterion_from_name


class BaseTrainer:
    @staticmethod
    def add_args(parser):
        """Add trainer-specific arguments to the parser."""
        pass
    
    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def __init__(self, args): # must args
        super().__init__()
        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id
        self.patience = args.patience // 5
        self.max_epoch = args.max_epoch
        self.lr = args.lr
        self.weight_decay = args.weight_decay

        self.early_stopping = build_criterion_from_name(args.criterion) # 只保留这俩
        self.optimizer, self.best_model, self.test_acc = None, None, None
    
    def fit(self, model, data, train_loader=None, val_loader=None, optimizer="Adam", infer_layer=False):
        epoch_iter = tqdm(range(self.max_epoch))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.early_stopping.reset()
        for i, epoch in enumerate(epoch_iter):
            train_acc, _ = self.train_step(model, data, train_loader)
            if infer_layer: # 开启infer_flag选项
                val_acc, val_loss = self.infer_step(model, data, split="val", loader=val_loader.get_loader())
            else:
                val_acc, val_loss = self.test_step(model, data, split="val", loader=val_loader)
            epoch_iter.set_description(f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}")
            if self.early_stopping.should_stop(i, val_acc, val_loss, model):
                print("early_stopping ...")
                break

        self.early_stopping.after_stopping()
        # self.best_model = torch.load_state_dict(self.early_stopping.get_best_model())
        self.best_model = self.early_stopping.get_best_model()
        self.best_acc = self.predict(data, val_loader)
        print(f"Final Test: {self.best_acc:.4f}")

    def predict(self, data, loader=None):
        acc, _ = self.test_step(self.best_model, data, split="test", loader=loader)
        return acc
    
    def train_step(self, model, data, loader=None):
        raise NotImplementedError

    def test_step(self, model, data, split, loader=None):
        raise NotImplementedError
    
    def infer_step(self, model, data, split, loader=None):
        return self.test_step(model, data, split, loader)